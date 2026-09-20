#include <iostream>
#include <vector>
#include <algorithm>
#include "crow_all.h"
#include "onnxruntime_cxx_api.h"
#include "tokenizer.h"

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SSM_Engine");
        Ort::SessionOptions session_options;
        // Optimized for Ryzen 5 4600G physical core count
        session_options.SetIntraOpNumThreads(6);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "Loading multimodal_ssm_0.75b_int8.onnx...\n";
        Ort::Session session(env, L"multimodal_ssm_0.75b_int8.onnx", session_options);

        SimpleSSMTokenizer tokenizer;
        if (tokenizer.load_vocab("vocab.txt")) {
            std::cout << "Vocabulary loaded successfully.\n";
        } else {
            std::cout << "Warning: vocab.txt not found. String decoding will be skipped.\n";
        }

        crow::SimpleApp app;

        CROW_ROUTE(app, "/health")([](){
            return "SSM C++ Engine Online & Model Ready";
        });

        CROW_ROUTE(app, "/v1/chat/completions").methods("POST"_method)
        ([&session, &tokenizer](const crow::request& req){
            auto json = crow::json::load(req.body);
            if (!json || !json.has("prompt")) {
                return crow::response(400, "{\"error\": \"Missing 'prompt' field\"}");
            }

            std::string prompt = json["prompt"].s();
            std::vector<int64_t> current_ids = tokenizer.encode(prompt);

            const char* input_names[] = {"input_ids"};
            const char* output_names[] = {"logits"};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            int max_new_tokens = 20;
            std::vector<int64_t> generated_token_ids;

            for (int step = 0; step < max_new_tokens; ++step) {
                // Enforce exact 8-token window slicing for ONNX input shape (1, 8)
                std::vector<int64_t> window_ids;
                if (current_ids.size() >= 8) {
                    window_ids.assign(current_ids.end() - 8, current_ids.end());
                } else {
                    window_ids.resize(8 - current_ids.size(), 0); // Left-pad with 0
                    window_ids.insert(window_ids.end(), current_ids.begin(), current_ids.end());
                }

                std::vector<int64_t> input_shape = {1, 8};
                Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info, window_ids.data(), window_ids.size(), input_shape.data(), input_shape.size()
                );

                auto output_tensors = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names, &input_tensor, 1,
                    output_names, 1
                );

                float* logits = output_tensors[0].GetTensorMutableData<float>();
                auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                int64_t seq_len = shape[1];
                int64_t vocab_size = shape[2];
                int64_t offset = (seq_len - 1) * vocab_size;

                int64_t next_token = 0;
                float max_val = -1e9f;
                for (int64_t v = 0; v < vocab_size; ++v) {
                    if (logits[offset + v] > max_val) {
                        max_val = logits[offset + v];
                        next_token = v;
                    }
                }

                if (next_token == 102 || next_token == 0) break;
                current_ids.push_back(next_token);
                generated_token_ids.push_back(next_token);
            }

            crow::json::wvalue res;
            res["status"] = "success";
            res["tokens_processed"] = current_ids.size();
            res["response"] = tokenizer.decode(generated_token_ids);
            
            crow::json::wvalue::list token_list;
            for (auto id : generated_token_ids) {
                token_list.push_back(id);
            }
            res["generated_token_ids"] = std::move(token_list);
            return crow::response(res);
        });

        std::cout << "Server listening on http://localhost:18080\n";
        app.port(18080).multithreaded().run();
    } catch (const std::exception& e) {
        std::cerr << "Engine Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}