#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cstdint>

class SimpleSSMTokenizer {
private:
    std::unordered_map<int64_t, std::string> id_to_token;
    std::unordered_map<std::string, int64_t> token_to_id;

public:
    bool load_vocab(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) return false;

        std::string line;
        int64_t id = 0;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back(); // Handle Windows newline characters
            }
            id_to_token[id] = line;
            token_to_id[line] = id;
            id++;
        }
        return true;
    }

    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> tokens;
        
        // Simple word-level/subword lookup matching vocab.txt
        std::string current_word = "";
        auto add_token_candidate = [&](std::string word) {
            if (word.empty()) return;
            // GPT-2 vocabulary often prefixes words with 'Ġ' for spaces
            std::string spaced_word = "\xC4\xA0" + word; 
            if (token_to_id.find(spaced_word) != token_to_id.end()) {
                tokens.push_back(token_to_id.at(spaced_word));
            } else if (token_to_id.find(word) != token_to_id.end()) {
                tokens.push_back(token_to_id.at(word));
            } else {
                tokens.push_back(0); // Fallback token ID if unknown
            }
        };

        for (char c : text) {
            if (c == ' ') {
                add_token_candidate(current_word);
                current_word = "";
            } else {
                current_word += c;
            }
        }
        add_token_candidate(current_word);

        if (tokens.empty()) {
            tokens.push_back(15496); // Default fallback ("Hello")
        }

        return tokens;
    }

    std::string decode(const std::vector<int64_t>& token_ids) const {
        std::string result = "";
        for (auto id : token_ids) {
            auto it = id_to_token.find(id);
            if (it != id_to_token.end()) {
                std::string token_str = it->second;
                
                // Replace GPT-2 space marker 'Ġ' (UTF-8: \xC4\xA0) with regular spaces
                std::string cleaned = "";
                for (size_t i = 0; i < token_str.size(); ++i) {
                    if (i + 1 < token_str.size() && 
                        (unsigned char)token_str[i] == 0xC4 && 
                        (unsigned char)token_str[i+1] == 0xA0) {
                        cleaned += " ";
                        i++;
                    } else {
                        cleaned += token_str[i];
                    }
                }
                result += cleaned;
            }
        }
        return result;
    }
};