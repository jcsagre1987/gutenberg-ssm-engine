import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

session = ort.InferenceSession("multimodal_ssm_0.75b_quant.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

prompt = "The sky is"
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", trust_remote_code=True)
input_ids = tokenizer.encode(prompt, add_special_tokens=True)

logits = session.run(None, {input_name: [input_ids]})[0]
last_logits = logits[0, -1, :]

print(f"Logit Tensor Shape:  {last_logits.shape}")
print(f"Min Logit Value:    {np.min(last_logits)}")
print(f"Max Logit Value:    {np.max(last_logits)}")
print(f"Mean Logit Value:   {np.mean(last_logits)}")
print(f"Unique Values Count: {len(np.unique(last_logits))}")