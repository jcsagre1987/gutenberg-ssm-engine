import os
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# 1. Load a standard 32k-compatible tokenizer (matching Llama/32k architecture)
print("Loading tokenizer...")
try:
    # Using an open-access 32k tokenizer compatible with 32k indices
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", trust_remote_code=True)
except Exception:
    # Fallback to standard fast tokenizer if offline/unavailable
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 

# 2. Load the quantized ONNX SSM model
print("Loading ONNX model...")
session = ort.InferenceSession("multimodal_ssm_0.75b_quant.onnx", providers=["CPUExecutionProvider"])

# 3. Encode prompt safely within 32k bounds
prompt = "Hello world"
input_ids = tokenizer.encode(prompt)
# Ensure all tokens are strictly clamped to the 0–31999 range
safe_ids = [min(max(tid, 0), 31999) for tid in input_ids]

print(f"Prompt: '{prompt}'")
generated_ids = list(safe_ids)

# 4. Generation loop
for _ in range(20):
    ort_inputs = {session.get_inputs()[0].name: [generated_ids]}
    logits = session.run(None, ort_inputs)[0]
    
    next_token_logits = logits[0, -1, :]
    next_token = int(next_token_logits.argmax())
    
    # Clamp next token to safe 32k range
    next_token = min(max(next_token, 0), 31999)
    
    if next_token == tokenizer.eos_token_id or next_token == 0:
        break
    generated_ids.append(next_token)

# 5. Decode output back to real text
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("\nGenerated Output:")
print(output_text)