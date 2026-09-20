import onnxruntime as ort

model_path = "multimodal_ssm_0.75b_int8.onnx"

print("=" * 60)
print(f"INSPECTING: {model_path}")
print("=" * 60)

try:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 6
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    
    print("\n--- MODEL INPUTS ---")
    for inp in session.get_inputs():
        print(f"Name: {inp.name:<15} | Shape: {str(inp.shape):<15} | Type: {inp.type}")

    print("\n--- MODEL OUTPUTS ---")
    for o in session.get_outputs():
        print(f"Name: {o.name:<15} | Shape: {str(o.shape):<15} | Type: {o.type}")
        
    print("\n[SUCCESS] Text engine loaded cleanly and is ready for inference!")
except Exception as e:
    print(f"Error loading {model_path}: {e}")