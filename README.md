## Author
Created and maintained by **Juan Carlo R. Sagre**.

## License
This project is open-source under the **GNU General Public License v2.0 (GPLv2)**. See the [LICENSE](LICENSE) file for details.

# Gutenberg SSM Edge Engine 🚀

A high-performance, resource-efficient local AI engine running natively on CPU via `llama.cpp` and FastAPI. Designed for edge devices (such as Android and Raspberry Pi) with zero Python loop overhead.

## Features
* **Native C++ Inference:** Bypasses heavy PyTorch frameworks using 4-bit GGUF quantization (`mamba-2.8b`).
* **Auto-Intent Routing:** Dynamically adjusts temperature and context limits based on whether the prompt requires strict factual recall or creative writing.
* **Responsive Web UI:** Clean, modern chat interface with real-time connection monitoring and cross-device compatibility.
* **Anti-Loop Protection:** Built-in repetition penalty and C++ stop-word filtering to prevent token hallucinations.

## 🚀 v2.0 Key Architecture Upgrades
* **Deep Context RAG:** Expanded text extraction to 800 characters, prioritizing DuckDuckGo for live news and utilizing a direct Wikipedia API fallback to bypass local DNS hijacking.
* **Temporal Query Normalization:** Dynamically maps conversational dates (e.g., "2016 to 2022") and relative historical spans ("prior to", "predecessor") into strict search engine targets.
* **Anti-Sycophancy Guardrails:** System prompts engineered to force absolute factual extraction, explicitly forbidding the model from agreeing with false user premises or hallucinating dual-officeholders.
* **Zero Token Drift:** Implemented explicit `\n` stop tokens to instantly terminate the generation stream the moment a factual entity is extracted, eliminating runaway hallucinations.

## Quick Start (Windows)
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/gutenberg-ssm-engine.git](https://github.com/your-username/gutenberg-ssm-engine.git)
   cd gutenberg-ssm-engine