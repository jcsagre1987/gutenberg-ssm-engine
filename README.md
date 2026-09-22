## Author
Created and maintained by **Juan Carlo R. Sagre**.

## License
This project is open-source under the **GNU General Public License v2.0 (GPLv2)**. See the [LICENSE](LICENSE) file for details.

# Gutenberg SSM Edge Engine 🚀

A high-performance, resource-efficient local AI engine running natively on CPU via `llama.cpp` and FastAPI. Designed for edge devices and local desktop deployment with zero Python loop overhead.

## Features
* **Native C++ Inference:** Bypasses heavy PyTorch frameworks using 4-bit GGUF quantization (`mamba-2.8b`).
* **Auto-Intent Routing:** Dynamically separates casual social chat from factual RAG and structured queries.
* **Responsive Web UI:** Clean, modern chat interface with real-time connection monitoring and cross-device compatibility.
* **Anti-Loop Protection:** Built-in repetition penalty and C++ stop-word filtering to prevent token hallucinations.

## 🚀 v2.1 Key Architecture Upgrades
* **JSON Knowledge Base Auto-Sync:** Centralized fact definitions in `knowledgebasereference.json` that automatically sync into SQLite upon server startup.
* **Precision RAG Filter Layer:** Programmatically filters out web noise, historical filler, and cross-contamination sentences before ingestion.
* **Social Routing Engine:** Seamlessly handles casual greetings and chat interactions via structured JSON persona definitions without workflow interference.
* **Comprehensive Guides:** Includes dedicated `INSTALL.txt` and `USAGE.txt` documentation files for quick setup and deployment.

## Documentation
* **Installation:** See [INSTALL.txt](INSTALL.txt) for step-by-step setup instructions and dependency requirements.
* **Usage Guide:** See [USAGE.txt](USAGE.txt) for server startup procedures and interaction modes.

## Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/gutenberg-ssm-engine.git](https://github.com/your-username/gutenberg-ssm-engine.git)
   cd gutenberg-ssm-engine