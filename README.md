# Ollama Token Counter & Performance Analyzer

> 🚀 **Note:** This project was **vibe coded using Gemini**.  
> ⚠️ **Disclaimer:** This software is provided for **educational purposes only**. Use at your own risk.

## 🎯 Purpose
The primary purpose of this script is to analyze and understand how **vocabulary and tokenization efficiency affect different LLM sizes** when processing identical input prompts across various local models in Ollama.

A Python utility to measure input (prompt) tokens, output (generated) tokens, processing speed (tokens/sec), and evaluation durations when benchmarking local LLMs running on **Ollama**.

---

## 📦 Installation & Setup

### 1. Prerequisites
* **Python 3.8+**
* **Ollama** installed and running:
  ```bash
  ollama serve
  # or launch the Ollama desktop app
  ```
* Pull a model to test (e.g., `qwen2.5:7b` or `llama3.2`):
  ```bash
  ollama pull qwen2.5:7b
  ```

### 2. Install Requirements (Optional)
`ollama_token_counter.py` runs with **zero external dependencies** using Python's standard library. 

If you plan to use the official SDK script (`example_ollama_sdk.py`), install the requirements via:
```bash
pip install -r requirements.txt
```

---

### 1. Native Model Token Counter (Recommended)
Run `ollama_native_token_counter.py` to use model-native tokenization via Ollama's `/api/tokenize` endpoint and model-aware BPE encodings, along with an explicit discrepancy analysis comparing prompt mapping against Ollama's evaluated prompt tokens (`prompt_eval_count`):

```bash
# Interactive mode
python3 ollama_native_token_counter.py

# CLI mode with specific model and prompt
python3 ollama_native_token_counter.py -m qwen2.5:14b -p "Explain quantum computing in 2 simple sentences."
```

---

### 2. Standard Token Counter
Run `ollama_token_counter.py` for standalone token and performance benchmarking:

```bash
python3 ollama_token_counter.py -m llama3.2 -p "Explain quantum computing in 2 simple sentences."
```

---

### 3. Using Official Ollama Python SDK
If you prefer using `pip install ollama`:

```bash
pip install ollama
python3 example_ollama_sdk.py
```


---

## 📋 Command Line Options

| Argument | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--prompt` | `-p` | Input prompt text. Triggers CLI single-run mode. | Interactive Mode |
| `--model` | `-m` | Ollama model name (e.g., `llama3.2`, `mistral`, `qwen2.5`). | Auto-detected / `llama3.2` |
| `--host` | | Ollama server host URL | `http://localhost:11434` |
| `--system` | `-s` | Optional system prompt instructions | `None` |
| `--no-stream` | | Disable real-time output streaming | Streaming Enabled |
| `--help` | `-h` | Show help message and exit | |

---

## 📊 Sample Output

```text
Response: Quantum computing uses the principles of quantum mechanics to process information...

+--------------------------------------------------------+
|           OLLAMA TOKEN & PERFORMANCE METRICS           |
+--------------------------------------------------------+
| Model Name           : qwen2.5:14b                      |
| Architecture Family  : qwen2                            |
| Parameter Size       : 14.8B                            |
| Quantization Level   : Q4_K_M                           |
| Vocabulary Size      : 151,646                          |
| Context Length       : 32,768                           |
+--------------------------------------------------------+
| Input Character Length: 48 chars (7 words)              |
| Input Tokens (Prompt)  : 39                             |
| Input Efficiency     : 1.23 chars/token                |
+--------------------------------------------------------+
| Output Character Length: 335 chars (46 words)           |
| Output Tokens (Result) : 55                             |
| Output Efficiency    : 6.09 chars/token                |
| Total Tokens          : 94                             |
+--------------------------------------------------------+
| Prompt Eval Duration : 245.50 ms                        |
| Prompt Processing    : 158.86 t/s                       |
| Response Duration    : 1.79 s                           |
| Generation Speed       : 30.71 t/s                      |
| Total Ollama Time    : 2.14 s                           |
| Client Wall Clock    : 2.14 s                           |
+--------------------------------------------------------+

--- Input Token Number Mappings ---
Prompt Length: 48 characters | 7 words

Token Number Mapping Table (BPE Encoding: cl100k_base):
---------------------------------------------------------------
Token #  | Token ID     | Token String Piece         | Char Len
---------------------------------------------------------------
1        | #849         | Ex                         | 2       
2        | #21435       | plain                      | 5       
3        | #31228       |  quantum                   | 8       
4        | #25213       |  computing                 | 10      
5        | #304         |  in                        | 3       
6        | #220         |                            | 1       
7        | #17          | 2                          | 1       
8        | #4382        |  simple                    | 7       
9        | #23719       |  sentences                 | 10      
10       | #13          | .                          | 1       
---------------------------------------------------------------
```

---

## 💡 How Token Counting Works
Ollama reports exact model statistics in its `/api/generate` and `/api/chat` endpoints:
* **`prompt_eval_count`**: Total number of tokens processed during the prefill phase (includes your input prompt + Ollama model chat template tags, BOS/EOS tokens, and system prompts).
* **`eval_count`**: Number of tokens generated in the output (generation phase).
* **`prompt_eval_duration`**: Time spent evaluating the prompt (in nanoseconds).
* **`eval_duration`**: Time spent generating tokens (in nanoseconds).

### 🔍 Why Mapping Table Token Count May Differ from `prompt_eval_count`
1. **Model Chat Templates**: Ollama automatically wraps prompts in model-specific control tags (e.g. `<|begin_of_text|>`, `<|start_header_id|>user<|end_header_id|>`), which adds extra tokens to `prompt_eval_count`.
2. **Tokenizer Vocabulary**: Model-native tokenizers (used by Ollama and `ollama_native_token_counter.py`) split sub-words differently than general client encoders like `tiktoken` or whitespace splitters.

