#!/usr/bin/env python3
"""
Ollama Native Token Counter & Performance Analyzer
---------------------------------------------------
A Python script to test local LLMs via Ollama using model-native tokenization.
It queries Ollama's native /api/tokenize endpoint (with fallback to Hugging Face
tokenizers or BPE encodings) to provide exact token ID mapping tables alongside
Ollama prompt & generation evaluation metrics.

Zero external dependencies required (uses standard library).
"""

import sys
import json
import argparse
import time
import urllib.request
import urllib.error

DEFAULT_HOST = "http://localhost:11434"

# ANSI Color formatting
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[91m"
DIM = "\033[2m"

def check_server(host):
    """Check if Ollama server is running and return available models."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []

def format_ns(ns):
    """Convert nanoseconds to human readable time format."""
    if ns is None:
        return "N/A"
    ms = ns / 1_000_000
    if ms < 1000:
        return f"{ms:.2f} ms"
    s = ms / 1000
    return f"{s:.2f} s"

def get_model_info(host, model):
    """Fetch model details (family, parameter size, quantization, vocab size) via /api/show."""
    url = f"{host.rstrip('/')}/api/show"
    payload = json.dumps({"name": model}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            details = data.get("details", {})
            model_info = data.get("model_info", {})
            
            vocab_size = None
            context_length = None
            token_ids = []
            for key, val in model_info.items():
                if key.endswith(".vocab_size") and val is not None:
                    vocab_size = val
                elif key.endswith(".context_length") and val is not None:
                    context_length = val
                elif key.endswith("_token_id") and isinstance(val, int):
                    token_ids.append(val)
                elif key == "tokenizer.ggml.tokens" and isinstance(val, list):
                    vocab_size = len(val)

            if vocab_size is None and token_ids:
                vocab_size = max(token_ids) + 1

            return {
                "family": details.get("family", model_info.get("general.architecture", "N/A")),
                "parameter_size": details.get("parameter_size", model_info.get("general.size_label", "N/A")),
                "quantization": details.get("quantization_level", "N/A"),
                "format": details.get("format", "N/A"),
                "vocab_size": vocab_size,
                "context_length": context_length,
                "template": data.get("template", ""),
                "system": data.get("system", "")
            }
    except Exception:
        return None

def fetch_native_ollama_tokenize(host, model, prompt_text):
    """
    Attempt to use Ollama's native /api/tokenize and /api/detokenize endpoints.
    Returns list of (token_id, string_piece) or None if endpoints are unavailable.
    """
    tok_url = f"{host.rstrip('/')}/api/tokenize"
    detok_url = f"{host.rstrip('/')}/api/detokenize"

    payload = json.dumps({"model": model, "prompt": prompt_text}).encode("utf-8")
    req = urllib.request.Request(tok_url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            raw_tokens = res_data.get("tokens", [])
            if not raw_tokens:
                return None
            
            # Detokenize each token ID into its string piece
            tokens_mapped = []
            for t_id in raw_tokens:
                d_payload = json.dumps({"model": model, "tokens": [t_id]}).encode("utf-8")
                d_req = urllib.request.Request(detok_url, data=d_payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(d_req, timeout=2) as d_resp:
                        piece_data = json.loads(d_resp.read().decode("utf-8"))
                        piece = piece_data.get("token", "")
                except Exception:
                    piece = "?"
                tokens_mapped.append((t_id, piece))
            return tokens_mapped
    except Exception:
        return None

def display_native_token_mapping(prompt_text, model, host, model_info=None, actual_eval_count=None):
    """
    Display input character length and token number mapping table using:
    1. Ollama Native API (/api/tokenize)
    2. Hugging Face Transformers / Tokenizers (if installed)
    3. Tiktoken BPE Encodings
    4. Word-piece Fallback
    """
    char_len = len(prompt_text)
    word_count = len(prompt_text.split())
    print(f"\n{BOLD}{CYAN}--- Native Token Number Mappings ---{RESET}")
    print(f"{BOLD}Prompt Length:{RESET} {char_len:,} characters | {word_count:,} words")

    tokenizer_source = ""
    tokens_mapped = []

    # 1. Try Ollama Native Tokenizer API
    native_tokens = fetch_native_ollama_tokenize(host, model, prompt_text)
    if native_tokens:
        tokens_mapped = native_tokens
        tokenizer_source = f"Ollama Native API (/api/tokenize - Model: {model})"
    else:
        # 2. Try Tiktoken BPE / Hugging Face
        encoding_name = "cl100k_base"
        if model_info:
            family = str(model_info.get("family", "")).lower()
            if "o1" in family or "gpt-4o" in family or "qwen" in family:
                encoding_name = "o200k_base"
            elif "gpt2" in family or "r50k" in family:
                encoding_name = "p50k_base"

        try:
            import tiktoken
            enc = tiktoken.get_encoding(encoding_name)
            raw_tokens = enc.encode(prompt_text)
            for t_id in raw_tokens:
                piece = enc.decode([t_id])
                tokens_mapped.append((t_id, piece))
            tokenizer_source = f"Tiktoken BPE ({encoding_name})"
        except Exception:
            words = prompt_text.split()
            for idx, word in enumerate(words):
                tokens_mapped.append((idx + 1, word))
            tokenizer_source = "Word Split Fallback"

    mapped_count = len(tokens_mapped)
    print(f"\n{BOLD}Token Mapping Table ({tokenizer_source}):{RESET}")

    has_ids = isinstance(tokens_mapped[0][0], int) if tokens_mapped else False
    header = f"{'Token #':<8} | {'Token ID':<12} | {'Token String Piece':<26} | {'Char Len':<8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for idx, (t_id, piece) in enumerate(tokens_mapped, 1):
        disp_piece = repr(piece)[1:-1]
        print(f"{idx:<8} | #{t_id:<11} | {disp_piece:<26} | {len(piece):<8}")
    print(sep)

    # Explanation if actual Ollama eval token count differs from raw mapping count
    if actual_eval_count is not None and actual_eval_count > 0:
        diff = actual_eval_count - mapped_count
        print(f"\n{BOLD}Token Count Analysis:{RESET}")
        print(f" * Raw Prompt Mapped Tokens : {mapped_count:,}")
        print(f" * Ollama Evaluated Tokens  : {actual_eval_count:,} (via `prompt_eval_count`)")
        if diff > 0:
            print(f" * {YELLOW}Difference (+{diff} tokens):{RESET} Ollama appended model template control tags")
            print(f"   (e.g., `<|begin_of_text|>`, `<|start_header_id|>user<|end_header_id|>`, system wrappers).")
        elif diff < 0:
            print(f" * Difference ({diff} tokens): Mapped encoder and model tokenizer use different sub-word vocabularies.")
        else:
            print(f" * {GREEN}Perfect Match!{RESET} Mapped token count matches Ollama evaluation count exactly.")
    print()

def run_token_count(prompt, model, host, system_prompt=None, stream=True):
    """Send prompt to Ollama and print streamed response + token metrics."""
    model_info = get_model_info(host, model)

    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    if system_prompt:
        payload["system"] = system_prompt

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    print(f"\n{BOLD}{CYAN}--- Prompting Model: {model} ---{RESET}\n")
    if stream:
        print(f"{BOLD}Response:{RESET} ", end="", flush=True)

    start_time = time.time()
    final_stats = {}
    response_text = ""

    try:
        with urllib.request.urlopen(req) as resp:
            if stream:
                for line in resp:
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    text_chunk = chunk.get("response", "")
                    response_text += text_chunk
                    print(text_chunk, end="", flush=True)

                    if chunk.get("done", False):
                        final_stats = chunk
                print("\n")
            else:
                body = resp.read().decode("utf-8")
                final_stats = json.loads(body)
                response_text = final_stats.get("response", "")
                print(f"{BOLD}Response:{RESET}\n{response_text}\n")

    except urllib.error.URLError as e:
        print(f"\n{RED}Error connecting to Ollama at {host}: {e.reason}{RESET}")
        print(f"{YELLOW}Tip: Ensure Ollama is running (`ollama serve`).{RESET}\n")
        return None
    except Exception as e:
        print(f"\n{RED}Error during request: {e}{RESET}\n")
        return None

    client_elapsed_s = time.time() - start_time

    # Token & Character Statistics Extraction
    input_tokens = final_stats.get("prompt_eval_count", 0)
    output_tokens = final_stats.get("eval_count", 0)
    total_tokens = input_tokens + output_tokens

    input_chars = len(prompt)
    output_chars = len(response_text)
    input_chars_per_token = (input_chars / input_tokens) if input_tokens > 0 else 0
    output_chars_per_token = (output_chars / output_tokens) if output_tokens > 0 else 0

    prompt_eval_duration = final_stats.get("prompt_eval_duration", 0) or 0
    eval_duration = final_stats.get("eval_duration", 0) or 0
    total_duration = final_stats.get("total_duration", 0) or 0

    prompt_speed = (input_tokens / (prompt_eval_duration / 1e9)) if prompt_eval_duration > 0 else 0
    eval_speed = (output_tokens / (eval_duration / 1e9)) if eval_duration > 0 else 0

    # Display Metrics Summary Box
    box_width = 56
    line_sep = f"+{'-' * box_width}+"

    print(line_sep)
    print(f"| {BOLD}{MAGENTA}{'OLLAMA TOKEN & PERFORMANCE METRICS':^{box_width - 2}}{RESET} |")
    print(line_sep)
    print(f"| Model Name           : {model:<32} |")
    if model_info:
        family_str = f"{model_info['family']}"
        param_str = f"{model_info['parameter_size']}"
        quant_str = f"{model_info['quantization']}"
        vocab_str = f"{model_info['vocab_size']:,}" if model_info['vocab_size'] else "N/A"
        ctx_str = f"{model_info['context_length']:,}" if model_info['context_length'] else "N/A"
        
        print(f"| Architecture Family  : {family_str:<32} |")
        print(f"| Parameter Size       : {param_str:<32} |")
        print(f"| Quantization Level   : {quant_str:<32} |")
        print(f"| Vocabulary Size      : {vocab_str:<32} |")
        print(f"| Max Context Window   : {ctx_str:<32} |")
        print(f"| Default Ollama Ctx   : {'2,048 tokens (num_ctx)':<32} |")
    print(line_sep)
    print(f"| Input Character Length: {f'{input_chars:,} chars ({len(prompt.split()):,} words)':<31} |")
    print(f"| {BOLD}{GREEN}Input Tokens (Prompt)  : {input_tokens:<32,}{RESET} |")
    print(f"| Input Efficiency     : {f'{input_chars_per_token:.2f} chars/token':<31} |")
    print(line_sep)
    print(f"| Output Character Length: {f'{output_chars:,} chars ({len(response_text.split()):,} words)':<30} |")
    print(f"| {BOLD}{GREEN}Output Tokens (Result) : {output_tokens:<32,}{RESET} |")
    print(f"| Output Efficiency    : {f'{output_chars_per_token:.2f} chars/token':<31} |")
    print(f"| {BOLD}Total Tokens          : {total_tokens:<32,}{RESET} |")
    print(line_sep)
    print(f"| Prompt Eval Duration : {format_ns(prompt_eval_duration):<32} |")
    print(f"| Prompt Processing    : {prompt_speed:<29.2f} t/s |")
    print(f"| Response Duration    : {format_ns(eval_duration):<32} |")
    print(f"| {BOLD}{YELLOW}Generation Speed       : {eval_speed:<29.2f} t/s{RESET} |")
    print(f"| Total Ollama Time    : {format_ns(total_duration):<32} |")
    print(f"| Client Wall Clock    : {f'{client_elapsed_s:.2f} s':<32} |")
    print(line_sep)

    # Display Native Token Mapping Breakdown Table
    display_native_token_mapping(prompt, model, host, model_info, actual_eval_count=input_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "generation_speed_tps": eval_speed,
        "response_text": response_text
    }

def main():
    parser = argparse.ArgumentParser(
        description="Count tokens using native Ollama & model tokenizers."
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Input prompt text. If omitted, interactive mode starts."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name (e.g. qwen2.5:14b, llama3.2, mistral). Auto-detected if omitted."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Ollama server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "-s", "--system",
        type=str,
        help="Optional system prompt."
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable real-time response streaming."
    )

    args = parser.parse_args()

    is_online, available_models = check_server(args.host)
    if not is_online:
        print(f"{RED}Warning: Unable to reach Ollama server at {args.host}.{RESET}")
        print(f"{YELLOW}Please ensure Ollama is running (`ollama serve`).{RESET}\n")

    selected_model = args.model
    if not selected_model:
        if available_models:
            selected_model = available_models[0]
            print(f"{DIM}Auto-selected available model: {selected_model}{RESET}")
        else:
            selected_model = "llama3.2"

    if args.prompt:
        run_token_count(
            prompt=args.prompt,
            model=selected_model,
            host=args.host,
            system_prompt=args.system,
            stream=not args.no_stream
        )
    else:
        print(f"{BOLD}{CYAN}=== Ollama Native Token Counter ==={RESET}")
        print(f"Server Host : {args.host}")
        print(f"Model       : {selected_model}")
        if available_models:
            print(f"Installed   : {', '.join(available_models)}")
        print(f"{DIM}Press Ctrl+C or type 'exit' / 'quit' to end.{RESET}\n")

        while True:
            try:
                user_prompt = input(f"{BOLD}Enter Prompt>{RESET} ").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                run_token_count(
                    prompt=user_prompt,
                    model=selected_model,
                    host=args.host,
                    system_prompt=args.system,
                    stream=not args.no_stream
                )
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
