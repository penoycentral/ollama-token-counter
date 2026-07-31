#!/usr/bin/env python3
"""
Minimal Ollama Token Counter using official Python SDK (`pip install ollama`).
Displays model information (family, parameter size, quantization, vocabulary size) along with token counts.
"""

try:
    import ollama
except ImportError:
    print("The `ollama` python library is not installed.")
    print("Install it with: pip install ollama")
    print("Or use `ollama_token_counter.py` which uses standard library.")
    exit(1)

def main():
    model = "llama3.2"  # Replace with your local model name
    prompt = "Explain quantum computing in 2 simple sentences."

    # Fetch model metadata via SDK
    try:
        info = ollama.show(model)
        details = info.get('details', {})
        model_info = info.get('model_info', {})

        vocab_size = None
        for k, v in model_info.items():
            if k.endswith('.vocab_size'):
                vocab_size = v

        print(f"Model Name           : {model}")
        print(f"Architecture Family  : {details.get('family', 'N/A')}")
        print(f"Parameter Size       : {details.get('parameter_size', 'N/A')}")
        print(f"Quantization Level   : {details.get('quantization_level', 'N/A')}")
        if vocab_size:
            print(f"Vocabulary Size      : {vocab_size:,}")
        print("-" * 50)
    except Exception as e:
        print(f"Could not fetch metadata for {model}: {e}\n")

    print(f"Sending prompt to local Ollama LLM ({model})...\n")

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    print("Response:")
    print(response['message']['content'])
    print("\n" + "="*50)
    
    # Token Counts
    input_tokens = response.get('prompt_eval_count', 0)
    output_tokens = response.get('eval_count', 0)
    total_tokens = input_tokens + output_tokens

    print(f"Input Tokens  (Prompt) : {input_tokens}")
    print(f"Output Tokens (Result) : {output_tokens}")
    print(f"Total Tokens           : {total_tokens}")
    print("="*50)

if __name__ == "__main__":
    main()
