"""
Script for benchmarking the Transformer model with and without torch.compile.
"""
import time
import torch
from src.config import Config
from src.model.model import GenerateInput, SamplingConfig
from src.utils.core import load_model_and_tokenizer, main_entrypoint

def run_benchmark(model_name: str, use_compile: bool):
    """
    Runs a benchmark on the specified model, optionally with torch.compile.

    Args:
        model_name: The name of the model to benchmark.
        use_compile: Whether to use torch.compile.
    """
    config = Config.from_json(f"models/{model_name}/config.json")
    config.hardware.torch_compile = use_compile

    model, tokenizer = load_model_and_tokenizer(
        model_name, config, load_in_4bit=False, quantized=False
    )

    prompt = "Once upon a time"
    input_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_tokens], device=model.device)

    gen_input = GenerateInput(
        start_tokens=input_tensor,
        max_new_tokens=50,
        sampling_config=SamplingConfig(temperature=0.0) # Greedy sampling
    )

    # Warm-up run
    for _, _ in model.generate(gen_input):
        pass

    # Benchmark run
    start_time = time.perf_counter()
    num_tokens = 0
    for chunk, _ in model.generate(gen_input):
        num_tokens += chunk.size(1)
    end_time = time.perf_counter()

    duration = end_time - start_time
    tokens_per_second = num_tokens / duration

    compile_status = "Enabled" if use_compile else "Disabled"
    print(f"--- torch.compile: {compile_status} ---")
    print(f"Generated {num_tokens} tokens in {duration:.2f} seconds.")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print("-" * 30)

@main_entrypoint
def main():
    """Main benchmark script."""
    # Assuming 'test_model' exists from previous runs.
    # In a real scenario, you might want to train a small model first.
    model_name = "test_model"

    # Benchmark without torch.compile
    run_benchmark(model_name, use_compile=False)

    # Benchmark with torch.compile
    run_benchmark(model_name, use_compile=True)

if __name__ == "__main__":
    main()
