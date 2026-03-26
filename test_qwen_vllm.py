import os
from vllm import LLM, SamplingParams

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

if __name__ == "__main__":
    # vLLM automatically handles distributed tensor parallelism across both GPUs natively!
    # This bypasses the memory fragmentation and conversion bugs in huggingface `accelerate`.
    llm = LLM(
        model="Qwen/Qwen3.5-35B-A3B-FP8", 
        tensor_parallel_size=2, # Use exactly 2 GPUs
        trust_remote_code=True,
        max_model_len=4096, # restrict max sequence length to save VRAM on KV cache
    )

    prompts = [
        "What are the three primary colors?"
    ]

    sampling_params = SamplingParams(max_tokens=1000)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n--- Result ---\nPrompt: {prompt}\nGenerated text: {generated_text}\n--------------")
