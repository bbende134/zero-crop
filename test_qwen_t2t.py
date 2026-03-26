import os
import torch
from transformers import pipeline

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Use the text-generation pipeline for pure text-to-text (T2T) tasks
pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen3.5-35B-A3B-FP8", 
    device_map="auto", 
    quantization_config=quantization_config,
    max_memory={0: "28GiB", 1: "28GiB"}
)

messages = [
    {"role": "system", "content": "You are a helpful and concise assistant."},
    {"role": "user", "content": "What are the three primary colors?"}
]

# Pass the messages directly; the pipeline will automatically apply the model's chat template
output = pipe(messages, max_new_tokens=100)

# Print the generated response
print(output[0]['generated_text'][-1]['content'])
