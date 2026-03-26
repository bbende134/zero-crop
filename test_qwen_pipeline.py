import os
import torch
from transformers import pipeline

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

pipe = pipeline(
    "image-text-to-text", 
    model="Qwen/Qwen3.5-35B-A3B-FP8", 
    device_map="auto", 
    torch_dtype="auto",
    max_memory={0: "28GiB", 1: "28GiB"}
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
result = pipe(text=messages)
print(result)
