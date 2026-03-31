"""
Generate natural-language query augmentations for flow_3 classes.

Uses a local LLM (via transformers) to produce the kind of sentences a user
would actually type when searching for a land-use/crop class — short queries,
synonyms, regional names, colloquial descriptions, etc.

Output: data_corine/query_augmentations.jsonl
Same format as corine_wiki_char_count.jsonl so flow_3 can load it directly.

Usage:
    uv run python data_processing/generate_query_augmentation.py
    uv run python data_processing/generate_query_augmentation.py --model Qwen/Qwen3-14B
"""

import json
import re
import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data_corine"


def build_class_info():
    """Load class names and descriptions from both HRL and CORINE sources."""
    classes = {}

    hrl_path = DATA_DIR / "hrl_classes_includes.json"
    if hrl_path.exists():
        with open(hrl_path) as f:
            for code, desc in json.load(f).items():
                classes[code] = desc

    corine_path = DATA_DIR / "corine_classes_includes.json"
    if corine_path.exists():
        with open(corine_path) as f:
            for code, desc in json.load(f).items():
                classes[code] = desc

    return classes


def build_messages(class_name, description):
    """Build chat messages for query generation."""
    return [
        {"role": "system", "content": "You are a helpful assistant that generates search queries."},
        {"role": "user", "content": f"""Context: I have a geospatial model that maps text queries to land-use density maps over Hungary.
The model knows {class_name} as one of its classes.

Class name: "{class_name}"
Full description: "{description}"

Task: Generate 30 diverse text queries that a user might type when looking for this land-use class on a map. Include:

1. Short keyword queries (1-3 words): "wheat fields", "lakes", "forest"
2. Natural language questions: "where is wheat grown in Hungary?"
3. Synonyms and alternate names: "corn" for maize, "rapeseed" for canola
4. Regional/local references if applicable: "Lake Balaton", "Danube wetlands"
5. Colloquial descriptions: "golden grain fields", "flooded paddies"
6. Queries mentioning related concepts: "bread cereal crops", "oil seed plants"
7. Negation-free descriptive phrases: "areas covered by shallow standing water"
8. Mixed-language or scientific terms if relevant

Rules:
- Each query on its own line, numbered 1-30
- No explanations, just the queries
- Vary length: some very short (1-2 words), some medium (5-10 words), some longer
- Focus on what a REAL USER would type, not Wikipedia-style prose
- Include at least 5 very short (1-3 word) queries
- For Hungary-specific classes, include Hungarian geographic references"""},
    ]


def parse_queries(generated_text):
    """Extract individual queries from LLM output."""
    queries = []
    for line in generated_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-•]\s*", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        if cleaned and 3 < len(cleaned) < 200:
            queries.append(cleaned)
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_path = args.output or str(DATA_DIR / "query_augmentations.jsonl")

    classes = build_class_info()
    print(f"Loaded {len(classes)} classes")

    # Resumability
    done_codes = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                done_codes.add(rec["code"])
        print(f"  Already done: {len(done_codes)} classes, skipping them")

    remaining = {k: v for k, v in classes.items() if k not in done_codes}
    if not remaining:
        print("All classes already processed!")
        return

    print(f"  Generating for {len(remaining)} classes")

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    codes = list(remaining.keys())
    descriptions = [remaining[c] for c in codes]

    for i, code in enumerate(codes):
        messages = build_messages(code, descriptions[i])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only generated tokens
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        queries = parse_queries(generated)

        print(f"\n[{i+1}/{len(codes)}] {code}: {len(queries)} queries")
        for q in queries[:5]:
            print(f"  - {q}")
        if len(queries) > 5:
            print(f"  ... and {len(queries) - 5} more")

        # Save in wiki JSONL format
        record = {
            "code": code,
            "description": descriptions[i],
            "wiki_texts": {"generated_queries": ". ".join(queries)},
            "wiki_char_count": sum(len(q) for q in queries),
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone! Saved to {output_path}")


if __name__ == "__main__":
    main()
