#!/usr/bin/env python3
"""
Build Wikipedia-augmented descriptions for HRL crop type and ancillary classes.

Reads class descriptions from data_corine/hrl_classes_includes.json (one entry per
desc_key used in full_flow.py), generates Wikipedia search terms via Qwen, fetches
Wikipedia content, and appends results to data_corine/hrl_wiki_char_count.jsonl.

Covers all 27 classes:
  CTY crop types (19): wheat, barley, maize, rice, other_cereals, fresh_vegetables,
    dry_pulses, potatoes, sugar_beet, sunflower, soybeans, rapeseed, flax_cotton_hemp,
    grapes, olives, fruits, nuts, unclassified_arable, unclassified_permanent
  CPCSY (2): single_growing_season, double_growing_season
  WVL (1): woody_vegetation
  SWF (1): small_woody_features
  HER (1): permanent_grassland
  CPMCH (1): main_crop_harvest_date
  CPBSA (1): bare_soil_before_sowing
  CPBSB (1): bare_soil_after_harvest

Usage:
    .venv/bin/python data_processing/wiki_hrl34_search.py
"""

import json
import re
import os
import wikipedia
from pathlib import Path
from vllm import LLM, SamplingParams

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

wikipedia.set_lang("en")

DATA_DIR = Path(__file__).parent.parent / "data_corine"


def format_qwen_prompt(description: str) -> str:
    user_prompt = (
        f"Context: I am extracting crop and land cover concepts from remote sensing product descriptors.\n"
        f"Description: '{description}'\n\n"
        "Task: Based on the description, provide exactly 1 to 6 short, distinct Wikipedia search terms "
        "for the core crop/land cover concepts described. "
        "First, wrap your reasoning inside <think> and </think> tags. WARNING: Your reasoning MUST be extremely brief (1-3 sentences max). "
        "After the </think> tag, output ONLY the search terms separated by commas. DO NOT include extra words, bullet points, or punctuation at the end. "
        "Example Output format:\n<think>Very brief reasoning...</think>\nterm 1, term 2, term 3"
    )
    return (
        f"<|im_start|>system\nYou are a precise assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_qwen_output(generated_text: str) -> list[str]:
    if "</think>" in generated_text:
        final_answer = generated_text.split("</think>")[-1]
    else:
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        final_answer = lines[-1] if lines else ""

    final_answer = final_answer.strip()
    final_answer = final_answer.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    final_answer = final_answer.replace("assistant\n", "")

    terms = [t.strip().strip("'\".").lower() for t in final_answer.split(",")]
    terms = [re.sub(r"[^a-z0-9\s]", "", t) for t in terms if t]
    terms = [t.strip() for t in terms if t and len(t) < 40 and (" " not in t or t.count(" ") <= 3)]

    stop_words = {"this", "that", "however", "looking", "at", "the", "a", "an", "and",
                  "or", "so", "but", "therefore", "thus"}
    terms = [t for t in terms if t not in stop_words]
    return terms[:6]


def already_processed(output_file: Path) -> set[str]:
    seen = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["code"])
                except Exception:
                    pass
    return seen


def gather_wikipedia_char_count(input_file: Path, output_file: Path):
    with open(input_file, encoding="utf-8") as f:
        hrl_data = json.load(f)

    seen = already_processed(output_file)
    pending_codes = [c for c in hrl_data if c not in seen]

    if not pending_codes:
        print("All HRL classes already in output file — nothing to do.")
        return

    print(f"Initializing vLLM Engine for {len(pending_codes)} classes...")
    llm = LLM(
        model="Qwen/Qwen3.5-35B-A3B-FP8",
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=4096,
    )

    descriptions = [hrl_data[c] for c in pending_codes]
    prompts = [format_qwen_prompt(d) for d in descriptions]

    sampling_params = SamplingParams(
        max_tokens=1024, temperature=0.7,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    outputs = llm.generate(prompts, sampling_params)

    generated_terms_map = {}
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"\n--- RAW OUTPUT FOR {pending_codes[i]} ---\n{text}\n---")
        terms = parse_qwen_output(text)
        generated_terms_map[pending_codes[i]] = terms
        print(f"Code {pending_codes[i]}: extracted terms: {terms}")

    print("\nStarting Wikipedia search...")
    for code in pending_codes:
        description = hrl_data[code]
        terms = generated_terms_map[code]
        total_chars = 0
        successful_terms = []
        wiki_texts = {}

        for term in terms[:6]:
            try:
                page = wikipedia.page(term, auto_suggest=True)
                content = page.content
                total_chars += len(content)
                successful_terms.append(term)
                wiki_texts[term] = content
            except wikipedia.exceptions.DisambiguationError:
                pass
            except wikipedia.exceptions.PageError:
                pass
            except Exception:
                pass

        record = {
            "code": code,
            "description": description,
            "qwen_generated_terms": terms,
            "successful_terms_found": successful_terms,
            "wiki_char_count": total_chars,
            "wiki_texts": wiki_texts,
        }
        print(f"Code {code}: {total_chars} wiki chars via {successful_terms}")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    input_path = DATA_DIR / "hrl_classes_includes.json"
    output_path = DATA_DIR / "hrl_wiki_char_count.jsonl"
    gather_wikipedia_char_count(input_path, output_path)
