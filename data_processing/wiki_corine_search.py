import json
import re
import wikipedia
import os
from pathlib import Path
from vllm import LLM, SamplingParams

# Load env vars if .env exists (as seen in test_qwen_vllm.py)
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

# Fix to handle some Wikipedia bugs gracefully
wikipedia.set_lang("en")

DATA_DIR = Path(__file__).parent.parent / "data_corine"

def format_qwen_prompt(description):
    user_prompt = (
        f"Context: I am extracting crop and land use types from CORINE descriptors.\n"
        f"Description: '{description}'\n\n"
        "Task: Based on the description, provide exactly 1 to 6 short, distinct Wikipedia search terms for the core crop/land use concepts described. "
        "First, wrap your reasoning inside <think> and </think> tags. WARNING: Your reasoning MUST be extremely brief (1-3 sentences max). "
        "After the </think> tag, output ONLY the search terms separated by commas. DO NOT include extra words, bullet points, or punctuation at the end. "
        "Example Output format:\n<think>Very brief reasoning...</think>\nterm 1, term 2, term 3"
    )
    # ChatML format used by Qwen
    return f"<|im_start|>system\nYou are a precise assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def parse_qwen_output(generated_text):
    # Extract text after </think> if present
    if "</think>" in generated_text:
        final_answer = generated_text.split("</think>")[-1]
    else:
        # Fallback if the model didn't use the tags or ran out of tokens
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        if not lines:
            return []
        final_answer = lines[-1]
        
    # Clean up output
    final_answer = final_answer.strip()
    final_answer = final_answer.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    final_answer = final_answer.replace("assistant\n", "")
    
    # Split by comma and clean terms further
    terms = [t.strip().strip("'\".").lower() for t in final_answer.split(',')]
    terms = [re.sub(r'[^a-z0-9\s]', '', t) for t in terms if t]
    terms = [t.strip() for t in terms if t and len(t) < 40 and " " not in t or t.count(" ") <= 3]
    
    # Filter out conversational filler and single-word stop words that might arise from truncated thoughts
    stop_words = {"this", "that", "however", "looking", "at", "the", "a", "an", "and", "or", "so", "but", "therefore", "thus"}
    terms = [t for t in terms if t not in stop_words]
    
    return terms[:6]

def gather_wikipedia_char_count(input_file, output_file):
    print("Initializing vLLM Engine...")
    llm = LLM(
        model="Qwen/Qwen3.5-35B-A3B-FP8", 
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=4096
    )
    
    with open(input_file, 'r', encoding='utf-8') as f:
        corine_data = json.load(f)
        
    codes = list(corine_data.keys())
    descriptions = [corine_data[c] for c in codes]
    prompts = [format_qwen_prompt(desc) for desc in descriptions]
    
    print(f"Generating search terms for {len(prompts)} classes using vLLM...")
    # Added stop parameter strictly for ChatML end sequence
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.7, stop=["<|im_end|>", "<|endoftext|>"])
    outputs = llm.generate(prompts, sampling_params)
    
    generated_terms_map = {}
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"\n--- RAW OUTPUT FOR CODE {codes[i]} ---")
        print(generated_text)
        print("--------------------------------------\n")
        
        terms = parse_qwen_output(generated_text)
        generated_terms_map[codes[i]] = terms
        print(f"Code {codes[i]} '{descriptions[i]}' extracted terms: {terms}")
        
    print("\nStarting Wikipedia search...")
    
    # Optionally remove the output file if we are starting fresh
    if os.path.exists(output_file):
        os.remove(output_file)
        
    for code in codes:
        description = corine_data[code]
        terms = generated_terms_map[code]
        
        total_chars = 0
        successful_terms = []
        
        # Create a detailed record
        result_record = {
            "code": code,
            "description": description,
            "qwen_generated_terms": terms,
            "successful_terms_found": successful_terms,
            "wiki_char_count": 0,
            "wiki_texts": {}
        }
        
        for term in terms[:6]:
            try:
                # Fetch full page content instead of just 2 sentences
                page = wikipedia.page(term, auto_suggest=True)
                content = page.content
                total_chars += len(content)
                successful_terms.append(term)
                
                # We can also store the text itself in case it's needed for augmentation later
                if "wiki_texts" not in result_record:
                    result_record["wiki_texts"] = {}
                result_record["wiki_texts"][term] = content
                
            except wikipedia.exceptions.DisambiguationError:
                pass
            except wikipedia.exceptions.PageError:
                pass
            except Exception:
                pass
                
        # Update the character count
        result_record["wiki_char_count"] = total_chars
        
        print(f"Code {code}: {total_chars} wiki chars found via {successful_terms}.")
        
        # Append to JSONL file immediately to prevent data loss
        with open(output_file, 'a', encoding='utf-8') as sf:
            sf.write(json.dumps(result_record) + "\n")
            
    print(f"\nDone! Results saved incrementally to {output_file}")

if __name__ == "__main__":
    input_path = DATA_DIR / "corine_classes_includes.json"
    output_path = DATA_DIR / "corine_wiki_char_count.jsonl"
    gather_wikipedia_char_count(input_path, output_path)
