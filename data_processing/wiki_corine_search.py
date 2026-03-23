import json
import re
import wikipedia
from pathlib import Path

# Fix to handle some Wikipedia bugs gracefully
wikipedia.set_lang("en")

DATA_DIR = Path(__file__).parent.parent / "data_corine"

def extract_search_terms(description):
    search_terms = set()
    
    # 1. Biological/scientific names (Capitalized + lowercase or spp.)
    bio_matches = re.findall(r'\b([A-Z][a-z]+ (?:[a-z]+|spp\.?))\b', description)
    for match in bio_matches:
        search_terms.add(match.replace('spp.', '').strip())
        
    # 2. Extract words inside parentheses as they are often families (e.g., 'graminacea')
    paren_matches = re.findall(r'\(([A-Za-z\s]+)\)', description)
    for match in paren_matches:
        if " etc" not in match:
            search_terms.add(match.strip())

    # 3. Extract core concepts from comma-separated parts
    parts = description.split(',')
    for part in parts:
        clean_part = re.sub(r'\(.*?\)', '', part).strip()
        # Take short, non-generic phrases
        if 0 < len(clean_part.split()) <= 3 and "vegetation" not in clean_part.lower():
            search_terms.add(clean_part.lower())
            
    return list(search_terms)

def gather_wikipedia_char_count(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        corine_data = json.load(f)
        
    results = {}
    
    print(f"Loaded {len(corine_data)} classes. Starting Wikipedia search...")
    for code, description in corine_data.items():
        terms = extract_search_terms(description)
        total_chars = 0
        
        # Limit to the top 4 terms per class to save time/API calls
        for term in terms[:4]:
            try:
                # Fetch summary and measure length
                summary = wikipedia.summary(term, sentences=2, auto_suggest=True)
                total_chars += len(summary)
            except wikipedia.exceptions.DisambiguationError:
                pass # Skip ambiguous terms
            except wikipedia.exceptions.PageError:
                pass # Skip missing pages
            except Exception:
                pass # Catch any requests/network errors
                
        results[code] = total_chars
        print(f"Code {code}: {total_chars} chars found.")
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    input_path = DATA_DIR / "corine_classes_includes.json"
    output_path = DATA_DIR / "corine_wiki_char_count.json"
    gather_wikipedia_char_count(input_path, output_path)
