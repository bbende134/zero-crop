import urllib.request
from pathlib import Path
import fitz
import re
import json

PDF_URL = "https://land.copernicus.eu/content/corine-land-cover-nomenclature-guidelines/docs/pdf/CLC2018_Nomenclature_illustrated_guide_20190510.pdf"
DATA_DIR = Path(__file__).parent.parent / "data_corine"
PDF_PATH = DATA_DIR / "CLC2018_Nomenclature_illustrated_guide.pdf"

def download_pdf():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        print(f"Downloading {PDF_URL}...")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
        print("Download complete.")

def clean_includes(includes_lines):
    cleaned_lines = []
    for line in includes_lines:
        line = line.strip()
        # Remove bullet points (circles, squares, characters, 'o')
        line = re.sub(r'^[·o]\s+', '', line)
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines)
    # Fix hyphenation across lines
    text = re.sub(r'-\n', '-', text)
    # Join all remaining lines
    text = text.replace('\n', ' ')
    
    # Split by semicolons, strip, and join with commas
    parts = re.split(r';', text)
    final_parts = []
    for p in parts:
        p = p.strip()
        if p.endswith('.'):
            p = p[:-1].strip()
        if p:
            final_parts.append(p)
            
    return ", ".join(final_parts)

def extract_classes_from_pdf():
    doc = fitz.open(PDF_PATH)
    lines = []
    
    # Extract text from all pages and split into lines, cleaning up page breaks
    for page in doc:
        text = page.get_text("text", sort=True)
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
                
    classes = {}
    current_class_code = None
    in_includes_section = False
    found_includes = False # to only capture the main includes
    includes_text = []

    # Regex to catch main class headers "231 Pastures..."
    class_header_re = re.compile(r'^([1-5]\d{2})\s+([A-Z].*)$')
    
    for i, line in enumerate(lines):
        # Check if line is a class header
        match = class_header_re.match(line)
        if match:
            # We found a new class! Save the old one if exists
            if current_class_code and includes_text:
                classes[current_class_code] = clean_includes(includes_text)
            
            current_class_code = match.group(1)
            
            in_includes_section = False
            found_includes = False # to only capture the main includes
            includes_text = []
            continue
            
        if current_class_code and not found_includes:
            if line.startswith("This class includes:"):
                in_includes_section = True
                continue
            elif in_includes_section and (line.startswith("This class is not applicable for:") or line.startswith("This class excludes:") or line.startswith("Particularity of class") or "This class refers to" in line):
                in_includes_section = False
                found_includes = True # We finished the main includes section
                
            if in_includes_section:
                includes_text.append(line)

    # Save the last one
    if current_class_code and includes_text:
        classes[current_class_code] = clean_includes(includes_text)
        
    return classes

if __name__ == "__main__":
    download_pdf()
    parsed_classes = extract_classes_from_pdf()
    
    # Filter by codes present in the geojson
    geojson_path = DATA_DIR / "U2018_CLC2018_V2020_20u1.json"
    valid_codes = set()
    if geojson_path.exists():
        print("Reading GeoJSON to determine valid codes...")
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
            for feature in geojson_data.get('features', []):
                code = feature.get('properties', {}).get('Code_18')
                if code:
                    valid_codes.add(str(code))
        print(f"Found {len(valid_codes)} unique Code_18 values in GeoJSON.")
    else:
        print(f"GeoJSON not found at {geojson_path}. Proceeding without filtering.")
        
    if valid_codes:
        parsed_classes = {code: text for code, text in parsed_classes.items() if code in valid_codes}
    
    out_path = DATA_DIR / "corine_classes_includes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_classes, f, indent=4, ensure_ascii=False)
        
    print(f"Extracted and filtered to {len(parsed_classes)} classes, saved to {out_path}.")
    
    # Print the specific requested class
    target = "231"
    if target in parsed_classes:
        print(f"\nFound {target}:\n")
        print(parsed_classes[target])
    else:
        print(f"\nClass {target} not found!")
