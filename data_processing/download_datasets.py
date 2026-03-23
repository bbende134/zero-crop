#!/usr/bin/env python3
"""
Dataset Download Pipeline
=========================
Downloads and preprocesses all datasets for text-to-embedding training.

Datasets:
- HuggingFace: EuroSAT, BigEarthNet, RESISC45, SkyScript, ChatEarthNet
- OSM: Landuse/crop tags for Hungary via Overpass API
- LUCAS: Requires manual download from ESDAC (registration needed)

Usage:
    uv run python data_processing/download_datasets.py --all
    uv run python data_processing/download_datasets.py --eurosat --skyscript
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import requests

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"


def download_huggingface_datasets(
    datasets_to_download: list[str],
    output_dir: Path,
    streaming: bool = False
) -> dict:
    """Download specified HuggingFace datasets.
    
    Args:
        datasets_to_download: List of dataset names to download
        output_dir: Where to save the datasets
        streaming: If True, use streaming mode (for large datasets)
    
    Returns:
        dict mapping dataset names to their local paths
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets huggingface_hub")
        from datasets import load_dataset
    
    # HuggingFace dataset IDs
    HF_DATASETS = {
        "eurosat": "blanchon/EuroSAT",
        "bigearth": "timm/BigEarthNet", 
        "resisc45": "timm/resisc45",
        "skyscript": "wangzhecheng/SkyScript",
        "chatearthnet": "zhu-xlab/ChatEarthNet",
        "soil_qa": "YuvrajSingh9886/Agriculture-Soil-QA-Pairs-Dataset",
    }
    
    downloaded = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name in datasets_to_download:
        if name not in HF_DATASETS:
            print(f"Unknown dataset: {name}")
            continue
            
        hf_id = HF_DATASETS[name]
        save_path = output_dir / name
        
        print(f"\n{'='*50}")
        print(f"Downloading: {name} ({hf_id})")
        print(f"{'='*50}")
        
        try:
            if streaming:
                # For very large datasets, use streaming
                ds = load_dataset(hf_id, streaming=True)
                print(f"Dataset loaded in streaming mode")
                # Save first N samples for inspection
                sample_path = save_path / "samples"
                sample_path.mkdir(parents=True, exist_ok=True)
                
                split = "train" if "train" in ds else list(ds.keys())[0]
                samples = list(ds[split].take(100))
                with open(sample_path / "sample_100.json", "w") as f:
                    json.dump([{k: str(v)[:200] for k, v in s.items()} for s in samples], f, indent=2)
                print(f"Saved 100 samples to {sample_path}")
            else:
                ds = load_dataset(hf_id)
                ds.save_to_disk(str(save_path))
                print(f"Saved to: {save_path}")
            
            downloaded[name] = str(save_path)
            
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    return downloaded


def download_osm_hungary(output_dir: Path) -> Path:
    """Download OpenStreetMap landuse data for Hungary via Overpass API.
    
    Returns:
        Path to saved GeoJSON file
    """
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    
    # Query for landuse and crop tags in Hungary
    query = """
    [out:json][timeout:600];
    area["ISO3166-1"="HU"]->.hungary;
    (
      way["landuse"="farmland"](area.hungary);
      way["landuse"="orchard"](area.hungary);
      way["landuse"="vineyard"](area.hungary);
      way["landuse"="meadow"](area.hungary);
      way["landuse"="forest"](area.hungary);
      way["crop"](area.hungary);
    );
    out center meta;
    """
    
    print("\n" + "="*50)
    print("Downloading OSM landuse data for Hungary...")
    print("="*50)
    print("This may take several minutes...")
    
    try:
        response = requests.post(
            OVERPASS_URL, 
            data={"data": query},
            timeout=600
        )
        response.raise_for_status()
        data = response.json()
        
        # Convert to simplified format
        features = []
        for element in data.get("elements", []):
            if "center" in element:
                feature = {
                    "id": element["id"],
                    "lat": element["center"]["lat"],
                    "lon": element["center"]["lon"],
                    "tags": element.get("tags", {})
                }
                features.append(feature)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "osm_hungary_landuse.json"
        
        with open(output_path, "w") as f:
            json.dump({
                "count": len(features),
                "features": features
            }, f, indent=2)
        
        print(f"Downloaded {len(features)} landuse features")
        print(f"Saved to: {output_path}")
        
        return output_path
        
    except requests.exceptions.Timeout:
        print("Request timed out. Try running during off-peak hours.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def print_lucas_instructions():
    """Print instructions for downloading LUCAS soil data."""
    print("\n" + "="*50)
    print("LUCAS Soil Data - Manual Download Required")
    print("="*50)
    print("""
LUCAS requires registration at ESDAC (European Soil Data Centre).

Steps:
1. Go to: https://esdac.jrc.ec.europa.eu/content/lucas-soil

2. Click 'Request Data Access' and fill the form:
   - Purpose: Research/Academic
   - Describe: Training ML models for land use classification

3. You'll receive download link via email (usually same day)

4. Download files:
   - LUCAS_Topsoil_2018.csv (or latest available)
   - Look for fields: POINT_ID, TH_LAT, TH_LONG, LC1, pH_H2O, OC, N, P, K

5. Place downloaded files in: data/datasets/lucas/

Alternative - LUCAS microdata (no registration):
   https://ec.europa.eu/eurostat/web/lucas/data
""")


def main():
    parser = argparse.ArgumentParser(description="Download EO datasets")
    
    # HuggingFace datasets
    parser.add_argument("--eurosat", action="store_true", help="Download EuroSAT (27k, 10 classes)")
    parser.add_argument("--bigearth", action="store_true", help="Download BigEarthNet (590k, multi-label)")
    parser.add_argument("--resisc45", action="store_true", help="Download RESISC45 (31.5k, 45 classes)")
    parser.add_argument("--skyscript", action="store_true", help="Download SkyScript (5.2M text-image pairs)")
    parser.add_argument("--chatearthnet", action="store_true", help="Download ChatEarthNet (173k captions)")
    parser.add_argument("--soil-qa", action="store_true", help="Download Soil Q&A dataset")
    
    # Other sources
    parser.add_argument("--osm", action="store_true", help="Download OSM landuse for Hungary")
    parser.add_argument("--lucas", action="store_true", help="Show LUCAS download instructions")
    
    # Convenience flags
    parser.add_argument("--all-hf", action="store_true", help="Download all HuggingFace datasets")
    parser.add_argument("--all", action="store_true", help="Download everything (except LUCAS)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming for large datasets")
    
    # Output directory
    parser.add_argument("--output", type=Path, default=DATA_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    # Determine which datasets to download
    hf_datasets = []
    
    if args.all or args.all_hf:
        hf_datasets = ["eurosat", "resisc45", "soil_qa", "chatearthnet"]
        # Use streaming for very large datasets
        if not args.streaming:
            print("Note: Using streaming for BigEarthNet and SkyScript (very large)")
    
    if args.eurosat:
        hf_datasets.append("eurosat")
    if args.bigearth:
        hf_datasets.append("bigearth")
    if args.resisc45:
        hf_datasets.append("resisc45")
    if args.skyscript:
        hf_datasets.append("skyscript")
    if args.chatearthnet:
        hf_datasets.append("chatearthnet")
    if args.soil_qa:
        hf_datasets.append("soil_qa")
    
    # Remove duplicates
    hf_datasets = list(set(hf_datasets))
    
    # Download HuggingFace datasets
    if hf_datasets:
        large_datasets = {"bigearth", "skyscript"}
        small = [d for d in hf_datasets if d not in large_datasets]
        large = [d for d in hf_datasets if d in large_datasets]
        
        if small:
            download_huggingface_datasets(small, args.output, streaming=False)
        if large:
            download_huggingface_datasets(large, args.output, streaming=True)
    
    # Download OSM
    if args.osm or args.all:
        download_osm_hungary(args.output / "osm")
    
    # LUCAS instructions
    if args.lucas or args.all:
        print_lucas_instructions()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample: uv run python data_processing/download_datasets.py --eurosat --osm")


if __name__ == "__main__":
    main()
