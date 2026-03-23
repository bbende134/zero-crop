#!/usr/bin/env python3
"""
OSM Enrichment Module
=====================
Queries OpenStreetMap via Overpass API to extract landuse and crop tags
for linking with satellite embeddings.

Extracts:
- landuse tags (farmland, orchard, vineyard, forest, meadow)
- crop tags (specific crop types)
- Centroids for coordinate matching with embeddings
"""

import json
import time
from pathlib import Path
from typing import Optional
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Landuse to text description mapping
LANDUSE_DESCRIPTIONS = {
    "farmland": "Agricultural farmland used for crop cultivation",
    "orchard": "Orchard with fruit trees",
    "vineyard": "Vineyard for grape cultivation and wine production",
    "meadow": "Meadow grassland, may be used for hay or grazing",
    "forest": "Forested area with trees",
    "grass": "Grassland or lawn area",
    "allotments": "Small-scale garden allotments for local food production",
    "greenhouse_horticulture": "Greenhouse or polytunnel for protected cultivation",
    "plant_nursery": "Plant nursery for growing seedlings and young plants",
}

# Crop to text description mapping  
CROP_DESCRIPTIONS = {
    "wheat": "Winter or spring wheat (Triticum aestivum), a major cereal crop",
    "corn": "Maize (Zea mays), grown for grain or silage",
    "maize": "Maize (Zea mays), grown for grain or silage",
    "sunflower": "Sunflower (Helianthus annuus), grown for oil production",
    "rapeseed": "Rapeseed (Brassica napus), grown for oil and animal feed",
    "barley": "Barley (Hordeum vulgare), used for animal feed and brewing",
    "oat": "Oat (Avena sativa), grown for grain and fodder",
    "rye": "Rye (Secale cereale), a hardy cereal grain",
    "soybean": "Soybean (Glycine max), a protein-rich legume crop",
    "potato": "Potato (Solanum tuberosum), a root vegetable crop",
    "sugar_beet": "Sugar beet (Beta vulgaris), grown for sugar production",
    "grape": "Grape vine (Vitis vinifera), for wine or table grapes",
}


def query_osm_region(
    bbox: tuple[float, float, float, float],
    landuse_types: Optional[list[str]] = None,
    timeout: int = 300
) -> list[dict]:
    """Query OSM for landuse features in a bounding box.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        landuse_types: List of landuse types to query (default: all agricultural)
        timeout: Request timeout in seconds
    
    Returns:
        List of features with lat, lon, tags, and generated text
    """
    if landuse_types is None:
        landuse_types = ["farmland", "orchard", "vineyard", "meadow", "forest"]
    
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    
    # Build query for each landuse type
    landuse_queries = "\n".join([
        f'way["landuse"="{lu}"]({bbox_str});' 
        for lu in landuse_types
    ])
    
    query = f"""
    [out:json][timeout:{timeout}];
    (
      {landuse_queries}
      way["crop"]({bbox_str});
    );
    out center meta;
    """
    
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        
        features = []
        for element in data.get("elements", []):
            if "center" not in element:
                continue
                
            tags = element.get("tags", {})
            landuse = tags.get("landuse", "")
            crop = tags.get("crop", "")
            
            # Generate text description
            text_parts = []
            if landuse in LANDUSE_DESCRIPTIONS:
                text_parts.append(LANDUSE_DESCRIPTIONS[landuse])
            if crop in CROP_DESCRIPTIONS:
                text_parts.append(CROP_DESCRIPTIONS[crop])
            elif crop:
                text_parts.append(f"Cultivated with {crop}")
            
            feature = {
                "id": element["id"],
                "lat": element["center"]["lat"],
                "lon": element["center"]["lon"],
                "landuse": landuse,
                "crop": crop,
                "text": " ".join(text_parts) if text_parts else f"Land use: {landuse or 'unspecified'}",
                "tags": tags
            }
            features.append(feature)
        
        return features
        
    except requests.exceptions.RequestException as e:
        print(f"OSM query failed: {e}")
        return []


def query_osm_hungary(output_path: Optional[Path] = None) -> list[dict]:
    """Query all agricultural landuse in Hungary.
    
    Hungary bbox: 16.0, 45.7, 22.9, 48.6
    
    Args:
        output_path: Optional path to save results
    
    Returns:
        List of landuse features
    """
    # Hungary bounding box
    HUNGARY_BBOX = (16.0, 45.7, 22.9, 48.6)
    
    print("Querying OSM for Hungary agricultural landuse...")
    print("This may take 5-10 minutes...")
    
    features = query_osm_region(HUNGARY_BBOX, timeout=600)
    
    print(f"Found {len(features)} landuse features")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"Saved to: {output_path}")
    
    return features


def generate_text_from_tags(tags: dict) -> str:
    """Generate natural language description from OSM tags.
    
    Args:
        tags: OSM tag dictionary
    
    Returns:
        Natural language description
    """
    parts = []
    
    landuse = tags.get("landuse", "")
    if landuse in LANDUSE_DESCRIPTIONS:
        parts.append(LANDUSE_DESCRIPTIONS[landuse])
    
    crop = tags.get("crop", "")
    if crop in CROP_DESCRIPTIONS:
        parts.append(CROP_DESCRIPTIONS[crop])
    elif crop:
        parts.append(f"Currently cultivated with {crop}")
    
    # Additional context
    if tags.get("organic") == "yes":
        parts.append("Organic farming practices")
    
    if tags.get("irrigated") == "yes":
        parts.append("Irrigated field")
    
    name = tags.get("name", "")
    if name:
        parts.append(f"Named: {name}")
    
    return ". ".join(parts) if parts else "Agricultural land"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query OSM landuse data")
    parser.add_argument("--hungary", action="store_true", help="Query all of Hungary")
    parser.add_argument("--bbox", type=float, nargs=4, 
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Custom bounding box")
    parser.add_argument("--output", type=Path, 
                        default=Path("data/datasets/osm/landuse.json"),
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.hungary:
        query_osm_hungary(args.output)
    elif args.bbox:
        features = query_osm_region(tuple(args.bbox))
        print(f"Found {len(features)} features")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(features, f, indent=2)
    else:
        parser.print_help()
