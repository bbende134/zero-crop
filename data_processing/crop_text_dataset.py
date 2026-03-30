#!/usr/bin/env python3
"""
Crop Text Dataset
=================
PyTorch Dataset for training text-to-satellite embedding alignment.

Combines:
- Crop descriptions from scrape_flora.py (Wikipedia + GBIF)
- Satellite embeddings from Milvus (terra_S2L2A or high_res_hun)

Usage:
    uv run python -m fine_tune.crop_text_dataset --test
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pymilvus import connections, Collection
import csv
from collections import defaultdict

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.scrape_flora import gather_for_species


@dataclass
class CropSample:
    """Single training sample with text and embedding."""
    scientific_name: str
    common_names: List[str]
    description: str  # Short summary
    long_description: Optional[str]  # Full Wikipedia text
    habitat: Optional[str]
    lat: float
    lon: float
    satellite_embedding: np.ndarray
    
    
class CropTextDataset(Dataset):
    """
    Dataset for crop text → satellite embedding alignment.
    
    Uses sliding window chunking:
    - Each text is split into 1024-char chunks with 124-char overlap
    - Each chunk is paired with the same satellite embedding
    
    Args:
        species_list: List of scientific names to include
        milvus_host: Milvus server host
        milvus_port: Milvus server port  
        collection_name: Milvus collection ("terra_S2L2A" for 768-dim, "high_res_hun" for 64-dim)
        embedding_field: Name of embedding vector field in collection
        search_radius: Radius in degrees for fetching nearby embeddings (~1km = 0.01)
        wiki_lang: Preferred Wikipedia language for descriptions
        cache_path: Optional path to cache scraped data
        input_chars: Number of characters per chunk (default: 1024)
        overlap_chars: Overlap between chunks (default: 124)
    """
    
    def __init__(
        self,
        species_list: List[str],
        milvus_host: str = "192.168.242.182",
        milvus_port: str = "19530",
        collection_name: str = "terra_S2L2A",
        embedding_field: str = "embedding",
        search_radius: float = 0.01,
        wiki_lang: str = "en",
        csv_path: Optional[str] = None,
        cache_path: Optional[Path] = None,
        limit_per_species: int = 500,
        input_chars: int = 1024,
        overlap_chars: int = 124,
    ):
        self.species_list = species_list
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.search_radius = search_radius
        self.search_radius = search_radius
        self.wiki_lang = wiki_lang
        self.csv_path = csv_path
        self.cache_path = Path(cache_path) if cache_path else None
        self.limit_per_species = limit_per_species
        self.input_chars = input_chars
        self.overlap_chars = overlap_chars
        
        # Determine embedding dimension based on collection
        self.embedding_dim = 768 if "terra" in collection_name.lower() else 64
        
        # Load or build samples (raw samples before chunking)
        self.samples: List[CropSample] = []
        self._build_dataset()
        
        # Calculate species counts for dynamic top_k
        from collections import Counter
        self.species_counts = Counter(s.scientific_name for s in self.samples)
        
        # Build chunked training pairs
        self.chunks: List[Dict] = []
        self._build_chunks()
        
    def _build_chunks(self):
        """Build sliding window chunks from all samples."""
        stride = self.input_chars - self.overlap_chars
        
        for sample in self.samples:
            # Format full text - use long_description if available
            common_str = ", ".join(sample.common_names) if sample.common_names else ""
            text_parts = [f"Species: {sample.scientific_name}"]
            if common_str:
                text_parts.append(f"Common names: {common_str}")
            
            # Prefer long_description (full Wikipedia article) over short summary
            if sample.long_description:
                text_parts.append(f"Description: {sample.long_description}")
            else:
                text_parts.append(f"Description: {sample.description}")
            
            if sample.habitat:
                text_parts.append(f"Habitat: {sample.habitat}")
            
            full_text = "\n".join(text_parts)
            
            # Create sliding window chunks
            if len(full_text) <= self.input_chars:
                # Text is short enough - use as single chunk
                self.chunks.append({
                    "text": full_text,
                    "embedding": sample.satellite_embedding,
                    "metadata": {
                        "scientific_name": sample.scientific_name,
                        "species_count": self.species_counts[sample.scientific_name],
                        "lat": sample.lat,
                        "lon": sample.lon,
                        "chunk_idx": 0,
                    }
                })
            else:
                # Slide through text
                pos = 0
                chunk_idx = 0
                while pos < len(full_text):
                    end_pos = min(pos + self.input_chars, len(full_text))
                    chunk_text = full_text[pos:end_pos]
                    
                    # Skip very short final chunks
                    if len(chunk_text) < self.input_chars // 2 and chunk_idx > 0:
                        break
                    
                    self.chunks.append({
                        "text": chunk_text,
                        "embedding": sample.satellite_embedding,
                        "metadata": {
                            "scientific_name": sample.scientific_name,
                            "species_count": self.species_counts[sample.scientific_name],
                            "lat": sample.lat,
                            "lon": sample.lon,
                            "chunk_idx": chunk_idx,
                        }
                    })
                    
                    pos += stride
                    chunk_idx += 1
                    
                    # Handle last chunk
                    if end_pos == len(full_text):
                        break
        
        print(f"Created {len(self.chunks)} chunks from {len(self.samples)} samples")
        
    def _connect_milvus(self) -> Collection:
        """Connect to Milvus and return collection."""
        connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
        collection = Collection(self.collection_name)
        collection.load()
        return collection
    
    def _get_embedding_at_location(
        self, 
        collection: Collection, 
        lat: float, 
        lon: float
    ) -> Optional[np.ndarray]:
        """Fetch median embedding within search radius of location."""
        epsilon = self.search_radius
        expr = (
            f"lat >= {lat - epsilon} && lat <= {lat + epsilon} && "
            f"lon >= {lon - epsilon} && lon <= {lon + epsilon}"
        )
        
        try:
            results = collection.query(
                expr=expr, 
                output_fields=[self.embedding_field], 
                limit=100
            )
            
            if not results:
                return None
                
            vectors = np.array([r[self.embedding_field] for r in results])
            return np.median(vectors, axis=0)
            
        except Exception:
            return None
    
    def _extract_habitat_info(self, sections: Optional[List[Dict]]) -> Optional[str]:
        """Extract habitat/ecology info from Wikipedia sections."""
        if not sections:
            return None
            
        habitat_keywords = ["habitat", "ecology", "distribution", "climate", "growing", "cultivation"]
        
        for section in sections:
            title = (section.get("title") or "").lower()
            if any(kw in title for kw in habitat_keywords):
                text = section.get("text", "")
                # Truncate to reasonable length
                return text[:500] if len(text) > 500 else text
                
        return None
    
    def _load_csv_occurrences(self) -> List[Dict]:
        """Load occurrences from CSV file."""
        rows = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            print(f"Error reading CSV {self.csv_path}: {e}")
        return rows
    
    def _build_dataset(self):
        """Build dataset by fetching descriptions and embeddings."""
        print(f"Building dataset for {len(self.species_list)} species...")
        
        # Check cache first
        if self.cache_path and self.cache_path.exists():
            print(f"Loading from cache: {self.cache_path}")
            self._load_cache()
            return
            
        # Connect to Milvus
        collection = self._connect_milvus()
        
        # Load CSV occurrences if path provided
        csv_occurrences = None
        if self.csv_path:
            print(f"Loading occurrences from CSV: {self.csv_path}")
            csv_occurrences = self._load_csv_occurrences()
            print(f"  Loaded {len(csv_occurrences)} total rows from CSV")
        
        for species_name in self.species_list:
            # Set limit_occ: 0 if using CSV (we don't need GBIF occurrences), else use config
            limit_occ = 0 if csv_occurrences else self.limit_per_species
            print(f"Processing: {species_name}")
            
            try:
                # Fetch species info and occurrences
                species_info, occurrences = gather_for_species(
                    name=species_name,
                    country="HU",  # Default to Hungary
                    geometry=None,
                    year_from=None,
                    year_to=None,
                    managed_only=False,
                    limit_occ=limit_occ,
                    wiki_lang=self.wiki_lang,
                )
                
                # 2. Get Occurrences (Source: CSV or GBIF)
                final_occurrences = []
                
                if csv_occurrences:
                    # Filter CSV data for this species (checking scientificName)
                    # We accept partial match or exact match
                    # Normalize names for comparison
                    norm_target = species_name.lower()
                    
                    found_rows = []
                    for row in csv_occurrences:
                        row_name = row.get("scientificName", "").lower()
                        if norm_target in row_name or row_name in norm_target:
                            found_rows.append(row)
                    
                    if not found_rows:
                        print(f"  Warning: No occurrences in CSV for {species_name}")
                        continue
                        
                    print(f"  Found {len(found_rows)} occurrences in CSV for {species_name}")
                    
                    # Convert to minimal format expected by loop
                    for row in found_rows:
                        try:
                            lat = float(row["decimalLatitude"])
                            lon = float(row["decimalLongitude"])
                            final_occurrences.append({"decimalLatitude": lat, "decimalLongitude": lon})
                        except (ValueError, KeyError):
                            continue
                else:
                    final_occurrences = fetched_occs
                
                if not final_occurrences:
                    print(f"  No occurrences found for {species_name}")
                    continue
                
                # 3. For each location, get satellite embedding from Milvus
                valid_count = 0
                for occ in final_occurrences:
                    lat = occ.get("decimalLatitude")
                    lon = occ.get("decimalLongitude")
                    
                    if lat is None or lon is None:
                        continue
                        
                    # Find nearest satellite embedding
                    emb = self._get_embedding_at_location(collection, lat, lon)
                    if emb is None:
                        continue
                        
                    # Create sample
                    sample = CropSample(
                        scientific_name=species_info.canonical_name or species_name,
                        common_names=[v["vernacularName"] for v in species_info.vernacular_names],
                        description=species_info.description or "",
                        long_description=species_info.long_description,
                        habitat=self._extract_habitat_info(species_info.sections),
                        lat=lat,
                        lon=lon,
                        satellite_embedding=emb,
                    )
                    self.samples.append(sample)
                    valid_count += 1
                
                print(f"  Added {valid_count} samples")
                
            except Exception as e:
                print(f"Error processing {species_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nTotal raw samples: {len(self.samples)}")
        
        # Save cache
        if self.cache_path:
            self._save_cache()
    
    def _save_cache(self):
        """Save dataset to cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for s in self.samples:
            data.append({
                "scientific_name": s.scientific_name,
                "common_names": s.common_names,
                "description": s.description,
                "long_description": s.long_description,
                "habitat": s.habitat,
                "lat": s.lat,
                "lon": s.lon,
                "satellite_embedding": s.satellite_embedding.tolist(),
            })
        
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Saved cache: {self.cache_path}")
    
    def _load_cache(self):
        """Load dataset from cache file."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            sample = CropSample(
                scientific_name=item["scientific_name"],
                common_names=item["common_names"],
                description=item["description"],
                long_description=item.get("long_description"),
                habitat=item.get("habitat"),
                lat=item["lat"],
                lon=item["lon"],
                satellite_embedding=np.array(item["satellite_embedding"]),
            )
            self.samples.append(sample)

        
        print(f"Loaded {len(self.samples)} samples from cache")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dict with:
            - text: Text chunk (up to 1024 chars)
            - embedding: Satellite embedding tensor
            - metadata: Additional info (lat, lon, species, chunk_idx)
        """
        chunk = self.chunks[idx]
        
        return {
            "text": chunk["text"],
            "embedding": torch.tensor(chunk["embedding"], dtype=torch.float32),
            "metadata": chunk["metadata"],
        }
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension based on collection."""
        return self.embedding_dim
    
    def get_num_raw_samples(self) -> int:
        """Return number of raw samples before chunking."""
        return len(self.samples)


# Default species list for training
DEFAULT_SPECIES = [
    "Triticum aestivum",     # Wheat
    "Zea mays",              # Corn
    "Helianthus annuus",     # Sunflower
    "Vitis vinifera",        # Grape
    "Malus domestica",       # Apple
    "Prunus domestica",      # Plum
    "Brassica napus",        # Rapeseed
    "Hordeum vulgare",       # Barley
    "Avena sativa",          # Oat
    "Solanum tuberosum",     # Potato
]


def test_dataset():
    """Quick test of dataset loading."""
    print("Testing CropTextDataset...")
    
    # Test with a single species
    dataset = CropTextDataset(
        species_list=["Triticum aestivum"],
        collection_name="terra_S2L2A",
        limit_per_species=10,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample text:\n{sample['text'][:300]}...")
        print(f"\nEmbedding shape: {sample['embedding'].shape}")
        print(f"Metadata: {sample['metadata']}")
        print("\n✓ Test passed!")
    else:
        print("⚠ No samples loaded (check Milvus connection)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop Text Dataset")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--collection", default="terra_S2L2A", help="Milvus collection name")
    parser.add_argument("--species", nargs="+", default=DEFAULT_SPECIES, help="Species to include")
    
    args = parser.parse_args()
    
    if args.test:
        test_dataset()
    else:
        # Build full dataset
        dataset = CropTextDataset(
            species_list=args.species,
            collection_name=args.collection,
            cache_path=Path("data/crop_dataset_cache.json"),
        )
        print(f"Dataset ready with {len(dataset)} samples")
