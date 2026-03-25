import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run zonal statistics on crop types.")
    parser.add_argument('--raster', 
                        default='data_corine/Results-2/CLMS_HRLVLCC_CPCSY_S2021_R10m_E47N25_03035_V01_R00.tif',
                        help="Path to the crop type raster file (.tif)")
    parser.add_argument('--polygons', required=True, help="Path to the polygons vector file (.geojson, .shp)")
    args = parser.parse_args()
    
    # 1. Create the dictionary mapping based on Table 7-1
    crop_nomenclature = {
        1110: "Wheat",
        1120: "Barley",
        1130: "Cereals",
        1140: "Rice",
        1150: "Other cereals",
        1210: "Fresh Vegetables",
        1220: "Dry pulses",
        1310: "Potatoes",
        1320: "Sugar Beet",
        1410: "Sunflower",
        1420: "Soybeans",
        1430: "Rapeseed",
        1440: "Flax, cotton and hemp",
        2100: "Grapes",
        2200: "Olives",
        2310: "Fruits",
        2320: "Nuts",
        3100: "Unclassified arable crop",
        3200: "Unclassified permanent crop"
    }

    polygon_path = args.polygons
    raster_path = args.raster
    
    if not os.path.exists(polygon_path):
        print(f"Error: Polygon file '{polygon_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(raster_path):
        print(f"Error: Raster file '{raster_path}' not found.")
        sys.exit(1)
    
    print(f"Running zonal statistics on {raster_path} using vectors from {polygon_path}...")
    
    try:
        # Load polygons to ensure CRS matches raster (omitted here since code assumes CRS matches)
        # polygons = gpd.read_file(polygon_path)
        
        # 2. Run your zonal stats (assuming you've loaded a GeoJSON and matched the CRS)
        stats = zonal_stats(
            vectors=polygon_path, 
            raster=raster_path, 
            categorical=True
        )

        # 3. Translate the raw codes to human-readable text
        for i, polygon_stat in enumerate(stats):
            print(f"\nPolygon {i} contains:")
            
            # If the polygon did not overlap with any valid pixels, it could be empty
            if not polygon_stat:
                print(" - No crops found or out of raster bounds.")
                continue

            # polygon_stat is a dictionary like {1110: 500, 1410: 1200}
            for pixel_code, pixel_count in polygon_stat.items():
                if pixel_code is None: 
                    continue # nodata or background
                    
                # Look up the name in our dictionary, default to "Unknown" if missing
                crop_name = crop_nomenclature.get(pixel_code, f"Unknown Code ({pixel_code})")
                
                # Calculate approximate area (assuming 10m x 10m pixels = 100 sqm per pixel)
                area_hectares = (pixel_count * 100) / 10000
                
                print(f" - {crop_name}: {pixel_count} pixels ({area_hectares} hectares)")

    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()
