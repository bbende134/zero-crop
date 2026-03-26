import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def inspect_tif(filepath, plot=False):
    print(f"\n--- Inspecting: {os.path.basename(filepath)} ---")
    try:
        with rasterio.open(filepath) as src:
            print(f"Shape: {src.shape}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Count (Bands): {src.count}")
            print(f"Data Types: {src.dtypes}")
            
            # Read first band
            band1 = src.read(1)
            valid_mask = band1 != src.nodata if src.nodata is not None else np.ones_like(band1, dtype=bool)
            valid_data = band1[valid_mask]
            
            if len(valid_data) > 0:
                print(f"Min: {valid_data.min()}, Max: {valid_data.max()}, Mean: {valid_data.mean():.4f}")
                print(f"NoData Value: {src.nodata}")
            else:
                print("Warning: All data is NoData.")

            if plot:
                plt.figure(figsize=(6, 6))
                plt.imshow(band1, cmap='viridis')
                plt.colorbar()
                plt.title(f"Sample: {os.path.basename(filepath)}")
                # save plot instead of showing it to avoid blocking in non-interactive environment
                plot_filename = f"sample_{os.path.basename(filepath)}.png"
                plt.savefig(plot_filename)
                print(f"Saved plot to {plot_filename}")
                plt.close()
                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def main():
    base_dir = "/home/bende/dev/zero-crop/data_corine/Results-2"
    tif_files = glob.glob(os.path.join(base_dir, "**/*.tif"), recursive=True)
    
    print(f"Found {len(tif_files)} .tif files in {base_dir}")
    
    if len(tif_files) == 0:
        return
        
    # Categorize files by prefix (e.g. CLMS_HRLVLCC_CTY, CLMS_HRLVLCC_CPCSY)
    categories = {}
    for f in tif_files:
        basename = os.path.basename(f)
        prefix = "_".join(basename.split("_")[:3])
        if prefix not in categories:
            categories[prefix] = []
        categories[prefix].append(f)
        
    print(f"Identified {len(categories)} categories: {list(categories.keys())}")
    
    # Inspect one sample from each category
    for prefix, files in categories.items():
        sample_file = files[0]
        inspect_tif(sample_file, plot=True)
        
if __name__ == "__main__":
    main()
