import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from rasterio.enums import Resampling

# 1. Define the Nomenclature and a Fixed Color Palette
# Mapping Code -> (Crop Name, Hex Color)
CROP_MAP = {
    1110: ("Wheat", "#FFD700"),               # Gold
    1120: ("Barley", "#DEB887"),              # Burlywood
    1130: ("Maize", "#7CFC00"),               # Lawn Green
    1140: ("Rice", "#00FFFF"),                # Cyan
    1150: ("Other cereals", "#F4A460"),       # Sandy Brown
    1210: ("Fresh Vegetables", "#32CD32"),    # Lime Green
    1220: ("Dry pulses", "#8FBC8F"),          # Dark Sea Green
    1310: ("Potatoes", "#D2B48C"),            # Tan
    1320: ("Sugar Beet", "#FFC0CB"),          # Pink
    1410: ("Sunflower", "#FFFF00"),           # Yellow
    1420: ("Soybeans", "#98FB98"),            # Pale Green
    1430: ("Rapeseed", "#006400"),            # Dark Green
    1440: ("Flax, cotton and hemp", "#BDB76B"),# Dark Khaki
    2100: ("Grapes", "#800080"),              # Purple
    2200: ("Olives", "#556B2F"),              # Dark Olive Green
    2310: ("Fruits", "#8B0000"),              # Dark Red
    2320: ("Nuts", "#A0522D"),                # Sienna
    3100: ("Unclassified arable", "#D3D3D3"), # Light Gray
    3200: ("Unclassified permanent", "#A9A9A9")# Dark Gray
}

def plot_consolidated_crop_map(directory_path, downsample_factor=10):
    # Find all Crop Type tiles
    search_pattern = os.path.join(directory_path, "*_CTY_*.tif")
    tif_files = glob.glob(search_pattern)
    
    if not tif_files:
        print(f"No *_CTY_*.tif files found in {directory_path}!")
        return

    print(f"Found {len(tif_files)} tiles. Generating map...")

    # 2. Set up the Master Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Copernicus Crop Types 2021 - Consolidated Map", fontsize=18, pad=20)
    ax.set_facecolor('#f0f0f0') # Light grey background for NoData areas
    
    # 3. Iterate through tiles and plot them geographically
    for filepath in tif_files:
        filename = os.path.basename(filepath)
        print(f" -> Rendering: {filename}")
        
        with rasterio.open(filepath) as src:
            # Calculate the new, lighter dimensions (e.g., factor of 10 = 100m resolution)
            new_height = int(src.height / downsample_factor)
            new_width = int(src.width / downsample_factor)
            
            # Read the data downsampled to save memory
            data = src.read(
                1, 
                out_shape=(1, new_height, new_width),
                resampling=Resampling.nearest
            )
            
            # Get the exact geographic coordinates for this specific tile
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            # Create a blank, transparent RGBA image array
            rgba_image = np.zeros((new_height, new_width, 4), dtype=np.float32)
            
            # 4. Colorize the pixels
            for code, (name, hex_color) in CROP_MAP.items():
                mask = (data == code)
                if np.any(mask):
                    # Convert hex to standard RGB
                    rgb = mcolors.to_rgb(hex_color)
                    rgba_image[mask, :3] = rgb
                    rgba_image[mask, 3] = 1.0  # Make crop pixels fully opaque (Alpha=1)
                    
            # Background/NoData values remain transparent (Alpha=0) so the background shows through
            
            # Plot this specific tile onto the master canvas
            ax.imshow(rgba_image, extent=extent, origin='upper', interpolation='none')

    # 5. Build the Master Legend
    # We build it using our dictionary so it's consistent regardless of which tile loaded first
    legend_patches = []
    for code, (name, hex_color) in CROP_MAP.items():
        legend_patches.append(mpatches.Patch(color=hex_color, label=f"{code} - {name}"))
        
    ax.legend(
        handles=legend_patches, 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        title="Crop Classifications",
        fontsize=10
    )
    
    # Format axes to look clean
    ax.set_xlabel("Easting (EPSG:3035)")
    ax.set_ylabel("Northing (EPSG:3035)")
    ax.ticklabel_format(style='plain') # Prevent scientific notation on coordinates
    
    plt.tight_layout()
    plt.savefig("crop_map.png")
    print("Map rendering complete!")
# --- RUN THE SCRIPT ---
# Replace "." with the path to the folder containing your downloaded files
# E.g., folder_path = "C:/Users/Name/Downloads/Copernicus_Data"
folder_path = "../data_corine/Results-2/" 
plot_consolidated_crop_map(folder_path)