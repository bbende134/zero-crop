#%% Import Required Libraries
# Import libraries such as rasterio, numpy, and matplotlib for geospatial data manipulation and visualization.

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def first_tif_in_dir(directory, keyword=None):
    files = sorted(glob.glob(os.path.join(directory, '**', '*.tif'), recursive=True))
    if keyword:
        files = [f for f in files if keyword.lower() in f.lower()]
    if not files:
        raise FileNotFoundError(f"No .tif files found in directory: {directory} with keyword: {keyword}")
    return files[0]

#%% Load the Data
# Use directory paths (not exact filenames). The helper picks the first .tif in each directory.

dem_dir = '/home/bende/dev/proc_inspect_lechner/BNPI_data'
dem_path = first_tif_in_dir(dem_dir, 'dem')

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    dem_meta = src.meta
    dem_crs = src.crs
    dem_bounds = src.bounds

#%% View Data Structure
# View the shape, CRS, and bounds of the DEM data.

print(f"DEM Shape: {dem.shape}")
print(f"DEM CRS: {dem_crs}")
print(f"DEM Bounds: {dem_bounds}")
print(f"DEM Meta: {dem_meta}")

#%% Inspect Data Types
# Check the data type of the DEM array.

print(f"DEM dtype: {dem.dtype}")

#%% Check for Missing Values
# Identify missing (nodata) values in the DEM.

nodata = dem_meta.get('nodata', None)
print(f"Nodata value: {nodata}")
if nodata is not None:
    missing_count = np.sum(dem == nodata)
    print(f"Number of missing values: {missing_count}")
else:
    print("No nodata value specified.")

#%% Generate Summary Statistics
# Compute descriptive statistics for the DEM values.

if nodata is not None:
    valid_dem = dem[dem != nodata]
else:
    valid_dem = dem.flatten()

print(f"Min: {np.min(valid_dem)}")
print(f"Max: {np.max(valid_dem)}")
print(f"Mean: {np.mean(valid_dem)}")
print(f"Std: {np.std(valid_dem)}")
print(f"Median: {np.median(valid_dem)}")

#%% Visualize Data Distributions
# Create a histogram of the DEM values to visualize the elevation distribution.

plt.figure(figsize=(10, 6))
plt.hist(valid_dem, bins=50, edgecolor='black')
plt.title('DEM Elevation Distribution')
plt.xlabel('Elevation (m)')
plt.ylabel('Frequency')
plt.show()

#%% Visualize the DEM Raster
# Display the DEM as an image.

plt.figure(figsize=(10, 10))
plt.imshow(dem, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Digital Elevation Model (DEM)')
plt.savefig('dem.png')

#%% Load and Visualize Satellite Imagery
# Use directory path for Sentinel-2 (pick first .tif found).

sentinel_dir = '/home/bende/dev/proc_inspect_lechner/BNPI_data'
sentinel_path = first_tif_in_dir(sentinel_dir, 'sentinel')

with rasterio.open(sentinel_path) as src:
    sentinel = src.read()
    sentinel_meta = src.meta

# Assuming bands 4,3,2 for RGB (true color)
if sentinel.shape[0] >= 3:
    rgb = np.stack([sentinel[2], sentinel[1], sentinel[0]], axis=-1)  # B4, B3, B2 (assuming bands ordered as B2, B3, B4)
    rgb = np.clip(rgb / 3000, 0, 1)  # Normalize for display
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title('Sentinel-2 RGB Composite')
    plt.savefig('sentinel.png')
else:
    print("Not enough bands for RGB.")

#%% Load and Visualize Thematic Data - Ecosystem Map
eco_dir = '/home/bende/dev/proc_inspect_lechner/BNPI_data'
eco_path = first_tif_in_dir(eco_dir, 'ecosystem')

with rasterio.open(eco_path) as src:
    eco = src.read(1)
    eco_meta = src.meta

plt.figure(figsize=(10, 10))
plt.imshow(eco, cmap='viridis')
plt.colorbar()
plt.title('Ecosystem Map')
plt.savefig('eco.png')

#%% Load and Visualize Thematic Data - Grass Croptype
grass_dir = '/home/bende/dev/proc_inspect_lechner/BNPI_data'
grass_path = first_tif_in_dir(grass_dir, 'grass')

with rasterio.open(grass_path) as src:
    grass = src.read(1)
    grass_meta = src.meta

plt.figure(figsize=(10, 10))
plt.imshow(grass, cmap='Set3')
plt.colorbar()
plt.title('Grass Croptype')
plt.savefig('grass.png')

#%% Load and Visualize Thematic Data - NHRL
nhrl_dir = '/home/bende/dev/proc_inspect_lechner/BNPI_data'
nhrl_path = first_tif_in_dir(nhrl_dir, 'nhrl')

with rasterio.open(nhrl_path) as src:
    nhrl = src.read(1)
    nhrl_meta = src.meta

plt.figure(figsize=(10, 10))
plt.imshow(nhrl, cmap='plasma')
plt.colorbar()
plt.title('NHRL')
plt.savefig('nhrl.png')

# %%
