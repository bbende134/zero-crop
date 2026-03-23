#%%
import os
import torch
import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
# from plotting_utils import plot_s2, plot_modality
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.tasks.tiled_inference import tiled_inference
from terratorch import ge

# Set PyTorch CUDA memory configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#%%
# Select device
if torch.cuda.is_available():
    device = 'cuda'    
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Using device: {device}')
# %%
# if not os.path.isfile('examples/S2L2A/Santiago.tif'):
#     hf_hub_download(repo_id='ibm-esa-geospatial/Examples', filename='S2L2A/Santiago.tif', repo_type='dataset', local_dir='examples/')
# # %%
# # Download Singapore large-scale example from Hugging Face (2000x2000 pixel)
# if not os.path.isfile('examples/S2L2A/Singapore_2025-01-09.tif'):
#     hf_hub_download(repo_id='ibm-esa-geospatial/Examples', filename='S2L2A/Singapore_2025-01-09.tif', repo_type='dataset', local_dir='examples/')
# %%

# Loading local files

base_dir = '/home/bende/map_data/local_data'

import glob

year_pattern = '2025-04*'

landsat_dir_base = os.path.join(base_dir, 'LANDSAT_L2')
s1_dir_base = os.path.join(base_dir, 'S1RTC')
s2_dir_base = os.path.join(base_dir, 'S2L2A')

landsat_dirs = glob.glob(os.path.join(landsat_dir_base, year_pattern))
s1_dirs = glob.glob(os.path.join(s1_dir_base, year_pattern))
s2_dirs = glob.glob(os.path.join(s2_dir_base, year_pattern))

print(f"Found {len(landsat_dirs)} Landsat directories")
print(f"Found {len(s1_dirs)} S1 directories")
print(f"Found {len(s2_dirs)} S2 directories")
# %%
# Tile inspection
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.plotting_utils import plot_s2

file_info = os.path.join(s2_dirs[0], 'S2L2A_tile0.tif') 
da = rxr.open_rasterio(file_info)
print(f"Processing: {file_info}")
print(f"DataArray shape: {da.shape}")
print(f"Band coordinates: {da.band.values}")
data = da.values
#%%

plot_s2(data)
#%%

# Processing files

# Tile the input data into 32x32 patches (16x16 might be too small for the model's downsampling/internal tensors)
print("Tiling input data into 224x224 patches...")
patch_size = 224
c, h, w = data.shape

# Ensure dimensions are divisible by patch_size
h_new = (h // patch_size) * patch_size
w_new = (w // patch_size) * patch_size
data_cropped = data[:, :h_new, :w_new]

# Reshape to (N, C, P, P)
num_patches_h = h_new // patch_size
num_patches_w = w_new // patch_size
patches = data_cropped.reshape(c, num_patches_h, patch_size, num_patches_w, patch_size)
patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, c, patch_size, patch_size)

print(f"Created {patches.shape[0]} patches of size {patch_size}x{patch_size}")

# Select a subset for demonstration (e.g., first 1 patch)
batch_size = 1
input_patches = patches[:batch_size]
# input_patches = np.repeat(input_patches, 2, axis=0) # Reverting duplication
print(f"Running embedding model on {input_patches.shape[0]} patch...")

# Use patches as the "original data" for embedding
original_data = input_patches.copy()
plot_s2(original_data)
#%%
# Embeddings for original data
print("Computing embeddings for patches...")
data_dict_original = {'S2L2A': torch.tensor(original_data, dtype=torch.float, device='cpu')}

# Verify the tensor we're about to feed to the model
print(f"Original tensor to model - shape: {data_dict_original['S2L2A'].shape}")
print(f"Original tensor to model - mean: {data_dict_original['S2L2A'].mean().item():.6f}, std: {data_dict_original['S2L2A'].std().item():.6f}")

embedding_model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    modalities=['S2L2A'],
    output_modalities=['S1RTC', 'DEM', 'LULC', 'NDVI'],
    pretrained=True,
    standardize=True,
)
embedding_model = embedding_model.to(device)

input_for_embedding = data_dict_original['S2L2A'].clone().to(device)
print(f"Input tensor on device - mean: {input_for_embedding.mean().item():.6f}, std: {input_for_embedding.std().item():.6f}")

with torch.no_grad():
    # Run model on the batch of patches
    _ = embedding_model(input_for_embedding, verbose=False, timesteps=10)

encoder_embeddings_original = embedding_model.encoder_embeddings
main_embedding_key = list(encoder_embeddings_original.keys())[0]
embedding_obj_original = encoder_embeddings_original[main_embedding_key]
embeddings_original = embedding_obj_original.pos_emb.cpu().clone()  # Move to CPU to save GPU memory
print(f"Original embeddings shape: {embeddings_original.shape}")
print(f"Original embeddings mean: {embeddings_original.mean().item():.6f}")
print(f"Original embeddings std: {embeddings_original.std().item():.6f}")

# Verify original data stats
print(f"Original data mean: {original_data.mean():.6f}")
print(f"Original data std: {original_data.std():.6f}")
#%%
# Clear memory
del input_for_embedding
torch.cuda.empty_cache()
#         plot_modality(m, out, ax=axes[k][j])
#         axes[k][j].axis('off')
#         
# plt.tight_layout()
# plt.savefig(f'any_to_any_{os.path.basename(file)}.pdf')
# plt.show()
#
# # %%
# # Get embeddings from the model for the input
# # Use the last model from the loop (or you can create a specific model)
# # (Embeddings already computed above for both original and modified data)
# %%

# Saving embeddings into vec db
