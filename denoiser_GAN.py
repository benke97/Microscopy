#%%
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import random
from skimage import io
from collections import Counter
import tifffile as tiff
#%%
def is_valid_file(file_name):
    """Check if the file name matches the pattern i_HAADF.tif where i is in the specified range."""
    pattern = re.compile(r"(\d+)_HAADF\.tif")
    match = pattern.match(file_name)

    if not match:
        return False

    number = int(match.group(1))
    return 0 <= number <= 1865 and number != 1234

def extract_number(file_name):
    """Extract the numerical part from the file name."""
    match = re.match(r"(\d+)_HAADF\.tif", file_name)
    return int(match.group(1)) if match else None

# Directory containing your files
directory = "data/simulated"

# List all image files and sort them
image_files = sorted([f for f in os.listdir(directory) if re.match(r"(\d+)_HAADF\.tif", f)], key=extract_number)

# Initialize an empty DataFrame for all data
data_dict = {}

# Process each image and corresponding .pkl file
for image_file in image_files:
    # Load image
    print(image_file)
    image_path = os.path.join(directory, image_file)
    image = Image.open(image_path)

    # Convert image to numpy array and ensure it's 32-bit
    image_array = np.array(image, dtype=np.float32)

    # Extract number from image file name to find corresponding .pkl file
    num = extract_number(image_file)
    pkl_file = os.path.join(directory, f"structure_{num}.pkl")
    # Load dataframe from .pkl file
    with open(pkl_file, 'rb') as file:
        df = pkl.load(file)
    # Add the image to the dataframe
    data_dict[num] = {'dataframe': df, 'image': image_array, 'pixel_size': df['pixel_size'].iloc[0]}
# %%
# go through all the dataframes and plot a histogram of the pixel sizes
pixel_sizes = [data['pixel_size'] for data in data_dict.values()]
# Define the target bin centers
bin_centers = [0.02, 0.025, 0.03, 0.035]

# Function to determine closest bin center for a given pixel size
def closest_bin(pixel_size, bins):
    return bins[np.argmin([abs(pixel_size - bin_val) for bin_val in bins])]

# Assign each pixel size to its closest bin
assigned_bins = [closest_bin(pixel_size, bin_centers) for pixel_size in pixel_sizes]


# Count how many pixel sizes are closest to each of the four values
bin_counts = Counter(assigned_bins)

# Extract counts for each bin center, ensuring all bin centers are present
counts = [bin_counts.get(center, 0) for center in bin_centers]

# Create a bar plot
plt.bar(bin_centers, counts, width=0.0025, color='blue', align='center')
plt.xlabel('Pixel Size')
plt.ylabel('Count')
plt.title('Count of Pixel Sizes Grouped by Closest Bin Center')
plt.xticks(bin_centers)
plt.show()
# %%
bin_dir_map = {0.02: "20 pm", 0.025: "25 pm", 0.03: "30 pm", 0.035: "35 pm"}
for i in range(len(bin_centers)):
    j = 0
    dir = os.path.join("data/experimental", bin_dir_map[bin_centers[i]])
    print(dir)
    if len(os.listdir(dir)) < 10:
        #print("yup")
        list_of_files = os.listdir(dir)
        #sort the list of files based on first number (until "_")
        list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print(list_of_files)
        for file in list_of_files:
            tif_stack = tiff.imread(os.path.join(dir, file))
            print(file, tif_stack.shape)
            for k in range(tif_stack.shape[0]):
                tiff.imsave(os.path.join(dir, f"{j}.tif"), tif_stack[k])
                j += 1
            #read tif stack and save in same folder as single frames

print(counts)
selected_images_dict = {}

for count, bin_dir in zip(counts, bin_dir_map.values()):
    dir_path = os.path.join("data/experimental", bin_dir)
    tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif') and re.match(r'\d+\.tif', f)]

    # Randomly select 'count' number of files
    selected_files = random.sample(tif_files, min(count, len(tif_files)))

    selected_images_dict[bin_dir] = [os.path.join(dir_path, f) for f in selected_files]
print(selected_images_dict)
# %%
