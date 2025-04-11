import os
import numpy as np
from skimage.transform import resize
import rasterio

def load_tiff_files(folder_path, target_shape=(64, 64)):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif"):
                filepath = os.path.join(root, file)
                with rasterio.open(filepath) as src:
                    img = src.read(1)
                    resized_img = resize(img, target_shape, mode='reflect', anti_aliasing=True)
                    images.append(resized_img.flatten())
                    if "MASK" in root.upper():  # Case insensitive
                        labels.append(1)
                    else:
                        labels.append(0)
    return np.array(images), np.array(labels)
