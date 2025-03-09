import os
import shutil

import kagglehub

# Download latest version
path = (
    kagglehub.dataset_download("spandan2/cats-faces-64x64-for-generative-models")
    + "/cats/cats"
)

print("Path to dataset files:", path)

os.makedirs("data/raw1", exist_ok=True)
os.makedirs("data/input", exist_ok=True)

idx = 1

for file in os.listdir(path):
    new_filename = f"IMG_{idx}.jpg"
    old_path = os.path.join(path, file)
    new_path = os.path.join("data/raw1", new_filename)
    shutil.move(old_path, new_path)
    idx += 1
