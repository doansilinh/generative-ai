import os
import shutil

import kagglehub

# Download latest version
path = kagglehub.dataset_download("spandan2/cats-faces-64x64-for-generative-models")
dataset_path = path + "/cats/cats"


print("Path to dataset files:", path)

os.makedirs("data", exist_ok=True)

idx = 1

for file in os.listdir(dataset_path):
    new_filename = f"IMG_{idx}.jpg"
    old_path = os.path.join(dataset_path, file)
    new_path = os.path.join("data", new_filename)
    shutil.move(old_path, new_path)
    idx += 1

shutil.rmtree(path)
