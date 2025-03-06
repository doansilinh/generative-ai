import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/animal-faces") + "\\afhq"

print("Path to dataset files:", path)

animals = ["cat", "dog", "wild"]
for animal in animals:
    os.makedirs(os.path.join("data/raw", animal), exist_ok=True)
    os.makedirs(os.path.join("data/input", animal), exist_ok=True)

cat_idx = 1
dog_idx = 1
wild_idx = 1

for f1 in os.listdir(path):
    path1 = os.path.join(path, f1)
    for f2 in os.listdir(path1):
        path2 = os.path.join(path1, f2)
        match f2:
            case "cat":
                for img in os.listdir(path2):
                    new_filename = f"IMG_{cat_idx}.jpg"
                    old_path = os.path.join(path2, img)
                    new_path = os.path.join("data/raw/cat", new_filename)
                    shutil.move(old_path, new_path)
                    cat_idx += 1
            case "dog":
                for img in os.listdir(path2):
                    new_filename = f"IMG_{dog_idx}.jpg"
                    old_path = os.path.join(path2, img)
                    new_path = os.path.join("data/raw/dog", new_filename)
                    shutil.move(old_path, new_path)
                    dog_idx += 1
            case "wild":
                for img in os.listdir(path2):
                    new_filename = f"IMG_{wild_idx}.jpg"
                    old_path = os.path.join(path2, img)
                    new_path = os.path.join("data/raw/wild", new_filename)
                    shutil.move(old_path, new_path)
                    wild_idx += 1
