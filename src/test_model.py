import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import data_loader
import g_model
import setting

# Count total data
num_test = int(len(os.listdir(setting.path_target)) * setting.test_ratio)
num_train = int((int(len(os.listdir(setting.path_target))) - num_test))

print("Number of train samples:", num_train)
print("Number of test samples:", num_test)

# Train , test split
random.seed(setting.rand_seed)
train_idxs = np.array(random.sample(range(num_test + num_train), num_train))
mask = np.ones(num_train + num_test, dtype=bool)
mask[train_idxs] = False

inputs = ["IMG_{}.jpg".format(i) for i in range(1, 27001)]
raws = ["IMG_{}.jpg".format(i) for i in range(1, 27001)]
train_input_img_paths = np.array(inputs)[train_idxs]
train_raw_img_path = np.array(raws)[train_idxs]
test_input_img_paths = np.array(inputs)[mask]
test_raw_img_path = np.array(raws)[mask]
# Test after train
random.Random(setting.rand_seed).shuffle(test_raw_img_path)
random.Random(setting.rand_seed).shuffle(test_input_img_paths)
subset_loader = data_loader.dataset(
    batch_size=5,
    img_size=setting.img_size,
    input_image_paths=test_input_img_paths,
    raw_image_paths=test_raw_img_path,
)

generator = g_model.GModel()
generator.load_state_dict(torch.load("models/generator.pth"))
generator.eval()
for X, y in subset_loader:
    fig, axes = plt.subplots(5, 3, figsize=(9, 9))

    for i in range(5):
        axes[i, 0].imshow(np.transpose(X.numpy()[i], (1, 2, 0)))
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.transpose(y.numpy()[i], (1, 2, 0)))
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")

        generated_image = generator(X[i].unsqueeze(0)).detach().numpy()[0]
        axes[i, 2].imshow(np.transpose(generated_image, (1, 2, 0)))
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.savefig("Test.jpg")
    plt.show()
    break
