import os
import random

import numpy as np
from noise import pnoise2
from PIL import Image

import setting


def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity):
    noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            noise[i][j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0,
            )
    return noise


def normalize_noise(noise):
    min_val = noise.min()
    max_val = noise.max()
    return (noise - min_val) / (max_val - min_val)


def generate_clouds(width, height, base_scale, octaves, persistence, lacunarity):
    clouds = np.zeros((height, width))
    for octave in range(1, octaves + 1):
        scale = base_scale / octave
        layer = generate_perlin_noise(width, height, scale, 1, persistence, lacunarity)
        clouds += layer * (persistence**octave)

    clouds = normalize_noise(clouds)
    return clouds


def overlay_clouds(image, clouds, alpha=0.5):
    clouds_rgb = np.stack([clouds] * 3, axis=-1)

    image = image.astype(float) / 255.0
    clouds_rgb = clouds_rgb.astype(float)

    blended = image * (1 - alpha) + clouds_rgb * alpha

    blended = (blended * 255).astype(np.uint8)
    return blended


for i in range(len(os.listdir(setting.raw_path))):
    base_scale = random.uniform(50, 200)  # noise frequency
    alpha = random.uniform(0.8, 1.0)  # transparency

    clouds = generate_clouds(
        setting.width,
        setting.height,
        base_scale,
        setting.octaves,
        setting.persistence,
        setting.lacunarity,
    )

    img = np.asarray(Image.open(os.path.join(setting.raw_path, f"IMG_{i + 1}.jpg")))
    image = Image.fromarray(overlay_clouds(img, clouds, alpha))
    image.save(os.path.join(setting.input_path, f"IMG_{i + 1}.jpg"))
    print(f"Đã xử lý {i + 1}/{len(os.listdir(setting.raw_path))}")
