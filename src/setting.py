import torch

batch_size = 1
num_epochs = 10
learning_rate_D = 1e-5
learning_rate_G = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_path = "data/raw"
input_path = "data/input"

test_ratio, train_ratio = 0.2, 0.8
rand_seed = 42
img_size = (512, 512)

width, height = 512, 512
octaves = 10  # number of noise layers combined
persistence = 0.5  # lower persistence reduces the amplitude of higher-frequency octaves
lacunarity = (
    1.5  # higher lacunarity increases the frequency of higher-frequency octaves
)
