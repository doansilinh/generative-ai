import torch

batch_size = 40
num_epochs = 10
learning_rate_D = 1e-5
learning_rate_G = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "data"

test_ratio, train_ratio = 0.2, 0.8
rand_seed = 42
img_size = (512, 512)

width, height = 512, 512
octaves = 10
persistence = 0.5
lacunarity = 2
