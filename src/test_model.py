import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from gan_model import DCGAN

latent_dim = 128

# Chạy thử đã mô hình đã huấn luyện với GPU
with tf.device("/GPU:0"):
    # Nạp mô hình generator và discriminator
    generator = load_model("./models/generator.keras")
    discriminator = load_model("./models/discriminator.keras")

    # Khởi tạo mô hình GAN
    gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

# Tạo ảnh ngẫu nhiên từ mô hình generator
noise = tf.random.normal([16, 128])
generated_images = gan.generator(noise)

# Lưu kết quả sau khi chạy thử mô hình
os.makedirs("./images", exist_ok=True)
fig = plt.figure(figsize=(8, 8))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((generated_images[i, :, :, :] * 0.5 + 0.5))
    plt.axis("off")
plt.savefig("./images/test_model.png")
