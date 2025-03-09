import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from gan_model import DCGAN

os.makedirs("./images", exist_ok=True)
latent_dim = 128

with tf.device("/GPU:0"):
    generator = load_model("./model/generator.keras")
    discriminator = load_model("./model/discriminator.keras")

    gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)

noise = tf.random.normal([16, 128])
generated_images = gan.generator(noise)

fig = plt.figure(figsize=(8, 8))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((generated_images[i, :, :, :] * 0.5 + 0.5))
    plt.axis("off")
plt.savefig("./images/test.png")
