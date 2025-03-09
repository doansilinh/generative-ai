import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from discriminator_model import build_discriminator
from gan_model import DCGAN
from generator_model import build_generator
from load_data import read_images

latent_dim = 128

generator = build_generator(latent_dim)
discriminator = build_discriminator()

gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
)
X_train = read_images("./data")

with tf.device("/GPU:0"):
    history = gan.fit(X_train, epochs=100)

os.makedirs(".model", exist_ok=True)
generator.save("./model/generator.keras")
discriminator.save("./model/discriminator.keras")

plt.plot(history.history["d_loss"])
plt.plot(history.history["g_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["d_loss", "g_loss"], loc="upper right")
plt.savefig("./images/result.png")

noise = tf.random.normal([16, 128])
generated_images = gan.generator(noise)

fig = plt.figure(figsize=(8, 8))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((generated_images[i, :, :, :] * 0.5 + 0.5))
    plt.axis("off")
plt.show()
