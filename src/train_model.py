import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from discriminator_model import build_discriminator
from gan_model import DCGAN
from generator_model import build_generator
from load_data import read_images
from setting import batch_size, d_learning_rate, epochs, g_learning_rate

latent_dim = 128

with tf.device("/GPU:0"):
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=Adam(learning_rate=d_learning_rate, beta_1=0.5),
        g_optimizer=Adam(learning_rate=g_learning_rate, beta_1=0.5),
    )
    X_train = read_images("./data")
    history = gan.fit(X_train, epochs=epochs, batch_size=batch_size)

os.makedirs("./model", exist_ok=True)
generator.save("./model/generator.keras")
discriminator.save("./model/discriminator.keras")

os.makedirs("./images", exist_ok=True)
plt.plot(history.history["d_loss"])
plt.plot(history.history["g_loss"])
plt.plot(history.history["fake_acc"])
plt.plot(history.history["real_acc"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["d_loss", "g_loss", "fake_acc", "real_acc"], loc="upper right")
plt.savefig("./images/result.png")
