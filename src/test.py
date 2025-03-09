import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    LeakyReLU,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore")


def read_images(data_dir):
    images = list()
    for file_name in glob.glob(data_dir + "/*.jpg"):
        img = image.load_img(file_name)
        img = image.img_to_array(img)
        images.append(img)
    return np.asarray(images) / 255.0


def build_generator(latent_dim):
    # Builds the generator model

    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="generator")

    model.add(Dense(4 * 4 * 1024, kernel_initializer=init, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((4, 4, 1024)))

    model.add(
        Conv2DTranspose(
            512, kernel_size=5, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(
            256, kernel_size=5, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(
            128, kernel_size=5, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(
            3, kernel_size=3, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(Activation("tanh"))

    model.summary()

    return model


def build_discriminator(image_shape=(64, 64, 3)):
    # Builds the generator model

    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="discriminator")

    model.add(
        Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding="same",
            input_shape=image_shape,
            kernel_initializer=init,
        )
    )
    model.add(LeakyReLU(alpha=0.2))

    model.add(
        Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(
        Conv2D(256, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(
        Conv2D(512, kernel_size=5, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=init))
    model.add(Activation("sigmoid"))

    model.summary()

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


class DCGAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def generator_loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the discriminator
        with tf.GradientTape() as discriminator_tape:
            generated_images = self.generator(random_latent_vectors)
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(generated_images)
            d_loss = self.discriminator_loss(real_output, fake_output)
        discriminator_grads = discriminator_tape.gradient(
            d_loss, self.discriminator.trainable_weights
        )
        self.d_optimizer.apply_gradients(
            zip(discriminator_grads, self.discriminator.trainable_weights)
        )
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator
        with tf.GradientTape() as generator_tape:
            generated_images = self.generator(random_latent_vectors)
            fake_output = self.discriminator(generated_images)
            g_loss = self.generator_loss(fake_output)
        generator_grads = generator_tape.gradient(
            g_loss, self.generator.trainable_weights
        )
        self.g_optimizer.apply_gradients(
            zip(generator_grads, self.generator.trainable_weights)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            random_latent_vectors = tf.random.normal(
                shape=(self.num_img, self.latent_dim)
            )
            generated_images = self.model.generator(random_latent_vectors)
            fig = plt.figure(figsize=(10, 4))
            for i in range(self.num_img):
                plt.subplot(2, 5, i + 1)
                plt.imshow(generated_images[i, :, :, :] * 0.5 + 0.5)
                plt.axis("off")
            plt.show()
            plt.close(fig)


latent_dim = 128

generator = build_generator(latent_dim)
discriminator = build_discriminator()

gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
)
X_train = read_images("./data/raw1")
print(X_train.shape)

with tf.device("/GPU:0"):
    history = gan.fit(
        X_train, epochs=100, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
    )

plt.plot(history.history["d_loss"])
plt.plot(history.history["g_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["d_loss", "g_loss"], loc="upper right")
plt.show()

noise = tf.random.normal([16, 128])
generated_images = gan.generator(noise)

fig = plt.figure(figsize=(8, 8))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow((generated_images[i, :, :, :] * 0.5 + 0.5))
    plt.axis("off")
plt.show()
