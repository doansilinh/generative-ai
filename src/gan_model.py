import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

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
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
