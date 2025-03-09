import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model


# Using binary cross entropy with label smoothing
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


class DCGAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = 1  # Number of discriminator training steps per generator step
        self.use_wgan = False  # Set to True to use Wasserstein GAN loss

    def compile(self, d_optimizer, g_optimizer, use_wgan=False):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.use_wgan = use_wgan

        # Additional metrics for monitoring
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.fake_accuracy = keras.metrics.BinaryAccuracy(name="fake_acc")

    def generator_loss(self, fake_output):
        if self.use_wgan:
            # For WGAN, we want to maximize the critic's output for fake images
            return -tf.reduce_mean(fake_output)
        else:
            # Use label smoothing (0.9 instead of 1.0)
            return tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(fake_output) * 0.9, fake_output
            )

    def discriminator_loss(self, real_output, fake_output):
        if self.use_wgan:
            # For WGAN, we want to maximize the difference between real and fake outputs
            return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        else:
            # Use label smoothing (0.9 instead of 1.0 for real samples)
            real_loss = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(real_output) * 0.9, real_output
            )
            fake_loss = tf.keras.losses.BinaryCrossentropy()(
                tf.zeros_like(fake_output), fake_output
            )
            total_loss = real_loss + fake_loss

            # Update accuracy metrics
            self.real_accuracy.update_state(tf.ones_like(real_output), real_output)
            self.fake_accuracy.update_state(tf.zeros_like(fake_output), fake_output)

            return total_loss

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Train discriminator multiple steps
        d_loss_value = 0
        for _ in range(self.d_steps):
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as discriminator_tape:
                generated_images = self.generator(random_latent_vectors, training=True)
                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(generated_images, training=True)
                d_loss = self.discriminator_loss(real_output, fake_output)

            discriminator_grads = discriminator_tape.gradient(
                d_loss, self.discriminator.trainable_weights
            )
            self.d_optimizer.apply_gradients(
                zip(discriminator_grads, self.discriminator.trainable_weights)
            )
            d_loss_value += d_loss

        # Average discriminator loss
        d_loss_value /= self.d_steps

        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as generator_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            g_loss = self.generator_loss(fake_output)

        generator_grads = generator_tape.gradient(
            g_loss, self.generator.trainable_weights
        )
        self.g_optimizer.apply_gradients(
            zip(generator_grads, self.generator.trainable_weights)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss_value)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "real_acc": self.real_accuracy.result(),
            "fake_acc": self.fake_accuracy.result(),
        }
