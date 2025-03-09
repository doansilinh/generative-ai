import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential


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
