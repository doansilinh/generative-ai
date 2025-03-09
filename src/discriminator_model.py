import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential


def build_discriminator(image_shape=(64, 64, 3)):
    # Builds the discriminator model with improved architecture

    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="discriminator")

    model.add(Input(shape=image_shape))

    # First block
    model.add(
        Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=init,
        )
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Second block
    model.add(
        Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Third block
    model.add(
        Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Fourth block
    model.add(
        Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    # Output
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=init))
    model.add(Activation("sigmoid"))

    model.summary()

    return model
