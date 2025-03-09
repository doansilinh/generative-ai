import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2DTranspose,
    Dense,
    Dropout,
    Input,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential


def build_generator(latent_dim):
    # Builds the generator model with improved architecture

    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="generator")

    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(4 * 4 * 1024, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(ReLU())  # ReLU thay vì LeakyReLU trong generator
    model.add(Reshape((4, 4, 1024)))

    # First upsampling block
    model.add(
        Conv2DTranspose(
            512, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    # Second upsampling block
    model.add(
        Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    # Third upsampling block
    model.add(
        Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    # Output layer
    model.add(
        Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )
    model.add(Activation("tanh"))

    model.summary()

    return model
