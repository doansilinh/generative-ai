import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2DTranspose,
    Dense,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Sequential


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
