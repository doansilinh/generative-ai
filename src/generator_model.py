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


# Xây dựng mô hình generator
def build_generator(latent_dim):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="generator")

    # Đầu vào của mô hình
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(4 * 4 * 1024, kernel_initializer=init))
    model.add(BatchNormalization())  # Chuẩn hóa dữ liệu
    model.add(ReLU())  # Hàm kích hoạt ReLU
    model.add(Reshape((4, 4, 1024)))  # Định hình lại dữ liệu đầu vào thành 4x4x1024

    # Khối đầu tiên
    model.add(
        Conv2DTranspose(
            512, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )  # Transposed convolutional layer với filter = 512, kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    model.add(BatchNormalization())  # Chuẩn hóa dữ liệu
    model.add(ReLU())  # Hàm kích hoạt ReLU
    model.add(Dropout(0.3))  # Loại bỏ 30% dữ liệu để tránh overfitting

    # Khối thứ hai
    model.add(
        Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )  # Transposed convolutional layer với filter = 256, kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    model.add(BatchNormalization())  # Chuẩn hóa dữ liệu
    model.add(ReLU())  # Hàm kích hoạt ReLU
    model.add(Dropout(0.3))  # Loại bỏ 30% dữ liệu để tránh overfitting

    # Khối thứ 3
    model.add(
        Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )  # Transposed convolutional layer với filter = 128, kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    model.add(BatchNormalization())  # Chuẩn hóa dữ liệu
    model.add(ReLU())  # Hàm kích hoạt ReLU
    model.add(Dropout(0.3))  # Loại bỏ 30% dữ liệu để tránh overfitting

    # Đầu ra của mô hình
    model.add(
        Conv2DTranspose(
            3, kernel_size=4, strides=2, padding="same", kernel_initializer=init
        )
    )  # Transposed convolutional layer với filter = 3, kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    model.add(Activation("tanh"))  # Hàm kích hoạt tanh

    model.summary()

    return model
