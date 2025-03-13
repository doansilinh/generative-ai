import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential


# Xây dựng mô hình discriminator
def build_discriminator(image_shape=(64, 64, 3)):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential(name="discriminator")

    # Đầu vào của mô hình là một ảnh
    model.add(Input(shape=image_shape))
    model.add(GaussianNoise(0.1))

    # Khối đầu tiên
    model.add(
        Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=init,
        )
    )  # Convolutional layer với filter = 64 ,kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    # model.add(BatchNormalization(momentum=0.8))  # Chuẩn hóa dữ liệu
    model.add(LeakyReLU(negative_slope=0.2))  # Hàm kích hoạt LeakyReLU
    model.add(Dropout(0.25))  # Loại bỏ 25% dữ liệu để tránh overfitting

    # Khối thứ hai
    model.add(
        Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )  # Convolutional layer với filter = 128 ,kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    # model.add(BatchNormalization(momentum=0.8))  # Chuẩn hóa dữ liệu
    model.add(LeakyReLU(negative_slope=0.2))  # Hàm kích hoạt LeakyReLU
    model.add(Dropout(0.25))  # Loại bỏ 25% dữ liệu để tránh overfitting

    # Khối thứ ba
    model.add(
        Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )  # Convolutional layer với filter = 265 ,kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    # model.add(BatchNormalization(momentum=0.8))  # Chuẩn hóa dữ liệu
    model.add(LeakyReLU(negative_slope=0.2))  # Hàm kích hoạt LeakyReLU
    model.add(Dropout(0.25))  # Loại bỏ 30% dữ liệu để tránh overfitting

    # Khối thứ tư
    model.add(
        Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=init)
    )  # Convolutional layer với filter = 512 ,kích thước kernel = 4, bước nhảy kernel = 2, giữ nguyên kích thước đầu ra
    # model.add(BatchNormalization(momentum=0.8))  # Chuẩn hóa dữ liệu
    model.add(LeakyReLU(negative_slope=0.2))  # Hàm kích hoạt LeakyReLU
    model.add(Dropout(0.25))  # Loại bỏ 25% dữ liệu để tránh overfitting

    # Đầu ra của mô hình là một giá trị từ 0 đến 1 để nhận biết ảnh thật và ảnh giả
    model.add(Flatten())  # Làm phẳng dữ liệu
    model.add(Dense(1, kernel_initializer=init))  # Lớp Dense với 1 nơ-ron đầu ra
    model.add(Activation("sigmoid"))  # Hàm kích hoạt sigmoid

    model.summary()

    return model
