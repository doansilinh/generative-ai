import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from discriminator_model import build_discriminator
from gan_model import DCGAN
from generator_model import build_generator
from load_data import read_images
from setting import d_learning_rate, epochs, g_learning_rate

os.makedirs("./images", exist_ok=True)
os.makedirs("./models", exist_ok=True)


class DCGANCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch - 9) % 10 == 0:
            self.model.generator.save(f"./models/generator_{epoch + 1}_epoch.keras")


latent_dim = 128

# Huấn luyện mô hình với GPU
start = time.time()
with tf.device("/GPU:0"):
    # Khởi tạo mô hình generator và discriminator
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Khởi tạo mô hình GAN
    gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=Adam(learning_rate=d_learning_rate, beta_1=0.5),
        g_optimizer=Adam(learning_rate=g_learning_rate, beta_1=0.5),
    )

    # Đọc dữ liệu và huấn luyện mô hình
    X_train = read_images("./data")
    history = gan.fit(X_train, epochs=epochs, callbacks=[DCGANCallback()])
end = time.time()
execution_time = end - start

print(
    f"Tổng thời gian chạy của mô hình là {execution_time:.2f}s, trung bình mỗi epoch là {(execution_time / epochs):.2f}s"
)

# Lưu kết quả huấn luyện
plt.plot(history.history["d_loss"])
plt.plot(history.history["g_loss"])
plt.title("Đánh giá mô hình")
plt.xlabel("Epoch")
plt.legend(["d_loss", "g_loss"], loc="upper right")
plt.savefig("./images/result_model.png")
