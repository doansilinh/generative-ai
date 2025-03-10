import glob

import numpy as np
from tensorflow.keras.preprocessing import image


# Hàm đọc dữ liệu ảnh từ thư mục
def read_images(data_dir):
    images = list()
    for file_name in glob.glob(data_dir + "/*.jpg"):
        img = image.load_img(file_name)
        img = image.img_to_array(img)
        images.append(img)
    return np.asarray(images) / 255.0
