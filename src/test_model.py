import tensorflow as tf

model = tf.keras.models.load_model("save.keras")
print(model.summary())
