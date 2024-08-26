import tensorflow as tf
import keras

def cnn_regression(input_shape):
    return tf.keras.models.Sequential([
        keras.layers.Input(input_shape),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(16, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
