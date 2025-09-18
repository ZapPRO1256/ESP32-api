import numpy as np

CLASSES = ["snap", "fall", "lift_rotate", "idle"]
#Генератор синтетичних даних (Python)
def generate_mpu_data(label, n=100):
    data = []
    for _ in range(n):
        if label == "snap":
            # клацання пальцями -> короткий різкий звук + невеликий рух
            acc = np.random.normal(0, 0.2, 3)
            gyro = np.random.normal(0, 0.5, 3)
            audio = np.random.uniform(0.7, 1.0)  # сильний звук
        elif label == "fall":
            # падіння -> різкий пік акселерометра, хаотичні дані
            acc = np.random.normal(0, 2.0, 3) + np.array([0, -9.8, 0])
            gyro = np.random.normal(0, 5.0, 3)
            audio = np.random.uniform(0.2, 0.5)  # звук удару (середній)
        elif label == "lift_rotate":
            # плавний рух + обертання
            acc = np.random.normal(0, 0.5, 3) + np.array([0, 1, 0])
            gyro = np.random.normal(5, 2.0, 3)   # сильні гіроскопічні дані
            audio = np.random.uniform(0.0, 0.2)  # майже беззвучно
        else:  # idle
            acc = np.random.normal(0, 0.05, 3)
            gyro = np.random.normal(0, 0.05, 3)
            audio = np.random.uniform(0.0, 0.05)  # тиша

        sample = np.concatenate([acc, gyro, [audio]])
        data.append(sample)
    return np.array(data)

# Будуємо датасет
X, y = [], []
for idx, cls in enumerate(CLASSES):
    d = generate_mpu_data(cls, n=500)
    X.append(d)
    y += [idx] * len(d)

X = np.vstack(X)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)


import tensorflow as tf
from tensorflow.keras import layers
#Модель (TensorFlow Lite Micro)
model = tf.keras.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

#Після тренування → tflite_convert для ESP32:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("gesture_model.tflite", "wb").write(tflite_model)
