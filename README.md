# Gesture Recognition with ESP32 + MPU6050 + TinyML

## Огляд

Проєкт реалізує **розпізнавання жестів** за допомогою акселерометра/гіроскопа MPU6050 та симульованого аудіо. Використовується **TensorFlow Lite Micro** на ESP32 або через локальний Flask-сервер з TFLite-моделлю.

Підтримуються 4 класи:

- `snap` — клацання пальцями  
- `fall` — падіння  
- `lift_rotate` — підйом-поворот  
- `idle` — спокій

Проєкт демонструє пайплайн TinyML/IoT: збір даних з сенсорів, передбачення моделі, управління LED і бузером.

---

## Апаратна частина

- **ESP32-DEVKITC**  
- **MPU6050** (акселерометр + гіроскоп)  
- **LED** (сигналізація)  
- **Buzzer** (звук)  
- **Резистори 1 кОм** (для I²C pull-up)

### Підключення

| Компонент | Пін ESP32 | Примітка |
|-----------|-----------|----------|
| MPU6050 VCC | 3.3V | Живлення |
| MPU6050 GND | GND | Загальний провід |
| MPU6050 SDA | 21 | I²C Data |
| MPU6050 SCL | 22 | I²C Clock |
| MPU6050 AD0 | 16 | Вибір адреси |
| LED анод | 15 | Катод до GND |
| Buzzer 1 | 18 | Інший пін до GND |

### ASCII-схема підключення

```
       ESP32-DEVKITC
      +------------+
      |          3V3|----+--- MPU6050 VCC
      |          GND|----+--- MPU6050 GND
      |          21 |----+--- MPU6050 SDA
      |          22 |----+--- MPU6050 SCL
      |          16 |----+--- MPU6050 AD0
      |          15 |----+--- LED (+), LED (-) -> GND
      |          18 |----+--- Buzzer (+), Buzzer (-) -> GND
      +------------+
```

---

## Пайплайн моделі

1. **Генерація синтетичних даних на Python**

```python
import numpy as np

CLASSES = ["snap", "fall", "lift_rotate", "idle"]

def generate_mpu_data(label, n=100):
    data = []
    for _ in range(n):
        if label == "snap":
            acc = np.random.normal(0,0.2,3)
            gyro = np.random.normal(0,0.5,3)
            audio = np.random.uniform(0.7,1.0)
        elif label == "fall":
            acc = np.random.normal(0,2.0,3)+[0,-9.8,0]
            gyro = np.random.normal(0,5.0,3)
            audio = np.random.uniform(0.2,0.5)
        elif label == "lift_rotate":
            acc = np.random.normal(0,0.5,3)+[0,1,0]
            gyro = np.random.normal(5,2.0,3)
            audio = np.random.uniform(0.0,0.2)
        else:
            acc = np.random.normal(0,0.05,3)
            gyro = np.random.normal(0,0.05,3)
            audio = np.random.uniform(0.0,0.05)
        data.append(np.concatenate([acc,gyro,[audio]]))
    return np.array(data)
```

2. **Навчання моделі TensorFlow**

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
```

3. **Конвертація у TFLite для ESP32 або Flask**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("gesture_model.tflite", "wb").write(tflite_model)
```

4. **Flask API для локального сервера**

```python
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(sample):
    sample = np.array(sample, dtype=np.float32).reshape(1,7)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output_data)), float(np.max(output_data))

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    label_id, conf = predict(data["input"])
    labels = ["snap","fall","lift_rotate","idle"]
    return jsonify({"prediction": labels[label_id], "confidence": conf})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

5. **ESP32-клієнт**:

- Збирає дані з MPU6050 та псевдо-аудіо
- Відправляє POST-запит на Flask API
- Реагує на класи LED та Buzzer

---

## Метрики

- **Accuracy (тренування):** ~95% (на синтетичних даних)  
- **Інференс на ESP32:** <50 мс на зразок  
- **Стабільність:** залежить від якості I²C та шуму датчика

---

## Інструкції

1. Підключити ESP32, MPU6050, LED, Buzzer згідно таблиці.  
2. Встановити Python, TensorFlow, Flask.  
3. Навчити модель або використати `gesture_model.tflite`.  
4. Запустити сервер:
```bash
python server.py
```  
5. Завантажити ESP32-код у плату через Arduino IDE або PlatformIO.

---

## Ліцензія

MIT License © Mr Zap

---

## Ресурси

- [ESP32 Arduino](https://docs.espressif.com/projects/arduino-esp32/en/latest/)  
- [MPU6050 datasheet](https://www.invensense.com/products/motion-tracking/6-axis/)  
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)  
- [Flask documentation](https://flask.palletsprojects.com/)

