import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Класи
CLASSES = ["snap", "fall", "lift_rotate", "idle"]

# Завантаження TFLite-моделі
interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(sample):
    sample = np.array(sample, dtype=np.float32).reshape(1, 7)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_id = int(np.argmax(output_data))
    return CLASSES[class_id], float(np.max(output_data))

# Flask API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    # очікуємо {"input":[...7 значень...]}
    if "input" not in data or len(data["input"]) != 7:
        return jsonify({"error": "expected 7 input values"}), 400
    label, confidence = predict(data["input"])
    return jsonify({"prediction": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
