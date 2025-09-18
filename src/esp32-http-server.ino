#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <MPU6050_light.h>

// --- Wi-Fi ---
const char* SSID = "Wokwi-GUEST";
const char* password = "";

// --- API (Flask-сервер на Python) ---
const char* MODEL_URL = "http://192.168.0.243:5000/predict";  
// заміни IP на адресу сервера, де крутиться Flask

// --- MPU6050 ---
MPU6050 mpu(Wire);

// --- Піни ---
const int LED_PIN = 15;
const int BUZZER_PIN = 18;

// --- Стан ---
String lastPrediction = "none";

unsigned long previousMillis = 0;
const unsigned long interval = 1000; // кожну секунду

// --- Генерація псевдозвукового сигналу ---
float simulateAudioFeature() {
  // шум від 0 до 1, іноді "сплеск"
  float base = random(0, 100) / 100.0;
  if (random(0, 20) == 0) {
    return base + 3.0;  // клацання
  }
  return base;
}

// --- Запит до Flask-моделі ---
String sendToModel(float accX, float accY, float accZ,
                   float gyroX, float gyroY, float gyroZ,
                   float audio) {
  HTTPClient http;
  http.begin(MODEL_URL);
  http.addHeader("Content-Type", "application/json");

  // JSON для Flask: { "input": [..7..] }
  StaticJsonDocument<256> doc;
  JsonArray arr = doc.createNestedArray("input");
  arr.add(accX);
  arr.add(accY);
  arr.add(accZ);
  arr.add(gyroX);
  arr.add(gyroY);
  arr.add(gyroZ);
  arr.add(audio);

  String body;
  serializeJson(doc, body);

  int httpResponseCode = http.POST(body);
  String result = "error";

  if (httpResponseCode == 200) {
    String payload = http.getString();
    StaticJsonDocument<256> resp;
    deserializeJson(resp, payload);

    // Читаємо поле "prediction" з Flask
    result = resp["prediction"].as<String>();
  } else {
    Serial.print("Error: ");
    Serial.println(httpResponseCode);
  }

  http.end();
  return result;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  WiFi.begin(SSID, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Connected, IP: ");
  Serial.println(WiFi.localIP());

  Wire.begin();
  mpu.begin();
  mpu.calcOffsets(true, true);
}

void loop() {
  mpu.update();
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    float accX = mpu.getAccX();
    float accY = mpu.getAccY();
    float accZ = mpu.getAccZ();
    float gyroX = mpu.getGyroX();
    float gyroY = mpu.getGyroY();
    float gyroZ = mpu.getGyroZ();
    float audio = simulateAudioFeature();

    String prediction = sendToModel(accX, accY, accZ,
                                    gyroX, gyroY, gyroZ,
                                    audio);
    lastPrediction = prediction;
    Serial.println("Predicted: " + prediction);

    // Реакція
    if (prediction == "snap") {               // клацання
      digitalWrite(LED_PIN, HIGH);
      tone(BUZZER_PIN,200,200);
      delay(200);
      digitalWrite(LED_PIN, LOW);
    } else if (prediction == "fall") {        // падіння
      tone(BUZZER_PIN,500,500);
      delay(500);
    } else if (prediction == "lift_rotate") { // підйом-поворот
      digitalWrite(LED_PIN, HIGH);
      delay(300);
      digitalWrite(LED_PIN, LOW);
    } else {
      // idle -> нічого не робимо
    }
  }
}
