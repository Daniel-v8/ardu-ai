#include <Arduino.h>
#include "config.h"
#include "camera.h"
#include "inference.h"
#include "mavlink_comm.h"

static int8_t input_buf[MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_CH];
static uint32_t last_inference_ms = 0;

static const char* class_names[] = {"LEFT", "STRAIGHT", "RIGHT"};

static uint16_t class_to_pwm(int cls) {
    switch (cls) {
        case CLASS_LEFT:     return RC_STEER_LEFT;
        case CLASS_RIGHT:    return RC_STEER_RIGHT;
        default:             return RC_STEER_CENTER;
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== ArduRevo ESP32-Cam ===");

    // Init camera
    if (!camera_init()) {
        Serial.println("FATAL: Camera init failed. Halting.");
        while (true) delay(1000);
    }

    // Init TFLite model
    if (!inference_init()) {
        Serial.println("FATAL: Inference init failed. Halting.");
        while (true) delay(1000);
    }

    // Init MAVLink
    mavlink_init();

    Serial.println("Setup complete. Starting main loop.");
}

void loop() {
    // Always process MAVLink (heartbeats, incoming messages)
    mavlink_update();

    // Run inference at configured rate
    uint32_t now = millis();
    if (now - last_inference_ms < INFERENCE_INTERVAL_MS) {
        return;
    }
    last_inference_ms = now;

    // Capture and preprocess
    if (!camera_capture_and_preprocess(input_buf)) {
        return;
    }

    // Run inference
    float confidence = 0.0f;
    int predicted_class = inference_run(input_buf, &confidence);

    // Apply confidence threshold — if unsure, go straight
    if (confidence < CONFIDENCE_THRESHOLD) {
        predicted_class = CLASS_STRAIGHT;
    }

    uint16_t pwm = class_to_pwm(predicted_class);

    Serial.printf("[%s] conf=%.2f pwm=%d\n",
                   class_names[predicted_class], confidence, pwm);

    // Send steering command
    mavlink_send_steering(pwm);
}
