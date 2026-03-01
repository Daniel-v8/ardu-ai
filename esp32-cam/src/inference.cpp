#include "inference.h"
#include "config.h"
#include "model_data.h"

#include <Arduino.h>
#include <cmath>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t* tensor_arena = nullptr;
static TfLiteTensor* input_tensor = nullptr;

bool inference_init() {
    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        Serial.println("Failed to allocate tensor arena in PSRAM");
        return false;
    }

    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema mismatch: %lu vs %d\n",
                       model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return false;
    }

    input_tensor = interpreter->input(0);

    Serial.printf("TFLite model loaded. Input: [%d, %d, %d, %d]\n",
                   input_tensor->dims->data[0],
                   input_tensor->dims->data[1],
                   input_tensor->dims->data[2],
                   input_tensor->dims->data[3]);

    return true;
}

int inference_run(const int8_t* input_buf, float* confidence) {
    // Copy input data to tensor
    int input_size = MODEL_INPUT_W * MODEL_INPUT_H * MODEL_INPUT_CH;
    memcpy(input_tensor->data.int8, input_buf, input_size);

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke() failed");
        *confidence = 0.0f;
        return CLASS_STRAIGHT;
    }

    // Read output — int8 quantized
    TfLiteTensor* output = interpreter->output(0);
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;

    // Dequantize and find argmax
    float max_val = -1e9f;
    int max_idx = CLASS_STRAIGHT;
    float sum_exp = 0.0f;
    float dequant[NUM_CLASSES];

    for (int i = 0; i < NUM_CLASSES; i++) {
        dequant[i] = (output->data.int8[i] - zero_point) * scale;
    }

    // Softmax (output may already be softmax, but ensure normalization)
    for (int i = 0; i < NUM_CLASSES; i++) {
        dequant[i] = expf(dequant[i]);
        sum_exp += dequant[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        dequant[i] /= sum_exp;
        if (dequant[i] > max_val) {
            max_val = dequant[i];
            max_idx = i;
        }
    }

    *confidence = max_val;
    return max_idx;
}
