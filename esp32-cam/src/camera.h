#pragma once

#include <esp_camera.h>
#include <stdint.h>
#include <stdbool.h>

// Initialize OV2640 in QVGA grayscale mode
bool camera_init();

// Capture a frame, resize/crop to model input, normalize to [0,1] float or int8.
// Output buffer must be MODEL_INPUT_W * MODEL_INPUT_H bytes.
// Returns true on success.
bool camera_capture_and_preprocess(int8_t* output_buf);
