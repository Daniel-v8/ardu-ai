#pragma once

#include <stdint.h>
#include <stdbool.h>

// Initialize TFLite Micro interpreter with the model
bool inference_init();

// Run inference on preprocessed int8 image buffer.
// Returns predicted class (0=left, 1=straight, 2=right).
// confidence is set to the softmax probability of the predicted class.
int inference_run(const int8_t* input_buf, float* confidence);
