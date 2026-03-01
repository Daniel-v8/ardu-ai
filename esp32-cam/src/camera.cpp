#include "camera.h"
#include "config.h"
#include <Arduino.h>

bool camera_init() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO;
    config.pin_d1       = Y3_GPIO;
    config.pin_d2       = Y4_GPIO;
    config.pin_d3       = Y5_GPIO;
    config.pin_d4       = Y6_GPIO;
    config.pin_d5       = Y7_GPIO;
    config.pin_d6       = Y8_GPIO;
    config.pin_d7       = Y9_GPIO;
    config.pin_xclk     = XCLK_GPIO;
    config.pin_pclk     = PCLK_GPIO;
    config.pin_vsync    = VSYNC_GPIO;
    config.pin_href     = HREF_GPIO;
    config.pin_sccb_sda = SIOD_GPIO;
    config.pin_sccb_scl = SIOC_GPIO;
    config.pin_pwdn     = PWDN_GPIO;
    config.pin_reset    = RESET_GPIO;
    config.xclk_freq_hz = CAMERA_XCLK_FREQ;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size   = FRAMESIZE_QVGA;  // 320x240
    config.fb_count     = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }

    Serial.println("Camera initialized (QVGA grayscale)");
    return true;
}

// Simple nearest-neighbor downscale with bottom crop
bool camera_capture_and_preprocess(int8_t* output_buf) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }

    // Source dimensions
    const int src_w = fb->width;   // 320
    const int src_h = fb->height;  // 240

    // Crop bottom portion (rover hardware visible)
    const int crop_h = (int)(src_h * (1.0f - CAMERA_CROP_BOTTOM_PCT));  // ~204 rows

    // Nearest-neighbor resize from (src_w x crop_h) → (MODEL_INPUT_W x MODEL_INPUT_H)
    for (int y = 0; y < MODEL_INPUT_H; y++) {
        int src_y = (y * crop_h) / MODEL_INPUT_H;
        for (int x = 0; x < MODEL_INPUT_W; x++) {
            int src_x = (x * src_w) / MODEL_INPUT_W;
            uint8_t pixel = fb->buf[src_y * src_w + src_x];
            // Convert uint8 [0,255] to int8 [-128,127] for quantized model
            output_buf[y * MODEL_INPUT_W + x] = (int8_t)(pixel - 128);
        }
    }

    esp_camera_fb_return(fb);
    return true;
}
