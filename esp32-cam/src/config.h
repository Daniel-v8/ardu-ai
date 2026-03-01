#pragma once

// --- Camera ---
// AI-Thinker ESP32-Cam pin mapping
#define PWDN_GPIO    32
#define RESET_GPIO   -1
#define XCLK_GPIO     0
#define SIOD_GPIO    26
#define SIOC_GPIO    27
#define Y9_GPIO      35
#define Y8_GPIO      34
#define Y7_GPIO      39
#define Y6_GPIO      36
#define Y5_GPIO      21
#define Y4_GPIO      19
#define Y3_GPIO      18
#define Y2_GPIO       5
#define VSYNC_GPIO   25
#define HREF_GPIO    23
#define PCLK_GPIO    22

#define CAMERA_XCLK_FREQ  20000000

// --- Model input ---
#define MODEL_INPUT_W   48
#define MODEL_INPUT_H   48
#define MODEL_INPUT_CH   1   // grayscale

// --- Inference ---
#define TENSOR_ARENA_SIZE  (100 * 1024)  // 100 KB in PSRAM
#define CONFIDENCE_THRESHOLD  0.55f       // below this → go straight

// --- Steering classes ---
#define CLASS_LEFT     0
#define CLASS_STRAIGHT 1
#define CLASS_RIGHT    2
#define NUM_CLASSES    3

// --- MAVLink UART ---
#define MAVLINK_SERIAL   Serial2
#define MAVLINK_TX_PIN   14
#define MAVLINK_RX_PIN   15
#define MAVLINK_BAUD     57600

// MAVLink IDs
#define MAV_SYS_ID       2   // companion computer
#define MAV_COMP_ID      1
#define MAV_TARGET_SYS   1   // autopilot
#define MAV_TARGET_COMP  1

// --- RC override values ---
#define RC_STEER_LEFT    1100
#define RC_STEER_CENTER  1500
#define RC_STEER_RIGHT   1900
#define RC_STEER_CHANNEL 1    // CH1 = steering

// --- Timing ---
#define HEARTBEAT_INTERVAL_MS  1000
#define INFERENCE_INTERVAL_MS   200   // 5 Hz
#define CAMERA_CROP_BOTTOM_PCT  0.15f // crop bottom 15% (rover hardware visible)
