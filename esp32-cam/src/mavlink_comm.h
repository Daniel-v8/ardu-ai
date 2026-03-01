#pragma once

#include <stdint.h>
#include <stdbool.h>

// Initialize MAVLink UART
void mavlink_init();

// Call in main loop — handles heartbeat sending and message parsing
void mavlink_update();

// Send RC_CHANNELS_OVERRIDE with steering value
// pwm_value: 1100 (left), 1500 (center), 1900 (right)
void mavlink_send_steering(uint16_t pwm_value);
