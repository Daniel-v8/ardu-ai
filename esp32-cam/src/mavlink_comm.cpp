#include "mavlink_comm.h"
#include "config.h"
#include <Arduino.h>

// MAVLink headers — using c_library_v2
#include "common/mavlink.h"

static uint32_t last_heartbeat_ms = 0;

void mavlink_init() {
    MAVLINK_SERIAL.begin(MAVLINK_BAUD, SERIAL_8N1, MAVLINK_RX_PIN, MAVLINK_TX_PIN);
    Serial.printf("MAVLink UART initialized (TX=%d, RX=%d, baud=%d)\n",
                   MAVLINK_TX_PIN, MAVLINK_RX_PIN, MAVLINK_BAUD);
}

void mavlink_update() {
    // Send heartbeat periodically
    uint32_t now = millis();
    if (now - last_heartbeat_ms >= HEARTBEAT_INTERVAL_MS) {
        last_heartbeat_ms = now;

        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_heartbeat_pack(MAV_SYS_ID, MAV_COMP_ID, &msg,
                                    MAV_TYPE_ONBOARD_CONTROLLER,
                                    MAV_AUTOPILOT_INVALID,
                                    0, 0, 0);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        MAVLINK_SERIAL.write(buf, len);
    }

    // Parse incoming messages (for future use — e.g., mode checking)
    while (MAVLINK_SERIAL.available()) {
        uint8_t c = MAVLINK_SERIAL.read();
        mavlink_message_t msg;
        mavlink_status_t status;
        if (mavlink_parse_char(MAVLINK_COMM_0, c, &msg, &status)) {
            // Handle incoming messages if needed
        }
    }
}

void mavlink_send_steering(uint16_t pwm_value) {
    mavlink_message_t msg;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];

    // RC_CHANNELS_OVERRIDE: set CH1 (steering), leave others unchanged (0 = no override)
    // MAVLink v2 rc_channels_override has 18 channels
    mavlink_msg_rc_channels_override_pack(
        MAV_SYS_ID, MAV_COMP_ID, &msg,
        MAV_TARGET_SYS, MAV_TARGET_COMP,
        pwm_value,                        // CH1 — steering
        0, 0, 0, 0, 0, 0, 0,             // CH2-CH8 — no override
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0     // CH9-CH18 — no override
    );

    uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
    MAVLINK_SERIAL.write(buf, len);
}
