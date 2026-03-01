# Ardurevo project

ESP32-Cam companion computer software for ArduPilot Rover.

## Plan

### ESP32-Cam in the car

- PlatformIO project for AI-Thinker ESP32-Cam
- Tensorflow Lite model
- Camera will take pictures of the road, and based on the pictures, the model will classify the orientations of the
  car on the road and return the steering angle to direct the car to the right or left.
- ArduPilot Rover will be running on a SpeedyBee controller, connected to the ESP32-Cam by UART port.

### PC software for traing the model

- Python software for training the model
- Source data will be pictures taken from the car and manually annotated with the steering angle.
