Plán: Základ projektu ArduRevo                                                                                                                                            
                                                                                                                                                                           
 Kontext                                                                                                                                                                   
                                                                                                                                                                           
 ESP32-Cam companion computer pre ArduPilot Rover. Kamera sníma cestu, TFLite model klasifikuje smer (3 triedy: doľava/rovno/doprava), ESP32 posiela MAVLink príkazy na    
 SpeedyBee FC cez UART. Trénovanie modelu na PC z ~53 anotovaných fotiek.

 Štruktúra projektu

 /home/daniel/Desktop/Ardu AI/                                                                                                                                             
 ├── Plan.md                              (existuje)
 ├── ArduRevo fotky usporiadane/          (existuje - trénovacie fotky)
 ├── esp32-cam/                           (NOVÝ - PlatformIO projekt)
 │   ├── platformio.ini
 │   ├── include/
 │   │   └── model_data.h                (placeholder TFLite model ako C pole)
 │   ├── src/
 │   │   ├── main.cpp                    (hlavná slučka: capture → infer → steer)
 │   │   ├── config.h                    (piny, konštanty)
 │   │   ├── camera.h / camera.cpp       (OV2640 init a snímanie)
 │   │   ├── inference.h / inference.cpp (TFLite Micro inference)
 │   │   └── mavlink_comm.h / mavlink_comm.cpp (MAVLink UART komunikácia)
 │   ├── lib/
 │   │   └── README
 │   └── test/
 │       └── README
 └── training/                            (NOVÝ - Python tréning)
     ├── requirements.txt
     ├── config.py                        (cesty, konštanty)
     ├── train.py                         (tréning CNN modelu)
     ├── export_model.py                  (konverzia na TFLite int8 + C header)
     ├── test_inference.py                (overenie modelu)
     ├── models/                          (uložené modely)
     └── output/                          (exportované TFLite + C headers)

 ESP32-Cam firmware

 Kľúčové rozhodnutia

 - Board: AI-Thinker ESP32-Cam, huge_app.csv partition pre dostatok flash
 - Model vstup: 48x48 grayscale (z 320x240 QVGA) — malý model (~16-20KB kvantizovaný)
 - Kamera: PIXFORMAT_GRAYSCALE priamo — bez JPEG encode/decode réžie
 - UART: UART2 na GPIO 14 (TX) / GPIO 15 (RX) pre MAVLink
 - MAVLink baud: 57600, system_id=2 (companion computer)
 - Riadenie: RC_CHANNELS_OVERRIDE — CH1=steering (1100=left, 1500=center, 1900=right)
 - TFLite arena: 100KB v PSRAM
 - Bezpečnosť: confidence threshold — ak model nie je istý, ide rovno
 - TFLite Micro lib: tanakamasayuki/TensorFlowLite_ESP32
 - MAVLink lib: mavlink/c_library_v2 header-only v lib/mavlink/

 Hlavná slučka (main.cpp)

 setup: camera_init() → inference_init() → mavlink_init()
 loop:  mavlink_update() → capture → preprocess → inference → send steering

 Python tréning

 Model — vlastný malý CNN (~16K parametrov, ~20KB int8 TFLite)

 Input 48x48x1 → Conv2D(8) → BN+ReLU → MaxPool
 → Conv2D(16) → BN+ReLU → MaxPool
 → Conv2D(32) → BN+ReLU → MaxPool
 → Conv2D(32) → BN+ReLU → GlobalAvgPool
 → Dropout(0.5) → Dense(16) → Dropout(0.3) → Dense(3, softmax)

 Data augmentation (kritické pri 53 fotkách)

 - Horizontálny flip so zámenou labelu (doľava↔doprava) — najdôležitejšie, zdvojnásobí left/right
 - Jas ±30%, kontrast ±20%, rotácia ±10°, random crop 10%, gaussian noise
 - Orezanie spodných ~15% obrázku (vidno hardvér roveru)
 - Class weights pre nevyváženosť (31 rovno vs 11 doľava/doprava)

 Export

 - Int8 kvantizácia (váhy aj aktivácie) s representative dataset
 - Export ako C header (model_data.h) s alignas(16) pre ESP32

 Požiadavky

 - Python 3.11/3.12 venv (TensorFlow nepodporuje 3.14)
 - tensorflow, numpy, Pillow, matplotlib, scikit-learn

 Čo vytvorím teraz (základ)

 1. esp32-cam/platformio.ini — konfigurácia buildu
 2. esp32-cam/src/config.h — piny a konštanty
 3. esp32-cam/src/camera.h + camera.cpp — snímanie a preprocessing
 4. esp32-cam/src/inference.h + inference.cpp — TFLite Micro inference
 5. esp32-cam/src/mavlink_comm.h + mavlink_comm.cpp — MAVLink komunikácia
 6. esp32-cam/src/main.cpp — hlavná slučka
 7. esp32-cam/include/model_data.h — placeholder model
 8. training/requirements.txt — Python závislosti
 9. training/config.py — spoločná konfigurácia
 10. training/train.py — trénovací skript s augmentáciou
 11. training/export_model.py — TFLite konverzia + C header export
 12. training/test_inference.py — overenie modelu

 Overenie

 - ESP32 firmware: pio run musí skompilovať (s placeholder modelom)
 - Python tréning: spustenie train.py natrénuje model z existujúcich fotiek
 - export_model.py vygeneruje model_data.h pre ESP32
 - test_inference.py ukáže accuracy a confusion matrix
