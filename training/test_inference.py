"""Test trained model inference and show metrics.

Usage:
    python test_inference.py [--tflite]

Options:
    --tflite    Test the TFLite model instead of Keras model
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config


def load_all_images():
    """Load all images with labels."""
    images = []
    labels = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = config.DATA_DIR / class_name
        if not class_dir.exists():
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue

            img = Image.open(img_path).convert('L')
            w, h = img.size

            crop_h = int(h * (1.0 - config.CROP_BOTTOM_PCT))
            img = img.crop((0, 0, w, crop_h))
            img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.BILINEAR)

            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_idx)

    images = np.array(images)[..., np.newaxis]
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def test_keras_model(images, labels):
    """Test Keras model."""
    model_path = config.MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        print(f"Keras model not found: {model_path}")
        return

    print(f"Loading Keras model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    print("\n=== Keras Model Results ===")
    print(classification_report(labels, pred_classes, target_names=config.CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, pred_classes))

    # Show per-image predictions
    print("\nPer-image predictions:")
    for i, (pred, true) in enumerate(zip(pred_classes, labels)):
        conf = predictions[i][pred]
        marker = "OK" if pred == true else "WRONG"
        print(f"  [{marker}] #{i}: true={config.CLASS_NAMES[true]}, "
              f"pred={config.CLASS_NAMES[pred]} (conf={conf:.3f})")


def test_tflite_model(images, labels):
    """Test TFLite quantized model."""
    tflite_path = config.TFLITE_MODEL_PATH
    if not tflite_path.exists():
        print(f"TFLite model not found: {tflite_path}")
        print("Run export_model.py first!")
        return

    print(f"Loading TFLite model from {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]['quantization'][0]
    input_zp = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zp = output_details[0]['quantization'][1]

    pred_classes = []
    confidences = []

    for img in images:
        # Quantize input
        quantized = np.round(img / input_scale + input_zp).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized[np.newaxis, ...])
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        dequant = (output.astype(np.float32) - output_zp) * output_scale

        # Softmax
        exp_vals = np.exp(dequant - np.max(dequant))
        probs = exp_vals / np.sum(exp_vals)

        pred = np.argmax(probs)
        pred_classes.append(pred)
        confidences.append(probs[0][pred])

    pred_classes = np.array(pred_classes)

    print("\n=== TFLite int8 Model Results ===")
    print(classification_report(labels, pred_classes, target_names=config.CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, pred_classes))

    # Show per-image predictions
    print("\nPer-image predictions:")
    for i, (pred, true, conf) in enumerate(zip(pred_classes, labels, confidences)):
        marker = "OK" if pred == true else "WRONG"
        print(f"  [{marker}] #{i}: true={config.CLASS_NAMES[true]}, "
              f"pred={config.CLASS_NAMES[pred]} (conf={conf:.3f})")


def main():
    images, labels = load_all_images()
    print(f"Loaded {len(images)} images")

    if '--tflite' in sys.argv:
        test_tflite_model(images, labels)
    else:
        test_keras_model(images, labels)


if __name__ == '__main__':
    main()
