"""Train CNN model for ArduRevo steering classification.

Usage:
    python train.py
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config


def load_dataset():
    """Load images from class folders, crop bottom, resize to model input."""
    images = []
    labels = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = config.DATA_DIR / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue

            img = Image.open(img_path).convert('L')  # grayscale
            w, h = img.size

            # Crop bottom (rover hardware)
            crop_h = int(h * (1.0 - config.CROP_BOTTOM_PCT))
            img = img.crop((0, 0, w, crop_h))

            # Resize to model input
            img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.BILINEAR)

            # Normalize to [0, 1]
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_idx)

    images = np.array(images)[..., np.newaxis]  # (N, H, W, 1)
    labels = np.array(labels, dtype=np.int32)

    print(f"Loaded {len(images)} images")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name}: {np.sum(labels == i)}")

    return images, labels


def augment_image(image, label):
    """Apply data augmentation to a single image.

    Args:
        image: (H, W, 1) float32 array in [0, 1]
        label: int class index

    Returns:
        list of (image, label) tuples (original + augmented)
    """
    results = [(image, label)]

    # --- Horizontal flip with label swap ---
    flipped = np.fliplr(image)
    flipped_label = config.FLIP_LABEL_MAP[label]
    results.append((flipped, flipped_label))

    # --- Additional augmentations on both original and flipped ---
    for base_img, base_label in [(image, label), (flipped, flipped_label)]:
        for _ in range(3):  # 3 random augmentations per image
            aug = base_img.copy()

            # Random brightness
            brightness = 1.0 + np.random.uniform(-config.BRIGHTNESS_RANGE, config.BRIGHTNESS_RANGE)
            aug = np.clip(aug * brightness, 0, 1)

            # Random contrast
            mean = np.mean(aug)
            contrast = 1.0 + np.random.uniform(-config.CONTRAST_RANGE, config.CONTRAST_RANGE)
            aug = np.clip((aug - mean) * contrast + mean, 0, 1)

            # Random rotation
            angle = np.random.uniform(-config.ROTATION_RANGE, config.ROTATION_RANGE)
            pil_img = Image.fromarray((aug[:, :, 0] * 255).astype(np.uint8), mode='L')
            pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=128)
            aug = np.array(pil_img, dtype=np.float32) / 255.0
            aug = aug[..., np.newaxis]

            # Random crop
            if np.random.random() < 0.5:
                h, w = aug.shape[:2]
                crop_px = int(max(h, w) * config.RANDOM_CROP_PCT)
                top = np.random.randint(0, crop_px + 1)
                left = np.random.randint(0, crop_px + 1)
                bottom = h - np.random.randint(0, crop_px + 1)
                right = w - np.random.randint(0, crop_px + 1)
                cropped = aug[top:bottom, left:right, :]
                pil_img = Image.fromarray((cropped[:, :, 0] * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.BILINEAR)
                aug = np.array(pil_img, dtype=np.float32) / 255.0
                aug = aug[..., np.newaxis]

            # Gaussian noise
            noise = np.random.normal(0, config.GAUSSIAN_NOISE_STD, aug.shape).astype(np.float32)
            aug = np.clip(aug + noise, 0, 1)

            results.append((aug, base_label))

    return results


def augment_dataset(images, labels):
    """Apply augmentation to entire dataset."""
    aug_images = []
    aug_labels = []

    for img, lbl in zip(images, labels):
        for aug_img, aug_lbl in augment_image(img, lbl):
            aug_images.append(aug_img)
            aug_labels.append(aug_lbl)

    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels, dtype=np.int32)

    print(f"\nAfter augmentation: {len(aug_images)} images")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name}: {np.sum(aug_labels == i)}")

    return aug_images, aug_labels


def build_model():
    """Build small CNN for 48x48 grayscale classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)),

        tf.keras.layers.Conv2D(8, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(config.DROPOUT_1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(config.DROPOUT_2),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax'),
    ])

    return model


def main():
    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    images, labels = load_dataset()
    if len(images) == 0:
        print("No images found! Check DATA_DIR in config.py")
        return

    # Split BEFORE augmentation to prevent data leakage
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=labels
    )

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # Augment training set only
    X_train, y_train = augment_dataset(X_train, y_train)

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(config.NUM_CLASSES),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")

    # Build model
    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODELS_DIR / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(str(config.MODELS_DIR / "final_model.keras"))
    print(f"\nModel saved to {config.MODELS_DIR}")

    # Plot training history
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'], label='train')
        ax1.plot(history.history['val_accuracy'], label='val')
        ax1.set_title('Accuracy')
        ax1.legend()

        ax2.plot(history.history['loss'], label='train')
        ax2.plot(history.history['val_loss'], label='val')
        ax2.set_title('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(str(config.MODELS_DIR / "training_history.png"), dpi=100)
        print(f"Training plot saved to {config.MODELS_DIR / 'training_history.png'}")
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")

    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")


if __name__ == '__main__':
    main()
