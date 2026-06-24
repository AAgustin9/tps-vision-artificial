"""
Train a MobileNetV2 binary classifier on PKLot parking space crops.

Transfer Learning rationale:
    With only 4,000 cropped space images, training from scratch risks severe overfitting.
    MobileNetV2 pretrained on ImageNet provides robust low-level feature detectors
    (edges, gradients, textures) that generalize well to parking space appearance.
    Car bodies, asphalt, and painted lines share visual statistics with ImageNet content.
    Reference: De Almeida et al., "PKLot — A robust dataset for parking lot
    classification", Expert Systems with Applications, 2015.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import EMPTY_DIR, OCCUPIED_DIR, MODEL_PATH, RESULTS_DIR, IMG_SIZE, ensure_dirs


def load_dataset(empty_dir: Path, occupied_dir: Path, img_size=IMG_SIZE):
    """Load and normalize all crops. Returns (X, y) numpy arrays."""
    from PIL import Image

    X, y = [], []

    for path in empty_dir.glob("*.jpg"):
        try:
            img = Image.open(path).convert("RGB").resize(img_size, Image.LANCZOS)
            X.append(np.array(img, dtype=np.float32) / 255.0)
            y.append(0)
        except Exception:
            continue

    for path in occupied_dir.glob("*.jpg"):
        try:
            img = Image.open(path).convert("RGB").resize(img_size, Image.LANCZOS)
            X.append(np.array(img, dtype=np.float32) / 255.0)
            y.append(1)
        except Exception:
            continue

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model(img_size=IMG_SIZE):
    """Build MobileNetV2-based binary classifier."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


def plot_training_curves(history, save_path: Path):
    """Save training accuracy and loss plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    ensure_dirs()

    print("Loading dataset...")
    X, y = load_dataset(EMPTY_DIR, OCCUPIED_DIR)
    print(f"  Total samples: {len(X)} (empty: {(y==0).sum()}, occupied: {(y==1).sum()})")

    if len(X) < 100:
        print("ERROR: Too few samples. Run prepare_dataset.py first.")
        return

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Build augmentation pipeline for training
    rotation_layer = tf.keras.layers.RandomRotation(factor=10/360)

    def augment(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.2)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = rotation_layer(tf.expand_dims(x, 0))[0]
        return x, y

    BATCH_SIZE = 32
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\nBuilding model...")
    model, base = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), save_best_only=True, monitor="val_accuracy"
        )
    ]

    print("\nPhase 1: Training head (base frozen)...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=30, callbacks=callbacks, verbose=1
    )

    # Fine-tune: unfreeze last 20 layers of base
    print("\nPhase 2: Fine-tuning last 20 layers...")
    base.trainable = True
    for layer in base.layers[-20:]:
        layer.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=10, callbacks=callbacks, verbose=1
    )

    # Merge histories
    combined = {}
    for k in history.history:
        combined[k] = history.history[k] + history2.history.get(k, [])
    history.history = combined

    plot_training_curves(history, RESULTS_DIR / "training_curves.png")

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(test_ds, verbose=0).flatten()
    y_pred_binary = (y_pred >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=["Empty", "Occupied"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_binary)
    print(cm)

    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Empty", "Occupied"],
                yticklabels=["Empty", "Occupied"], cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {RESULTS_DIR / 'confusion_matrix.png'}")

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
