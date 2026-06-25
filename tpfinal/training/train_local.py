"""
Entrenamiento local de PKLot con GPU.
Equivalente al notebook PKLot_training.ipynb pero pensado para correr
en una maquina con GPU dedicada (NVIDIA + CUDA o Apple Silicon).

Uso:
    python train_local.py --zip PKLot.zip --output ./output
    python train_local.py --dataset ./PKLot_data --output ./output  # si ya descomprimiste
"""

import argparse
import json
import os
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Argumentos ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Entrena clasificador PKLot localmente con GPU.")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--zip",     type=str, help="Ruta al archivo PKLot.zip")
src.add_argument("--dataset", type=str, help="Ruta a la carpeta ya descomprimida")
parser.add_argument("--output",    type=str, default="./output", help="Carpeta de salida del modelo")
parser.add_argument("--epochs",    type=int, default=20)
parser.add_argument("--batch",     type=int, default=32)
parser.add_argument("--lr",        type=float, default=1e-4)
parser.add_argument("--img-size",  type=int, default=96, help="Lado de la imagen cuadrada (px)")
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--no-mixed",  action="store_true", help="Desactivar mixed precision (float16)")
args = parser.parse_args()

IMAGE_SIZE    = (args.img_size, args.img_size)
BATCH_SIZE    = args.batch
EPOCHS        = args.epochs
LEARNING_RATE = args.lr
RANDOM_SEED   = args.seed
OUTPUT_PATH   = Path(args.output)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ── TensorFlow (importar despues de parsear args para no demorar --help) ──────

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print(f"TensorFlow {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs disponibles: {gpus if gpus else 'ninguna (CPU mode)'}")

if gpus and not args.no_mixed:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision activada (float16).")

# ── Dataset ───────────────────────────────────────────────────────────────────

if args.zip:
    dataset_path = Path(args.zip).parent / "PKLot_data"
    if not dataset_path.is_dir():
        print(f"Descomprimiendo {args.zip} en {dataset_path} ...")
        with zipfile.ZipFile(args.zip, "r") as zf:
            zf.extractall(dataset_path)
        print("Listo.")
    else:
        print(f"Usando dataset ya descomprimido en {dataset_path}")
else:
    dataset_path = Path(args.dataset)
    assert dataset_path.is_dir(), f"No existe la carpeta: {dataset_path}"

# ── Recolectar pares (path, bbox, label, split) ───────────────────────────────

def collect_pairs(dataset_path):
    dataset_path = Path(dataset_path)
    empty_kw    = {"space-empty", "empty", "free"}
    occupied_kw = {"space-occupied", "occupied", "busy"}
    img_exts    = {".jpg", ".jpeg", ".png"}
    pairs = []

    for split in ["train", "valid", "test"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        coco_path = split_path / "_annotations.coco.json"

        if coco_path.exists():
            with open(coco_path) as f:
                coco = json.load(f)
            cat_label = {}
            for cat in coco["categories"]:
                name = cat["name"].lower()
                if any(kw in name for kw in empty_kw):
                    cat_label[cat["id"]] = 0
                elif any(kw in name for kw in occupied_kw):
                    cat_label[cat["id"]] = 1
            id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
            before = len(pairs)
            for ann in coco["annotations"]:
                label = cat_label.get(ann["category_id"])
                if label is None:
                    continue
                fname = id_to_file.get(ann["image_id"])
                if not fname:
                    continue
                pairs.append((str(split_path / fname), ann["bbox"], label, split))
            print(f"  {split}: {len(pairs) - before} patches (COCO)")
        else:
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            for label_name, label in [("Empty",0),("empty",0),("Occupied",1),("occupied",1)]:
                for img_path in split_path.rglob(f"{label_name}/*"):
                    if img_path.is_file() and img_path.suffix.lower() in img_exts:
                        parent = img_path.parent.parent.name
                        date_key = parent if date_pattern.match(parent) else split
                        pairs.append((str(img_path), None, label, date_key))

    return pairs


print("\nCargando anotaciones...")
pairs = collect_pairs(dataset_path)
uses_coco = len(pairs) > 0 and pairs[0][1] is not None
print(f"Total patches: {len(pairs)} | COCO mode: {uses_coco}")
assert len(pairs) > 0, "No se encontraron pares. Revisa --dataset o --zip."

# ── Split ─────────────────────────────────────────────────────────────────────

has_roboflow = uses_coco or all(s in ("train","valid","test") for _,_,_,s in pairs)

if has_roboflow:
    train_pairs = [(p,b,l) for p,b,l,s in pairs if s == "train"]
    val_pairs   = [(p,b,l) for p,b,l,s in pairs if s == "valid"]
    test_pairs  = [(p,b,l) for p,b,l,s in pairs if s == "test"]
    print(f"Splits Roboflow: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
else:
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    has_dates = all(date_pattern.match(s) for _,_,_,s in pairs)
    random.seed(RANDOM_SEED)

    if has_dates:
        dates = sorted(set(s for _,_,_,s in pairs))
        random.shuffle(dates)
        n = len(dates)
        train_d = set(dates[:int(0.7*n)])
        val_d   = set(dates[int(0.7*n):int(0.85*n)])
        test_d  = set(dates[int(0.85*n):])
        train_pairs = [(p,b,l) for p,b,l,s in pairs if s in train_d]
        val_pairs   = [(p,b,l) for p,b,l,s in pairs if s in val_d]
        test_pairs  = [(p,b,l) for p,b,l,s in pairs if s in test_d]
    else:
        all_p = [p for p,b,l,s in pairs]
        all_b = [b for p,b,l,s in pairs]
        all_l = [l for p,b,l,s in pairs]
        idxs  = list(range(len(all_p)))
        tr_i, rest_i = train_test_split(idxs, test_size=0.3, stratify=all_l, random_state=RANDOM_SEED)
        rest_l = [all_l[i] for i in rest_i]
        va_i, te_i = train_test_split(rest_i, test_size=0.5, stratify=rest_l, random_state=RANDOM_SEED)
        train_pairs = [(all_p[i], all_b[i], all_l[i]) for i in tr_i]
        val_pairs   = [(all_p[i], all_b[i], all_l[i]) for i in va_i]
        test_pairs  = [(all_p[i], all_b[i], all_l[i]) for i in te_i]

def label_counts(sp):
    c = defaultdict(int)
    for _,_,l in sp: c[l] += 1
    return dict(c)

print(f"Train {len(train_pairs)} {label_counts(train_pairs)}")
print(f"Val   {len(val_pairs)}   {label_counts(val_pairs)}")
print(f"Test  {len(test_pairs)}  {label_counts(test_pairs)}")

# ── tf.data pipeline ──────────────────────────────────────────────────────────

_rotation_layer = tf.keras.layers.RandomRotation(0.04)


def load_and_crop(path, bbox, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    if uses_coco:
        img_h = tf.cast(tf.shape(image)[0], tf.float32)
        img_w = tf.cast(tf.shape(image)[1], tf.float32)
        x = bbox[0] / img_w
        y = bbox[1] / img_h
        w = bbox[2] / img_w
        h = bbox[3] / img_h
        box = tf.reshape(tf.stack([
            tf.clip_by_value(y,     0.0, 1.0),
            tf.clip_by_value(x,     0.0, 1.0),
            tf.clip_by_value(y + h, 0.0, 1.0),
            tf.clip_by_value(x + w, 0.0, 1.0),
        ]), [1, 4])
        image = tf.image.crop_and_resize(
            tf.expand_dims(image, 0), box, [0], IMAGE_SIZE
        )[0] / 255.0
    else:
        image = tf.image.resize(image, IMAGE_SIZE) / 255.0

    return image, tf.cast(label, tf.float32)


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = _rotation_layer(tf.expand_dims(image, 0))[0]
    return image, label


def make_dataset(split_pairs, training, workers=None):
    paths  = [p for p,b,l in split_pairs]
    bboxes = [[float(v) for v in b] if b is not None else [0.,0.,0.,0.] for p,b,l in split_pairs]
    labels = [l for p,b,l in split_pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, bboxes, labels))
    ds = ds.map(load_and_crop, num_parallel_calls=workers or tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(len(split_pairs), 5000), seed=RANDOM_SEED)
        ds = ds.map(augment, num_parallel_calls=workers or tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


print("\nConstruyendo datasets...")
train_ds = make_dataset(train_pairs, training=True)
val_ds   = make_dataset(val_pairs,   training=False)
test_ds  = make_dataset(test_pairs,  training=False)

# ── Modelo ────────────────────────────────────────────────────────────────────

print("\nConstruyendo modelo...")
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

use_fp16 = gpus and not args.no_mixed
output_dtype = "float32"  # siempre float32 en la salida para estabilidad

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid", dtype=output_dtype),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# ── Entrenamiento ─────────────────────────────────────────────────────────────

checkpoint_path = str(OUTPUT_PATH / "best_checkpoint.keras")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    tf.keras.callbacks.TensorBoard(log_dir=str(OUTPUT_PATH / "logs")),
]

print(f"\nEntrenando por hasta {EPOCHS} epochs (batch={BATCH_SIZE}, img={IMAGE_SIZE})...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── Evaluacion ────────────────────────────────────────────────────────────────

print("\nEvaluando en test set...")
y_true, y_pred = [], []
for images, batch_labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()
    y_pred.extend((probs >= 0.5).astype(int).tolist())
    y_true.extend(batch_labels.numpy().astype(int).tolist())

print(classification_report(y_true, y_pred, target_names=["Empty", "Occupied"]))

cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ConfusionMatrixDisplay(cm, display_labels=["Empty", "Occupied"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix - Test")

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(history.history["accuracy"], label="train")
axes[2].plot(history.history["val_accuracy"], label="val")
axes[2].set_title("Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
plot_path = OUTPUT_PATH / "resultados.png"
plt.savefig(plot_path, dpi=150)
print(f"Graficos guardados en {plot_path}")

# ── Exportar modelo ───────────────────────────────────────────────────────────

final_path = OUTPUT_PATH / "modelo_pklot.keras"
model.save(str(final_path))
print(f"\nModelo guardado en: {final_path}")
print("Usalo con --model en inference/detect.py")
