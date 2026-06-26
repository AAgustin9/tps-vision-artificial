# Detección de Espacios de Estacionamiento con Deep Learning

## Índice
1. [Resumen del proyecto](#1-resumen-del-proyecto)
2. [Dataset: PKLot](#2-dataset-pklot)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Transfer Learning y MobileNetV2](#4-transfer-learning-y-mobilenetv2)
5. [Pipeline de entrenamiento](#5-pipeline-de-entrenamiento)
6. [Módulo de inferencia](#6-módulo-de-inferencia)
7. [Calibración de espacios](#7-calibración-de-espacios)
8. [Resultados y métricas](#8-resultados-y-métricas)
9. [Cómo correr el proyecto](#9-cómo-correr-el-proyecto)

---

## 1. Resumen del proyecto

El objetivo es determinar automáticamente si los espacios de un estacionamiento están **libres u ocupados** a partir de imágenes de cámara fija, usando visión artificial y aprendizaje profundo.

El sistema tiene dos etapas:

```
Imagen del estacionamiento
         │
         ▼
  [Calibración] ──► spots.json  (coordenadas de cada espacio, se hace una sola vez)
         │
         ▼
  [Inferencia]  ──► imagen anotada + "7/14 espacios libres"
```

**Tecnologías usadas:**
- Python 3.11
- TensorFlow / Keras — entrenamiento e inferencia del modelo
- OpenCV — procesamiento de imágenes y visualización
- MobileNetV2 — arquitectura base (transfer learning)
- Dataset PKLot — 12.416 imágenes de estacionamientos con anotaciones COCO

---

## 2. Dataset: PKLot

**PKLot** (Parking Lot dataset) es un dataset público de la Universidad Federal de Paraná (Brasil), licenciado bajo CC BY 4.0.

### Características
- **12.416 imágenes** de estacionamientos tomadas con cámaras fijas
- Imágenes de días soleados, nublados y lluviosos
- Cada imagen contiene el estacionamiento completo (vista aérea)
- Las anotaciones están en formato **COCO** con bounding boxes de cada espacio

### Clases
| Categoría | Label | Descripción |
|-----------|-------|-------------|
| `space-empty` | 0 | Espacio libre |
| `space-occupied` | 1 | Espacio ocupado |

### Estructura del zip
```
PKLot.zip
├── train/
│   ├── _annotations.coco.json   ← coordenadas y clase de cada espacio
│   └── *.jpg                    ← imágenes completas del estacionamiento
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

### Por qué no hay carpetas Empty/Occupied
El zip de Roboflow exporta en formato de **detección de objetos** (bounding boxes), no clasificación. Las imágenes son vistas completas del estacionamiento — el sistema recorta cada espacio individualmente usando las coordenadas del JSON antes de clasificar.

### Subsampling
Para reducir el tiempo de entrenamiento de ~2 horas a ~30 minutos, se usan solo **2000 imágenes fuente** (`MAX_IMAGES = 2000`) de las 12.416 disponibles, manteniendo la proporción train/val/test.

---

## 3. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRENAMIENTO (Colab)                    │
│                                                                 │
│  PKLot.zip ──► Extraer patches por bbox ──► Aumentación        │
│                      │                                          │
│                       ▼                                         │
│              MobileNetV2 (congelado)                            │
│                  + cabeza densa                                 │
│                      │                                          │
│                       ▼                                         │
│              modelo_pklot.h5  ──► Google Drive                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │  descarga
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCIA (local)                       │
│                                                                 │
│  foto.jpg + spots.json                                          │
│       │                                                         │
│       ▼                                                         │
│  crop_spot()  →  warpPerspective  →  recorte 96×96             │
│       │                                                         │
│       ▼                                                         │
│  modelo_pklot.h5  →  probabilidad  →  libre/ocupado            │
│       │                                                         │
│       ▼                                                         │
│  imagen anotada (verde=libre, rojo=ocupado)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Transfer Learning y MobileNetV2

### ¿Qué es Transfer Learning?
En lugar de entrenar una red neuronal desde cero (lo que requeriría millones de imágenes y días de cómputo), se toma una red **preentrenada en ImageNet** (1.2 millones de imágenes, 1000 clases) y se reusan sus pesos como punto de partida.

### ¿Por qué MobileNetV2?
MobileNetV2 es una arquitectura diseñada para ser **liviana y eficiente**, ideal para dispositivos con recursos limitados. Usa:
- **Depthwise separable convolutions** — mucho más baratas que convoluciones estándar
- **Bottleneck residual blocks** — permiten redes profundas con pocos parámetros
- ~3.4 millones de parámetros (vs ~25M de ResNet50)

### Cómo se adapta al problema
```python
base_model = MobileNetV2(input_shape=(96, 96, 3),
                          include_top=False,      # sin la cabeza de clasificación
                          weights="imagenet")     # pesos preentrenados
base_model.trainable = False                      # congelar: no modificar pesos base

model = Sequential([
    base_model,                        # extractor de características
    GlobalAveragePooling2D(),          # reducir mapa de características a vector
    Dense(128, activation="relu"),     # capa densa intermedia
    Dropout(0.3),                      # regularización
    Dense(1, activation="sigmoid"),    # salida: probabilidad de ocupado (0 a 1)
])
```

La capa `sigmoid` en la salida devuelve un valor entre 0 y 1:
- Cercano a **0** → espacio **libre**
- Cercano a **1** → espacio **ocupado**
- Umbral de decisión: **0.5**

---

## 5. Pipeline de entrenamiento

### Celda 3 — Configuración
```python
IMAGE_SIZE  = (96, 96)   # tamaño de cada patch recortado
BATCH_SIZE  = 16         # imágenes por paso de entrenamiento
EPOCHS      = 20         # épocas máximas
MAX_IMAGES  = 2000       # imágenes fuente a usar (None = todas)
```

### Celda 5 — Recolección de pares
Lee `_annotations.coco.json` de cada split y construye una lista de:
```
(ruta_imagen, [x, y, w, h], label, split)
```
Donde `[x, y, w, h]` es el bounding box del espacio en la imagen original.

Con `MAX_IMAGES=2000` se samplea proporcionalmente en train/valid/test.

### Celda 6 — Split
Usa los splits pre-existentes de Roboflow (train/valid/test), sin re-dividir.

### Celda 7 — Pipeline tf.data
Para cada anotación:
1. Cargar imagen completa del estacionamiento
2. Recortar el bbox con `crop_and_resize` → patch 96×96
3. Normalizar píxeles a [0, 1]
4. En train: aplicar aumentación (flip, brillo, rotación leve)

```
imagen 640×640 ──► crop bbox ──► resize 96×96 ──► normalizar ──► batch
```

**Por qué 96×96 y no 224×224:**
Los bboxes de PKLot miden ~23×40 píxeles en el original. Escalar a 224×224 no agrega información útil y usa 5× más memoria.

### Celda 8 — Modelo con Mixed Precision
```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```
Computa en **float16** (ocupa la mitad de VRAM) y acumula gradientes en float32. Reduce el uso de memoria ~2× sin impacto en accuracy.

### Celda 9 — Entrenamiento con callbacks
| Callback | Función |
|----------|---------|
| `EarlyStopping` | Detiene si val_loss no mejora en 5 épocas |
| `ModelCheckpoint` | Guarda el mejor modelo por val_loss |
| `ReduceLROnPlateau` | Baja el learning rate a la mitad si val_loss se estanca 2 épocas |

### Celda 10 — Evaluación
Calcula sobre el test set (imágenes nunca vistas):
- **Accuracy** — porcentaje de espacios clasificados correctamente
- **Precision** — de los que dijo "ocupado", ¿cuántos realmente lo estaban?
- **Recall** — de los que estaban ocupados, ¿cuántos detectó?
- **F1** — promedio armónico de precision y recall
- **Matriz de confusión** — distribución de errores por clase

---

## 6. Módulo de inferencia

### `detect.py` — flujo completo

```
foto.jpg
   │
   ▼
cv2.imread()                    # leer imagen en BGR
   │
   ▼
load_spots("spots.json")        # cargar coordenadas de los espacios
   │
   ▼
Para cada espacio:
   │
   ├─► order_points()           # ordenar 4 puntos: TL, TR, BR, BL
   │
   ├─► crop_spot()              # warpPerspective → recorte recto 96×96
   │        └─ corrige la perspectiva aunque el espacio esté inclinado
   │
   ├─► preprocess_crop()        # BGR→RGB, dividir por 255, agregar dim batch
   │
   └─► model.predict()          # probabilidad → libre/ocupado (umbral 0.5)
   │
   ▼
draw_results()                  # dibujar polígonos y texto sobre la imagen
   │
   ▼
imagen_resultado.jpg + consola: "7/14 espacios libres"
```

### Funciones clave

**`order_points(points)`**
Ordena los 4 puntos marcados por el usuario en el orden correcto (top-left, top-right, bottom-right, bottom-left) para que `warpPerspective` funcione independientemente del orden en que fueron clickeados.

**`crop_spot(image, points, output_size)`**
Aplica una transformación de perspectiva para "enderezar" el espacio aunque aparezca inclinado en la imagen. Usa `cv2.getPerspectiveTransform` + `cv2.warpPerspective`.

**`preprocess_crop(crop)`**
Convierte el crop de BGR (formato OpenCV) a RGB (formato que espera el modelo), normaliza a [0, 1] y agrega la dimensión de batch: `(1, 96, 96, 3)`.

**`classify_spot(model, crop, threshold=0.5)`**
Corre el modelo y aplica el umbral de decisión. Devuelve `(is_occupied, probability)`.

---

## 7. Calibración de espacios

### ¿Por qué es necesaria?
El modelo clasifica **patches individuales** de cada espacio. Para saber dónde está cada espacio en la imagen hay que marcarlo manualmente una sola vez por layout de cámara.

### `calibrate.py` — cómo funciona
```
python calibrate.py --image foto.jpg --output spots.json
```
1. Se abre una ventana con la imagen
2. Se clickean los **4 vértices** de cada espacio en orden
3. Al 4to click se cierra el espacio y se pasa al siguiente
4. Al terminar todos los espacios: presionar `s` para guardar

**Teclas:**
| Tecla | Acción |
|-------|--------|
| Click izquierdo | Marcar vértice |
| `s` | Guardar y salir |
| `u` | Deshacer último punto o último espacio |
| `q` / `ESC` | Salir sin guardar |

### Formato de spots.json
```json
{
  "spots": [
    {
      "id": "spot_1",
      "points": [[120, 80], [200, 80], [200, 160], [120, 160]]
    },
    {
      "id": "spot_2",
      "points": [[210, 80], [290, 80], [290, 160], [210, 160]]
    }
  ]
}
```
Cada espacio tiene un `id` y 4 puntos `[x, y]` que definen su cuadrilátero en la imagen.

---

## 8. Resultados y métricas

### Métricas reportadas por el modelo
Al finalizar el entrenamiento (Celda 10), se imprime un `classification_report` con:

```
              precision    recall  f1-score   support

       Empty       0.xx      0.xx      0.xx      xxxx
    Occupied       0.xx      0.xx      0.xx      xxxx

    accuracy                           0.xx      xxxx
```

### Interpretación de la matriz de confusión

```
                 Predicho
                 Empty  Occupied
Real  Empty    [  TP  |   FP  ]
      Occupied [  FN  |   TN  ]
```

- **FP (falso positivo):** dice que hay lugar pero está ocupado → el sistema manda al auto a un lugar que no existe
- **FN (falso negativo):** dice que está ocupado pero hay lugar → pierde lugares disponibles

Para este problema, los FP son más costosos que los FN.

---

## 9. Cómo correr el proyecto

### Entrenamiento (Google Colab)
1. Subir `PKLot.zip` a Google Drive en `MyDrive/PKLot.zip`
2. Abrir `training/PKLot_training.ipynb` en Colab
3. Correr todas las celdas en orden
4. El modelo queda en `MyDrive/PKLot_output/modelo_pklot.h5`

### Inferencia (local)
```bash
# 1. Crear entorno virtual con Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install opencv-python numpy tensorflow

# 3. Calibrar espacios (una sola vez por imagen/cámara)
python inference/calibrate.py --image foto.jpg --output spots.json

# 4. Detectar
python inference/detect.py \
    --model modelo_pklot.h5 \
    --image foto.jpg \
    --spots spots.json
```

### Estructura del proyecto
```
tpfinal/
├── training/
│   ├── PKLot_training.ipynb     # notebook de Colab
│   ├── train_local.py           # script para entrenar con GPU local
│   ├── LOCAL_TRAINING.md        # guía de entrenamiento local
│   └── requirements-local.txt
├── inference/
│   ├── detect.py                # clasificación e inferencia
│   ├── calibrate.py             # calibración manual de espacios
│   ├── spots.json.example       # ejemplo de formato spots
│   └── requirements.txt
└── PRESENTACION.md              # este archivo
```
