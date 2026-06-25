# Diseño: Detección de espacios de estacionamiento libres/ocupados (PKLot)

## Resumen

Proyecto de visión artificial con dos partes que corren en entornos distintos, dentro
del mismo repo:

- `/training`: notebook de Google Colab que entrena un clasificador binario
  (libre/ocupado) sobre patches del dataset PKLot usando transfer learning con
  MobileNetV2, y exporta el modelo a Google Drive.
- `/inference`: scripts Python para correr localmente (sin GPU) que permiten
  calibrar una sola vez los espacios de un layout de cámara fija, y luego clasificar
  cada espacio en imágenes nuevas, dibujando el resultado sobre la imagen.

Reemplaza por completo el enfoque anterior basado en YOLOv8 (detección automática +
clasificador), cuyos archivos (`app.py`, `src/*`, `README.md`, `requirements.txt`,
`COLAB_GUIDE.md`) ya están eliminados del working tree y se confirman como borrados
(`git rm`) en este cambio.

## Estructura de archivos

```
tpfinal/
├── README.md                          # overview corto: qué es cada carpeta, link a ambas
├── .gitignore                         # ignora modelos .h5, spots.json real, venv, etc.
│
├── training/
│   ├── README.md                      # cómo subir el dataset a Drive y correr el notebook
│   └── PKLot_training.ipynb           # notebook único para Colab
│
└── inference/
    ├── README.md                      # instalación local + uso de calibrate.py y detect.py
    ├── requirements.txt                # opencv-python, tensorflow-cpu, numpy, etc.
    ├── calibrate.py                    # marcado manual de espacios -> spots.json
    ├── detect.py                       # carga spots.json + modelo, clasifica, dibuja resultado
    └── spots.json.example              # ejemplo de formato (el spots.json real no se versiona)
```

`modelo_pklot.h5` (descargado de Drive por el usuario) y el `spots.json` real de cada
layout no se versionan en git — solo se incluye `spots.json.example` para mostrar el
formato esperado.

## Parte 1: Notebook de entrenamiento (`training/PKLot_training.ipynb`)

Todas las celdas con comentarios en español; nombres de variables/funciones en inglés.

Secciones del notebook, en orden:

1. **Setup**: instala/importa dependencias (tensorflow, sklearn, matplotlib, etc.),
   monta Google Drive (`drive.mount`).
2. **Configuración**: variables al inicio, fácilmente editables:
   - `DATASET_PATH = "/content/drive/MyDrive/PKLot/..."`
   - `OUTPUT_PATH = "/content/drive/MyDrive/PKLot_output/"` (carpeta de salida del modelo)
   - hiperparámetros básicos (batch size, epochs, learning rate, image size).
3. **Exploración de estructura**: recorre `DATASET_PATH` recursivamente e imprime el
   árbol de carpetas (hasta cierta profundidad) y conteo de imágenes por carpeta
   `Empty`/`Occupied` encontradas, para que el usuario verifique el path antes de
   entrenar. No asume una única estructura fija (PKLot varía: puede venir como
   `PKLot/PKLotSegmented/<LOT>/<CLIMA>/<FECHA>/Empty|Occupied/`).
4. **Carga del dataset y split**: busca recursivamente todas las imágenes bajo carpetas
   `Empty`/`Occupied` (case-insensitive) y construye una lista de (ruta, label).
   - Si se detecta estructura por fecha (subcarpetas tipo `YYYY-MM-DD`), se hace el
     split 70/15/15 agrupando por fecha (todas las imágenes de un mismo día van al
     mismo split) para evitar data leakage entre frames del mismo video.
   - Si no se detecta esa estructura, fallback a split aleatorio estratificado por
     clase 70/15/15.
   - Imprime conteos finales de train/val/test por clase.
5. **Pipeline de datos**: `tf.data.Dataset` o `ImageDataGenerator` con resize a
   224x224 (input de MobileNetV2), normalización, y data augmentation en el set de
   train (flip horizontal, brightness, rotación leve ±15°). Val/test sin augmentation.
6. **Modelo**: MobileNetV2 preentrenado (`include_top=False`, pesos `imagenet`) +
   `GlobalAveragePooling2D` + capa densa + salida sigmoide (clasificación binaria).
   Base congelada inicialmente (fine-tuning simple, sin etapas múltiples salvo que
   el usuario lo pida).
7. **Entrenamiento**: `model.compile` (Adam, binary_crossentropy, métricas
   accuracy/precision/recall), `model.fit` con callbacks:
   - `EarlyStopping` (monitor val_loss, patience razonable)
   - `ModelCheckpoint` (guarda el mejor modelo)
   - `ReduceLROnPlateau`
8. **Evaluación en test**: accuracy, precision, recall, F1 (sklearn
   `classification_report`), matriz de confusión (`ConfusionMatrixDisplay` o
   `seaborn.heatmap`).
9. **Gráficos**: curvas de loss y accuracy (train vs val) por epoch con matplotlib.
10. **Export**: guarda el modelo final en `OUTPUT_PATH` dentro de Drive, en formato
    `.h5` (nombre fijo tipo `modelo_pklot.h5`) para que el usuario lo descargue y use
    en `/inference`.

## Parte 2: Inferencia local (`/inference`)

### `calibrate.py`

- Uso: `python calibrate.py --image referencia.jpg --output spots.json`
- Abre la imagen de referencia con OpenCV en una ventana.
- El usuario hace click para marcar las 4 esquinas de cada espacio (rectángulo simple,
  no necesariamente alineado a ejes — guarda los 4 puntos tal cual se clickearon).
- Tras el 4to click de un espacio, se cierra ese espacio (se dibuja el contorno sobre
  la ventana), se le asigna un ID incremental (`spot_1`, `spot_2`, ...) y el usuario
  puede seguir marcando el siguiente espacio.
- Tecla para deshacer el último punto/espacio, tecla para terminar y guardar.
- Guarda `spots.json` con la lista de espacios: `{"spots": [{"id": "spot_1", "points":
  [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}, ...]}`.
- Es un script que se corre una sola vez por layout de cámara.

### `detect.py`

- Uso: `python detect.py --image estacionamiento.jpg --spots spots.json --model
  modelo_pklot.h5 [--output resultado.jpg]`
- Fuerza ejecución en CPU (no requiere GPU).
- Carga `spots.json` y el modelo `.h5` (Keras).
- Para cada espacio: recorta el cuadrilátero definido por sus 4 puntos (usando
  bounding box o warpPerspective si el cuadrilátero no es un rectángulo alineado),
  resize a 224x224, normaliza igual que en entrenamiento, pasa por el modelo →
  probabilidad de "ocupado".
- Dibuja sobre una copia de la imagen original: rectángulo/contorno verde si libre,
  rojo si ocupado, con el ID del espacio.
- Calcula y muestra en la imagen el conteo total, ej. `"12/40 espacios libres"`.
- Guarda la imagen resultado (en `--output` o nombre derivado de la imagen de entrada)
  y la muestra en pantalla (`cv2.imshow` + `waitKey`).

### `requirements.txt`

`opencv-python`, `tensorflow-cpu` (o `tensorflow` con nota de usar CPU), `numpy`,
sin más dependencias que las estrictamente necesarias para correr `calibrate.py` y
`detect.py`.

## Manejo de errores

- `detect.py`: si `spots.json` o el modelo no existen, o la imagen no se puede leer,
  error claro por stderr y salida con código distinto de 0 (sin estructuras de
  manejo de errores especulativas más allá de estos casos reales).
- `calibrate.py`: ignora clicks fuera de la ventana de imagen; al guardar, si ya
  existe `spots.json` en esa ruta, lo sobreescribe (comportamiento esperado de un
  script de calibración que se vuelve a correr).

## Testing / verificación

- No hay suite de tests automatizada formal dada la naturaleza del proyecto (notebook
  interactivo + scripts de CV con I/O de mouse/imagen). La verificación se hace:
  - Notebook: ejecución manual en Colab por el usuario, inspeccionando las celdas de
    exploración de estructura y las métricas/gráficos de evaluación.
  - `detect.py`: se puede probar con una imagen de muestra y un `spots.json.example`
    de pocos espacios para confirmar que carga el modelo, recorta y dibuja
    correctamente, sin necesitar GPU.

## Fuera de alcance

- No se implementa tracking entre frames/video, solo clasificación de una imagen
  estática a la vez.
- No se reentrena ni se hace fine-tuning incremental desde `/inference`.
- No se contempla mover/redefinir spots automáticamente si la cámara se mueve.
