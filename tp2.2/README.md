# TP2.2 - Clasificacion con Machine Learning

Proyecto en Python que toma la deteccion de formas del TP anterior y reemplaza la clasificacion manual por un modelo entrenado con `scikit-learn`.

La solucion implementa lo que pide la consigna:

- genera descriptores con los 7 momentos invariantes de Hu
- arma un dataset en CSV
- entrena un clasificador con machine learning
- guarda el modelo entrenado en disco
- usa ese modelo para clasificar las formas detectadas en nuevas imagenes

## Estructura

```text
tp2.2/
├── data/
├── input/
├── models/
├── output/
├── train/
│   ├── circle/
│   ├── rectangle_outline/
│   └── star/
├── requirements.txt
├── README.md
├── main.py
├── src/
│   └── tp22/
│       ├── __init__.py
│       ├── cli.py
│       ├── descriptors.py
│       ├── detector.py
│       ├── drawing.py
│       ├── synthetic.py
│       └── training.py
└── tests/
    └── test_pipeline.py
```

## Criterio de implementacion

- Los descriptores de cada figura se construyen con momentos invariantes de Hu y transformacion logaritmica.
- El dataset puede combinar muestras reales del alumno en `train/` con muestras sinteticas generadas por el proyecto.
- El entrenamiento usa un `DecisionTreeClassifier`, alineado con la consigna y con la documentacion enlazada en el PDF.
- El modelo se persiste con `joblib`.

## Instalacion

```bash
cd /Users/agussoul/projects/ua/va/tps/tp2.2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Flujo recomendado

### 1. Generar dataset + entrenar el modelo

Esto crea `data/hu_moments.csv`, guarda el modelo en `models/shape_classifier.joblib` y escribe metricas en `output/training_metrics.json`.

```bash
python3 main.py bootstrap
```

### 2. Clasificar imagenes

```bash
python3 main.py classify --input-dir input --output-dir output --save-masks
```

### Generar imagenes de ejemplo para `input/`

```bash
python3 generate_input_samples.py
```

Por default crea 2 imagenes por clase dentro de `input/`.

### 3. Ver ayuda

```bash
python3 main.py --help
python3 main.py generate-dataset --help
python3 main.py train --help
python3 main.py classify --help
```

## Comandos utiles

### Generar solo el dataset

```bash
python3 main.py generate-dataset --output-csv data/hu_moments.csv
```

### Entrenar desde un CSV existente

```bash
python3 main.py train \
  --dataset-csv data/hu_moments.csv \
  --model-path models/shape_classifier.joblib \
  --metrics-path output/training_metrics.json
```

### Clasificar una sola imagen

```bash
python3 main.py classify \
  --image /ruta/a/imagen.png \
  --output-dir output \
  --model-path models/shape_classifier.joblib
```

## Que tienes que agregar tu mismo

El proyecto ya funciona con datos sinteticos, pero para ajustarlo a tu entrega conviene que agregues ejemplos reales tuyos:

- coloca imagenes de circulos en `train/circle/`
- coloca imagenes de rectangulos de contorno en `train/rectangle_outline/`
- coloca imagenes de estrellas en `train/star/`
- coloca las imagenes a clasificar en `input/`

Cada vez que agregues nuevas imagenes de entrenamiento, vuelve a correr:

```bash
python3 main.py bootstrap
```

## Salidas

Al ejecutar `classify` se generan:

- una imagen anotada por cada entrada
- `output/detections.json` con etiqueta, confianza, bounding box y valores Hu
- opcionalmente mascaras binarias con `--save-masks`

## Pruebas

```bash
pytest
```

## Nota

Si el modelo clasifica mal alguna imagen real, normalmente se corrige agregando mas ejemplos reales en `train/` y reentrenando.
