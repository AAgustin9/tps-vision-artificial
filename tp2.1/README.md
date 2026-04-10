# TP2.1 - Deteccion y clasificacion de formas con webcam

Implementacion alineada con la consigna del PDF: el sistema toma referencias desde imagenes, abre la webcam de la computadora y detecta/clasifica objetos en tiempo real usando contornos y `matchShapes()`.

## Figuras configuradas

Las referencias actuales estan en [input](/Users/agussoul/projects/ua/va/tps/tp2.1/input):

- `circle.jpeg` -> `circle`
- `rectangle.jpeg` -> `rectangle_outline`
- `star.png` -> `star`

El nombre detectado sale del nombre del archivo de referencia. Caso especial: `rectangle` se normaliza a `rectangle_outline`.

## Instalacion

```bash
cd /Users/agussoul/projects/ua/va/tps/tp2.1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso principal: webcam en tiempo real

```bash
python3 main.py
```

Eso hace lo siguiente:

- carga las imagenes de referencia desde `input/`
- abre la webcam (`camera-index 0`)
- muestra una ventana con la imagen anotada
- muestra otra ventana con la mascara binaria
- crea una ventana de controles con sliders

### Controles sugeridos por la consigna

- `Threshold`: umbral de binarizacion
- `Kernel`: tamano del elemento estructural para operaciones morfologicas
- `Min area`: area minima para descartar ruido
- `Match x1000`: distancia maxima aceptada para `matchShapes()`

### Teclas

- `q`: salir
- `s`: guardar el frame anotado y su mascara en `output/`

## Opciones

```bash
python3 main.py --help
```

Opciones principales:

- `--camera-index`: el indice de webcam a usar
- `--input-dir`: carpeta con imagenes de referencia
- `--image`: procesa una imagen puntual en vez de abrir la webcam
- `--output-dir`: carpeta de salida para modo imagen
- `--save-masks`: guarda la mascara binaria en modo imagen

## Modo imagen

Sirve para pruebas rapidas sin webcam:

```bash
python3 main.py --image input/circle.jpeg --output-dir output --save-masks
```

## Salida visual

La salida principal es una ventana con la imagen original anotada:

- objetos reconocidos con color y etiqueta
- objetos desconocidos en rojo
- mascara binaria en una ventana separada

Esto sigue el output pedido en la consigna: localizacion de objetos, etiqueta de clase y visualizacion de pasos intermedios.

## Pruebas

```bash
pytest
```

Las pruebas validan:

- carga de referencias
- clasificacion con `matchShapes()`
- deteccion de objetos desconocidos

## Nota importante

Para que funcione bien con webcam, la consigna asume ambiente controlado:

- fondo liso y contrastante
- buena iluminacion
- camara frontal o casi perpendicular
- formas bien recortadas respecto del fondo
