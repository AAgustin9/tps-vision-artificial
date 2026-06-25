# Entrenamiento (Google Colab)

## 1. Subir el dataset PKLot a Google Drive

Subi la carpeta del dataset PKLot a tu Google Drive (por ejemplo a
`MyDrive/PKLot`). No importa la estructura interna exacta (PKLot suele venir
como `PKLot/PKLotSegmented/<LOTE>/<CLIMA>/<FECHA>/Empty|Occupied/...`); el
notebook explora la estructura real antes de entrenar.

## 2. Abrir y correr el notebook

1. Abri `PKLot_training.ipynb` en Google Colab (botón derecho sobre el
   archivo en Drive > "Abrir con" > Google Colaboratory, o subilo directo a
   [colab.research.google.com](https://colab.research.google.com)).
2. Activa GPU: Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU.
3. Corré la celda de montaje de Drive y autorizá el acceso cuando lo pida.
4. En la celda de configuración, ajustá `DATASET_PATH` a la ruta real donde
   subiste el dataset, y `OUTPUT_PATH` a la carpeta de Drive donde querés que
   se guarde el modelo entrenado.
5. Corré la celda de exploración de estructura y confirmá que encuentra
   imágenes `Empty`/`Occupied` antes de seguir.
6. Corré el resto de las celdas en orden (carga/split, entrenamiento,
   evaluación, gráficos, export).

## 3. Descargar el modelo entrenado

Al final del notebook se guarda `modelo_pklot.h5` dentro de `OUTPUT_PATH` en
tu Drive. Descargalo a tu máquina local para usarlo con `/inference`.
