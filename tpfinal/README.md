# PKLot: Detección de espacios de estacionamiento libres/ocupados

Proyecto de visión artificial con dos partes que corren en entornos distintos:

- [`/training`](training/README.md): notebook de Google Colab que entrena un
  clasificador binario (libre/ocupado) con transfer learning sobre MobileNetV2
  usando el dataset PKLot, y exporta el modelo a Google Drive.
- [`/inference`](inference/README.md): scripts Python para correr localmente
  (sin GPU) que calibran los espacios de un layout de cámara fija una sola
  vez, y luego clasifican y dibujan el resultado sobre fotos nuevas del
  estacionamiento.

Ver el diseño completo en
[`docs/superpowers/specs/2026-06-25-pklot-parking-detection-design.md`](docs/superpowers/specs/2026-06-25-pklot-parking-detection-design.md).
