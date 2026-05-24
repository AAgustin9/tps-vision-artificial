# TP4 - Proyecto de perspectiva homografica

Aplicacion OpenCV para obtener una homografia desde una webcam y visualizar la transformacion de perspectiva pedida en el TP.

## Que implementa

- Modo QR (`q`): detecta los 4 vertices de un codigo QR con `cv2.QRCodeDetector` y calcula la homografia.
- Modo manual (`h`): permite hacer clic en 4 vertices de un cuadrado visto en perspectiva y calcula la homografia.
- Visualizacion sobre la camara: dibuja una grilla de celdas cuadradas en perspectiva sobre la imagen original.
- Vista frontal: muestra en otra ventana el cuadrado rectificado usando `cv2.warpPerspective`.

## Instalacion

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Ejecucion

```bash
./.venv/bin/python main.py
```

Opciones utiles:

```bash
./.venv/bin/python main.py --camera 0 --square-size 600 --grid-cells 3
```

## Controles

- `q`: entra en modo QR. Alinea un QR en la camara y presiona cualquier tecla para calcular la homografia. Si falla, conserva la homografia anterior.
- `h`: entra en modo manual. Hace clic en los 4 vertices de un cuadrado en perspectiva. Luego del cuarto clic calcula la homografia. Cualquier tecla aborta y conserva la homografia anterior.
- `ESC`: sale.

## Que necesitas conseguir vos

- Una webcam funcional y permisos de camara para la terminal/app que ejecutes.
- Un codigo QR fisico o mostrado en otra pantalla para el modo QR. Puede ser cualquier QR cuadrado y plano.
- Un cuadrado plano para el modo manual. Puede ser una hoja cuadrada, una marca cuadrada impresa, un azulejo, una mesa con esquinas visibles, etc.
- Buena iluminacion y que los 4 vertices esten dentro de la imagen.
- Python 3.10+ recomendado.

## Notas de uso

- Para mejores resultados, manten el QR/cuadrado lo mas plano posible y evita reflejos fuertes.
- En modo manual no importa demasiado el orden de los clics: el programa reordena los puntos como arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda.
- Si tenes mas de una camara, cambia `--camera 1`, `--camera 2`, etc.
