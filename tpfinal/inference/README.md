# Inferencia local

Corre 100% en tu máquina, sin GPU (usa `tensorflow-cpu`).

## 1. Instalar dependencias

```bash
cd inference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Calibrar los espacios (una sola vez por cámara/layout)

```bash
python calibrate.py --image referencia.jpg --output spots.json
```

Se abre una ventana con la imagen de referencia. Hacé click en las 4 esquinas
de cada espacio (en cualquier orden) para cerrarlo; repetí para cada espacio.

Teclas:
- `u`: deshacer el último punto o el último espacio cerrado.
- `s`: guardar `spots.json` y salir.
- `q` / `ESC`: salir sin guardar.

Mirá `spots.json.example` para ver el formato esperado.

## 3. Clasificar una imagen nueva

```bash
python detect.py --image estacionamiento.jpg --spots spots.json --model modelo_pklot.h5
```

Esto recorta cada espacio según `spots.json`, lo clasifica con el modelo
descargado de Colab, y muestra/guarda la imagen con cada espacio en verde
(libre) o rojo (ocupado), junto con el conteo total (ej. `"12/40 espacios
libres"`). La imagen resultado se guarda como `<imagen>_resultado.jpg` salvo
que se pase `--output`.
