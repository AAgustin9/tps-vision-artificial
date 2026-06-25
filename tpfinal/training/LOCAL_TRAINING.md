# Entrenamiento local con GPU

Guía para correr `train_local.py` en una PC con GPU dedicada.  
El script es equivalente al notebook de Colab pero corre desde la terminal y aprovecha toda la VRAM disponible.

---

## Requisitos de hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | NVIDIA 4 GB VRAM | NVIDIA 8+ GB VRAM |
| RAM | 8 GB | 16 GB |
| Disco | 5 GB libres | 10 GB libres |

> **Mac con Apple Silicon (M1/M2/M3):** funciona con el backend `tensorflow-metal`. Ver sección correspondiente más abajo.

---

## Paso 1 — Instalar CUDA y cuDNN (solo NVIDIA)

Saltear este paso si usás CPU o Apple Silicon.

1. Verificar qué versión de CUDA soporta tu driver:
   ```bash
   nvidia-smi
   ```
   La esquina superior derecha muestra la versión máxima de CUDA compatible.

2. Instalar CUDA Toolkit 12.x desde [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

3. TensorFlow 2.17+ en Linux instala cuDNN automáticamente vía pip (no hace falta instalarlo manualmente).

---

## Paso 2 — Crear entorno virtual

```bash
# Desde la raiz del proyecto
python -m venv .venv

# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## Paso 3 — Instalar dependencias

### Linux (NVIDIA GPU)
```bash
pip install -r training/requirements-local.txt
```

### Windows (NVIDIA GPU)
```bash
pip install tensorflow>=2.17.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 numpy>=1.26.0
```
En Windows el soporte CUDA no viene incluido en el paquete pip — necesitás instalar CUDA Toolkit y cuDNN manualmente desde el sitio de NVIDIA antes de instalar TensorFlow.

### Mac Apple Silicon (M1/M2/M3)
```bash
pip install tensorflow-macos tensorflow-metal scikit-learn matplotlib numpy
```

---

## Paso 4 — Verificar que TensorFlow ve la GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Deberías ver algo como:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Si la lista está vacía revisá la instalación de CUDA o el entorno virtual.

---

## Paso 5 — Correr el entrenamiento

### Opción A: pasarle el zip directamente (lo descomprime automático)
```bash
python training/train_local.py \
    --zip /ruta/a/PKLot.zip \
    --output ./training/output
```

### Opción B: dataset ya descomprimido
```bash
python training/train_local.py \
    --dataset /ruta/a/PKLot_data \
    --output ./training/output
```

### Con parámetros personalizados
```bash
python training/train_local.py \
    --zip PKLot.zip \
    --output ./training/output \
    --epochs 30 \
    --batch 64 \
    --img-size 96 \
    --lr 0.0001
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--zip` | — | Ruta al PKLot.zip (mutuamente exclusivo con `--dataset`) |
| `--dataset` | — | Ruta a la carpeta ya descomprimida |
| `--output` | `./output` | Carpeta donde se guardan modelo y gráficos |
| `--epochs` | 20 | Épocas máximas (EarlyStopping puede cortar antes) |
| `--batch` | 32 | Batch size (bajar a 16 si hay OOM) |
| `--img-size` | 96 | Tamaño de imagen cuadrada en píxeles |
| `--lr` | 1e-4 | Learning rate inicial |
| `--no-mixed` | False | Desactiva mixed precision float16 |

---

## Paso 6 — Monitorear con TensorBoard (opcional)

En otra terminal, con el entorno virtual activado:

```bash
tensorboard --logdir ./training/output/logs
```

Luego abrir [http://localhost:6006](http://localhost:6006) en el navegador.

---

## Paso 7 — Resultados

Al terminar el entrenamiento encontrás en `--output`:

```
output/
├── modelo_pklot.keras       # modelo final para usar en detect.py
├── best_checkpoint.keras    # mejor checkpoint por val_loss
├── resultados.png           # curvas de loss/accuracy + confusion matrix
└── logs/                    # logs de TensorBoard
```

### Usar el modelo entrenado con detect.py

```bash
python inference/detect.py \
    --model ./training/output/modelo_pklot.keras \
    --image /ruta/a/foto.jpg \
    --spots inference/spots.json.example
```

---

## Solución de problemas

### OOM (out of memory) durante el entrenamiento
Reducir batch size e imagen:
```bash
python training/train_local.py --zip PKLot.zip --batch 16 --img-size 64
```

### `No module named tensorflow`
El entorno virtual no está activado, o TensorFlow no se instaló correctamente:
```bash
source .venv/bin/activate   # Linux/Mac
pip install tensorflow
```

### GPU no detectada en Windows
Verificar que CUDA Toolkit y cuDNN están instalados y que las variables de entorno `CUDA_PATH` y `PATH` apuntan a la instalación correcta.

### `mixed_float16` causa NaN en los gradientes
Correr con `--no-mixed` para deshabilitar mixed precision:
```bash
python training/train_local.py --zip PKLot.zip --no-mixed
```
