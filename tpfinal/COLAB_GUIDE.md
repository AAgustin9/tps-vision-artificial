# Running the Parking Detector on Google Colab

**Goal:** Use Colab's free T4 GPU to train a YOLOv8 object detector that automatically locates parking spaces and classifies each as empty/occupied — then run it locally to detect empty spots from an uploaded image or your live camera.

**Setup assumed:**
- Repo: [github.com/AAgustin9/tps-vision-artificial](https://github.com/AAgustin9/tps-vision-artificial) (public), project lives in `tpfinal/`
- PKLot dataset zip uploaded to the root of your Google Drive ("My Drive")

---

## Part 1 — Setup in Colab

### Step 1 — Open a new Colab notebook with GPU

[colab.research.google.com](https://colab.research.google.com) → new notebook → **Runtime → Change runtime type → T4 GPU → Save**

### Step 2 — Mount your Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3 — Clone the repo

```bash
%cd /content
!git clone https://github.com/AAgustin9/tps-vision-artificial.git
%cd tps-vision-artificial/tpfinal
!pwd
```

`!pwd` must print exactly `/content/tps-vision-artificial/tpfinal`. If you ever reconnect to a fresh runtime or re-run this step, always start with `%cd /content` first — re-running the clone while already inside the repo nests it inside itself and breaks every path below.

### Step 4 — Install dependencies

```bash
!pip install -q ultralytics opencv-python-headless streamlit numpy Pillow
```

---

## Part 2 — Prepare the Dataset

### Step 5 — Extract the dataset once, then cache it in Drive

`PKLot.zip` lives in Drive. Re-unzipping it every session is slow and error-prone. Instead, **extract it once** and keep the already-extracted folder in Drive — every future session just copies the extracted folder directly, no zip step needed.

**First time only** — extract and verify, then push the result back to Drive:

```bash
!pwd  # should be /content/tps-vision-artificial/tpfinal
!mkdir -p data/pklot_raw
!unzip -q "/content/drive/MyDrive/PKLot.zip" -d data/pklot_raw

# sanity check: should show train/valid/test folders with images + _annotations.coco.json
!ls data/pklot_raw

# cache the extracted folder in Drive for all future sessions
!cp -r data/pklot_raw "/content/drive/MyDrive/PKLot_extracted"
```

**Every session after that** — skip the zip entirely and just copy the cached extracted folder:

```bash
!pwd  # should be /content/tps-vision-artificial/tpfinal
!cp -r "/content/drive/MyDrive/PKLot_extracted" data/pklot_raw
!ls data/pklot_raw
```

### Step 6 — Convert the dataset to YOLO format

```bash
!python src/convert_to_yolo.py
```

This reads the COCO annotations (`space-empty`/`space-occupied` categories) and writes a YOLO-format dataset to `data/yolo/` — images plus per-image label `.txt` files and a `data.yaml`.

---

## Part 3 — Train the Model

### Step 7 — Run training

```bash
!python src/train_yolo.py
```

Fine-tunes YOLOv8n on the T4 GPU for up to 50 epochs (early stopping enabled). Reports mAP50/mAP50-95 on the test split at the end. The best weights are copied to `models/parking_yolo.pt`.

### Step 8 — Save the trained model to Drive

```bash
!cp models/parking_yolo.pt "/content/drive/MyDrive/"
```

---

## Part 4 — Run the App Locally

### Step 9 — Download the model to your computer

From Drive, download `parking_yolo.pt` to your local machine (e.g. into `~/Downloads`).

### Step 10 — Place it in your local project

```bash
# from inside your local tpfinal folder
mv ~/Downloads/parking_yolo.pt models/
```

### Step 11 — Install dependencies locally (first time only)

Use Python 3.9–3.12 (Ultralytics/PyTorch don't yet support 3.13+):

```bash
pip install -r requirements.txt
```

### Step 12 — Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### Step 13 — Choose Upload Image or Live Camera

The sidebar has a **Mode** selector with two options:

- **Upload Image** — pick an existing photo (`.jpg`/`.jpeg`/`.png`); the model automatically locates every visible parking space and classifies it empty/occupied — no manual setup required.
- **Live Camera** — uses your machine's webcam directly (must run locally — Colab has no camera access). Grant browser camera permission when prompted.

Both modes run the same trained detector on whatever image/frame they get. Detection quality depends on how visually similar the input is to PKLot's top-down lot photos — don't expect it to generalize well to very different camera angles or lot layouts.

---

## Troubleshooting

**"Dataset not found" or "No annotated images converted"** — usually means the repo isn't cloned where you think, or the extracted dataset doesn't match the expected `train/valid/test` + COCO JSON layout. Check with `!ls data/pklot_raw` and `!ls data/pklot_raw/train`.

**"Model not found" in the app** — make sure `parking_yolo.pt` is inside `models/` in your local project before running `streamlit run app.py`.

**Colab session expired mid-training** — just re-run `python src/train_yolo.py`; Ultralytics resumes cleanly since the dataset conversion (Step 6) doesn't need to be redone unless the session itself reset and `data/yolo` was wiped along with it.

**Camera mode doesn't show anything** — Live Camera only works when running the app locally (`streamlit run app.py` on your machine), not inside Colab.
