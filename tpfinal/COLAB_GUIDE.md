# Running the Parking Detector on Google Colab

**Goal:** Use Colab's free T4 GPU to train the model, then run the app locally.

**Assumption:** PKLot dataset is already uploaded to your Google Drive.

---

## Part 1 — Setup in Colab

### Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com), create a new notebook, and enable the GPU:

> Runtime → Change runtime type → T4 GPU → Save

---

### Step 2 — Mount your Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize when prompted.

---

### Step 3 — Upload the project files

Option A — if your project is on GitHub:
```bash
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
```

Option B — upload the project folder to Drive and copy it:
```bash
!cp -r "/content/drive/MyDrive/tpfinal" /content/tpfinal
%cd /content/tpfinal
```

---

### Step 4 — Install dependencies

```bash
!pip install -q tensorflow opencv-python-headless streamlit numpy pandas matplotlib seaborn scikit-learn Pillow
```

---

## Part 2 — Prepare the Dataset

### Step 5 — Link the PKLot dataset from Drive

Replace the path below with wherever you placed the PKLot folder in your Drive:

```bash
!ln -s "/content/drive/MyDrive/PKLot" /content/tpfinal/data/pklot_raw
```

Or copy it (slower but avoids symlink issues):
```bash
!cp -r "/content/drive/MyDrive/PKLot" /content/tpfinal/data/pklot_raw
```

---

### Step 6 — Run the dataset preparation script

```bash
!python src/prepare_dataset.py
```

This crops up to 2,000 empty and 2,000 occupied space images from the raw dataset.
Expected output:
```
Dataset preparation complete:
  Empty:    2000 crops → data/processed/empty/
  Occupied: 2000 crops → data/processed/occupied/
  Total:    4000
```

> **Tip:** Save the processed crops to Drive so you don't need to redo this step next session:
> ```bash
> !cp -r /content/tpfinal/data/processed "/content/drive/MyDrive/tpfinal_processed"
> ```

---

## Part 3 — Train the Model

### Step 7 — Run training

```bash
!python src/train.py
```

Expected time on T4: **3–5 minutes** (vs 30–60 min on CPU).

Training runs in two phases:
- Phase 1: 30 epochs with MobileNetV2 base frozen
- Phase 2: 10 epochs fine-tuning the last 20 layers

At the end you'll see a classification report. Target: >90% validation accuracy.

---

### Step 8 — Save the model to Drive

```bash
!cp /content/tpfinal/models/parking_classifier.h5 "/content/drive/MyDrive/"
!cp /content/tpfinal/results/training_curves.png "/content/drive/MyDrive/"
!cp /content/tpfinal/results/confusion_matrix.png "/content/drive/MyDrive/"
```

---

## Part 4 — Run the App Locally

### Step 9 — Download the model

In Colab, click the folder icon on the left sidebar, navigate to `tpfinal/models/`, right-click `parking_classifier.h5` and download it.

Or download it from Drive directly to your computer.

---

### Step 10 — Place the model in your local project

```bash
# From your local machine, inside the tpfinal folder:
mv ~/Downloads/parking_classifier.h5 models/
```

---

### Step 11 — Install dependencies locally (first time only)

```bash
pip install -r requirements.txt
```

---

### Step 12 — Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Quick Reference

| Task | Where |
|---|---|
| Dataset preparation | Colab |
| Model training | Colab (T4 GPU) |
| Model storage | Google Drive |
| Streamlit app | Local machine |
| Live camera | Local machine only |

---

## Troubleshooting

**"Dataset not found"** — Check that the path in Step 5 matches exactly where PKLot is in your Drive. Run `!ls /content/tpfinal/data/pklot_raw` to verify.

**"Model not found"** — Make sure `parking_classifier.h5` is inside the `models/` folder in your local project before running the app.

**Colab session expired mid-training** — Re-run from Step 6. If you saved the processed crops to Drive (Step 6 tip), link them back and skip straight to Step 7:
```bash
!cp -r "/content/drive/MyDrive/tpfinal_processed" /content/tpfinal/data/processed
```

**TensorFlow version warning** — Safe to ignore as long as training completes and the `.h5` file is saved.
