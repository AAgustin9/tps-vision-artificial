### Task 1: Limpieza del repo y scaffolding de carpetas

**Files:**
- Delete (git rm): `app.py`, `requirements.txt`, `README.md`, `COLAB_GUIDE.md`, `src/convert_to_yolo.py`, `src/detector.py`, `src/train_yolo.py`, `src/utils.py`
- Create: `.gitignore`
- Create: `training/.gitkeep` (placeholder, removed once notebook exists in Task 7)
- Create: `inference/.gitkeep` (placeholder, removed once scripts exist in Task 3/6)

**Interfaces:**
- Produces: clean repo root with empty `training/` and `inference/` directories ready for subsequent tasks.

- [ ] **Step 1: Confirm the deleted files and stage the deletion**

```bash
git status
```

Expected: the 8 files (`app.py`, `requirements.txt`, `README.md`, `COLAB_GUIDE.md`, `src/convert_to_yolo.py`, `src/detector.py`, `src/train_yolo.py`, `src/utils.py`) are listed as tracked/unmodified — they have NOT been pre-deleted in this workspace (unlike the original checkout). Just proceed to remove them with `git rm` directly.

```bash
git rm app.py requirements.txt README.md COLAB_GUIDE.md src/convert_to_yolo.py src/detector.py src/train_yolo.py src/utils.py
```

Expected: `rm 'app.py'` etc. printed for each file; no errors. If `src/` becomes empty, git will not track the directory itself, which is fine — it will be removed.

- [ ] **Step 2: Create the new directory skeletons**

```bash
mkdir -p training inference
touch training/.gitkeep inference/.gitkeep
```

- [ ] **Step 3: Create `.gitignore`**

```
# Modelos entrenados (se descargan de Drive, no se versionan)
*.h5
*.keras
saved_model/

# Calibración real de un layout específico (solo se versiona spots.json.example)
inference/spots.json

# Python
__pycache__/
*.pyc
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Sistema
.DS_Store
```

Write this to `.gitignore` at the repo root for this subproject (i.e. the `tpfinal/` directory you are working in).

- [ ] **Step 4: Verify the resulting tree**

```bash
git status
find . -maxdepth 2 -not -path '*/.git*'
```

Expected: `git status` shows the 8 deletions staged plus `.gitignore`, `training/.gitkeep`, `inference/.gitkeep` as new files. `find` shows `training/` and `inference/` directories present.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Remove YOLOv8 approach, scaffold training/ and inference/ folders"
```
