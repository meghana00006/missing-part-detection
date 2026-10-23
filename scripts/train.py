"""
Retrain the detector at a larger image size so tiny parts (pins, bolts) are learned.
 
Run from the project root:
    python scripts/train.py
Afterwards the new weights are copied to model/best.pt, which UI/app.py picks up
automatically (it loads the newest .pt in /model).
"""
import shutil
from pathlib import Path
 
from ultralytics import YOLO
 
ROOT = Path(__file__).resolve().parent.parent
IMGSZ = 1280          # photos are 5568x3712; 640 shrinks pins to ~15 px
EPOCHS = 100
BATCH = 4             # raise to 8/16 if you have a GPU with enough memory
 
model = YOLO("yolov8s.pt")                       # 's' sees small objects better than 'n'
model.train(
    data=str(ROOT / "data.yaml"),
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    patience=30,
    workers=2,
    project=str(ROOT / "runs" / "detect"),
    name="industrial_parts_1280",
    exist_ok=True,
)
 
best = ROOT / "runs" / "detect" / "industrial_parts_1280" / "weights" / "best.pt"
if best.exists():
    (ROOT / "model").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "model" / "best.pt")
    print("Copied ->", ROOT / "model" / "best.pt")
else:
    print("best.pt not found - check the training log above.")