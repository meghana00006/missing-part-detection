"""
Build the reference library used by UI/app.py.
 
Run from the project root, ONCE at first, and again whenever you add or relabel images:
    python scripts/build_references.py
 
It reads every image + YOLO label file in dataset/images/{train,val} and writes
    references/expected_parts.json   (labelled parts per reference photo)
    references/images/*.jpg          (small copies used for feature matching)
"""
import json
from pathlib import Path
 
import cv2
 
ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val"]                       # every split that has images + labels
OUT_JSON = ROOT / "references" / "expected_parts.json"
OUT_IMG = ROOT / "references" / "images"
CLASS_NAMES = {0: "bolt", 1: "bearing", 2: "plate", 3: "pin"}
SMALL_SIDE = 1400                               # longest side of the stored feature copy
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
 
OUT_IMG.mkdir(parents=True, exist_ok=True)
for old in OUT_IMG.glob("*.jpg"):               # start clean so removed images don't linger
    old.unlink()
 
refs = {}
for split in SPLITS:
    img_dir = ROOT / "dataset" / "images" / split
    lbl_dir = ROOT / "dataset" / "labels" / split
    if not img_dir.exists():
        print(f"skip split '{split}' (no folder)")
        continue
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            print("skip (no label):", img_path.name)
            continue
        img = cv2.imread(str(img_path))          # cv2 applies EXIF rotation, same as the app
        if img is None:
            print("skip (unreadable):", img_path.name)
            continue
        h, w = img.shape[:2]
        parts, counts = [], {}
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:5])
            name = CLASS_NAMES.get(c, f"class_{c}")
            parts.append({
                "class": c, "class_name": name,
                "bbox": [int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                         int((cx + bw / 2) * w), int((cy + bh / 2) * h)],
            })
            counts[name] = counts.get(name, 0) + 1
        refs[img_path.name] = {"shape": [h, w], "parts": parts, "part_counts": counts}
        s = min(1.0, SMALL_SIDE / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT_IMG / f"{img_path.stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
 
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(refs, indent=1))
print(f"Done: {len(refs)} reference views -> {OUT_JSON}")