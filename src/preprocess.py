"""
Phase 2 - Multi-Dataset Preprocessing & Hospital Simulation
=============================================================
Pools Brain Tumor MRI (4 classes) + Pneumonia Chest X-Ray (2 classes)
into a unified 6-class partitioned dataset.

Source layout under data/raw/:
    brain_tumer_classification/
        Training/{glioma, meningioma, notumor, pituitary}/
        Testing/ {glioma, meningioma, notumor, pituitary}/
    pneumonia_classification/
        train/{NORMAL, PNEUMONIA}/
        val/  {NORMAL, PNEUMONIA}/
        test/ {NORMAL, PNEUMONIA}/

Output layout written to data/partitions/:
    client_1/{glioma, meningioma, notumor, pituitary, normal, pneumonia}/
    client_2/{glioma, meningioma, notumor, pituitary, normal, pneumonia}/
    client_3/{glioma, meningioma, notumor, pituitary, normal, pneumonia}/
    global_test/{glioma, meningioma, notumor, pituitary, normal, pneumonia}/

All classes use their real names so evaluation reports are human-readable.
Every image is resized to 128x128 grayscale to keep CPU memory low.

Usage (run from project root):
    python src/preprocess.py
"""

import shutil
import sys
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent
RAW_DIR   = PROJ_ROOT / "data" / "raw"
PART_DIR  = PROJ_ROOT / "data" / "partitions"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = (128, 128)
NUM_CLIENTS  = 3
TEST_RATIO   = 0.15
RANDOM_STATE = 42
IMG_EXTS     = {".jpg", ".jpeg", ".png"}

# ── Class definitions ─────────────────────────────────────────────────────────
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PNEUMONIA_MAP = {"NORMAL": "normal", "PNEUMONIA": "pneumonia"}
ALL_CLASSES   = BRAIN_CLASSES + list(PNEUMONIA_MAP.values())   # 6 classes


# ─────────────────────────────────────────────────────────────────────────────
def _collect(folder: Path, cls_name: str, paths: list, labels: list) -> int:
    if not folder.exists():
        return 0
    found = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    paths.extend(found)
    labels.extend([cls_name] * len(found))
    return len(found)


def preprocess_and_save(src: Path, dest_dir: Path) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_dir / src.name), resized)
    return True


def write_split(paths: list, labels: list, split_dir: Path) -> dict:
    counts = {c: 0 for c in ALL_CLASSES}
    total  = len(paths)
    for i, (p, lbl) in enumerate(zip(paths, labels)):
        if i % 500 == 0:
            print(f"    {i}/{total} ...", end="\r")
        if preprocess_and_save(p, split_dir / lbl):
            counts[lbl] += 1
    print(f"    {total}/{total} done.  ")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
def collect_all_images() -> tuple:
    paths, labels = [], []

    # 1. Brain Tumor (Training + Testing pooled)
    print("\n[Dataset 1] Brain Tumor MRI")
    bt_root = RAW_DIR / "brain_tumer_classification"
    for split in ("Training", "Testing"):
        for cls in BRAIN_CLASSES:
            n = _collect(bt_root / split / cls, cls, paths, labels)
            print(f"  {split:8s}/{cls:12s}: {n:5d} images")

    # 2. Pneumonia Chest X-Ray (train + val + test only, skip __MACOSX and chest_xray duplicates)
    print("\n[Dataset 2] Pneumonia Chest X-Ray")
    pn_root = RAW_DIR / "pneomonia_classification"   # folder has typo: pneomonia
    for split in ("train", "val", "test"):
        for src_name, dst_name in PNEUMONIA_MAP.items():
            n = _collect(pn_root / split / src_name, dst_name, paths, labels)
            print(f"  {split:8s}/{src_name:10s} -> {dst_name}: {n:5d} images")

    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
def main():
    BANNER = "=" * 65
    print(BANNER)
    print("  FL Project -- Multi-Dataset Preprocessing (6 classes)")
    print(BANNER)

    print("\n[1/4] Collecting image paths ...")
    all_paths, all_labels = collect_all_images()
    n_total = len(all_paths)
    if n_total == 0:
        print("[ERROR] No images found. Check data/raw/ structure.")
        sys.exit(1)

    print(f"\n  Total images : {n_total}")
    for cls in ALL_CLASSES:
        cnt = all_labels.count(cls)
        print(f"    {cls:12s}: {cnt:5d}  ({cnt/n_total*100:.1f}%)")

    print(f"\n[2/4] Carving out global test set ({int(TEST_RATIO*100)}%) ...")
    train_p, test_p, train_l, test_l = train_test_split(
        all_paths, all_labels,
        test_size=TEST_RATIO,
        stratify=all_labels,
        random_state=RANDOM_STATE,
    )
    print(f"  Training pool : {len(train_p)}")
    print(f"  Global test   : {len(test_p)}")

    print(f"\n[3/4] Partitioning into {NUM_CLIENTS} client splits ...")
    remaining_p = train_p
    remaining_l = train_l
    client_splits = []

    for cid in range(1, NUM_CLIENTS + 1):
        if cid < NUM_CLIENTS:
            n_left   = NUM_CLIENTS - cid
            fraction = 1.0 / (n_left + 1)
            c_p, remaining_p, c_l, remaining_l = train_test_split(
                remaining_p, remaining_l,
                test_size=(1.0 - fraction),
                stratify=remaining_l,
                random_state=RANDOM_STATE + cid,
            )
        else:
            c_p, c_l = remaining_p, remaining_l
        client_splits.append((c_p, c_l))
        print(f"  client_{cid}: {len(c_p)} images | " +
              "  ".join(f"{k}={c_l.count(k)}" for k in ALL_CLASSES))

    print(f"\n[4/4] Writing preprocessed images to {PART_DIR} ...")
    print("       (resizing all to 128x128 grayscale -- this may take a few minutes)")

    # Wipe old partitions for clean run
    if PART_DIR.exists():
        shutil.rmtree(PART_DIR)
    PART_DIR.mkdir(parents=True)

    print("\n  >> global_test")
    cnt = write_split(test_p, test_l, PART_DIR / "global_test")
    print("     " + "  ".join(f"{k}={v}" for k, v in cnt.items()))

    for i, (c_p, c_l) in enumerate(client_splits, start=1):
        print(f"\n  >> client_{i}")
        cnt = write_split(c_p, c_l, PART_DIR / f"client_{i}")
        print("     " + "  ".join(f"{k}={v}" for k, v in cnt.items()))

    # Summary
    print(f"\n{BANNER}")
    print("  Preprocessing complete -- Partition Summary")
    print(BANNER)
    header = f"  {'Split':<14}" + "".join(f"{c:>12}" for c in ALL_CLASSES) + f"{'TOTAL':>8}"
    print(header)
    print("-" * len(header))
    for split_dir in sorted(PART_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        row_counts = []
        for cls in ALL_CLASSES:
            cls_dir = split_dir / cls
            n = sum(1 for f in cls_dir.iterdir() if f.is_file()) if cls_dir.exists() else 0
            row_counts.append(n)
        row = f"  {split_dir.name:<14}" + "".join(f"{n:>12}" for n in row_counts) + f"{sum(row_counts):>8}"
        print(row)
    print(BANNER)


if __name__ == "__main__":
    main()

