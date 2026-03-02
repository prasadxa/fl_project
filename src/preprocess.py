"""
Phase 2 - Brain Tumor Dataset Preprocessing & Hospital Simulation
==================================================================
KEY DESIGN: All images from Train_1, Train_2, Test_1, Test_2 are POOLED
into a single unified folder first, then re-split into our own stratified
FL partitions. This ensures maximum data usage and eliminates artificial
train/test boundaries from the original datasets.

Source layout under DataSet/:
    brain_tumer_train/
        Train_1/{glioma, meningioma, notumor, pituitary}/   <- POOLED
        Train_2/{glioma, meningioma, notumor, pituitary}/   <- POOLED
    brain_tumer_testing/
        Test_1/{glioma, meningioma, notumor, pituitary}/    <- POOLED
        Test_2/{glioma, meningioma, notumor, pituitary}/    <- POOLED

Step 1 -- Unified pool written to data/all_images/:
    data/all_images/{glioma, meningioma, notumor, pituitary}/
    (ALL images from ALL original splits merged into one folder per class,
     capped at MAX_PER_CLASS for perfect class balance)

Step 2 -- Re-partitioned from that unified pool to data/partitions/:
    client_1/{glioma, meningioma, notumor, pituitary}/
    client_2/{glioma, meningioma, notumor, pituitary}/
    client_3/{glioma, meningioma, notumor, pituitary}/
    global_test/{glioma, meningioma, notumor, pituitary}/

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
PROJ_ROOT    = Path(__file__).parent.parent
DATASET_DIR  = PROJ_ROOT / "DataSet"
UNIFIED_DIR  = PROJ_ROOT / "data" / "all_images"   # intermediate pool
PART_DIR     = PROJ_ROOT / "data" / "partitions"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = (128, 128)
NUM_CLIENTS   = 3
TEST_RATIO    = 0.15
RANDOM_STATE  = 42
IMG_EXTS      = {".jpg", ".jpeg", ".png"}
MAX_PER_CLASS = 1500   # cap every class at this count → balanced 6×1500

# ── Class definitions ─────────────────────────────────────────────────────────
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PNEUMONIA_MAP = {"NORMAL": "normal", "PNEUMONIA": "pneumonia"}  # folder → class name
ALL_CLASSES   = BRAIN_CLASSES + list(PNEUMONIA_MAP.values())   # 6 classes

# Brain tumor sub-directories inside DataSet/ (all four splits pooled)
SOURCE_SPLITS = [
    "brain_tumer_train/Train_1",
    "brain_tumer_train/Train_2",
    "brain_tumer_testing/Test_1",
    "brain_tumer_testing/Test_2",
]
# Pneumonia sub-directories inside DataSet/
PNEUMONIA_SPLITS = [
    "pneumonia_train",
    "pneumonia_testing",
]


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_and_save(src: Path, dest_dir: Path, out_name: str | None = None) -> bool:
    """Resize to 128×128 greyscale and save. out_name overrides src.name."""
    img = cv2.imread(str(src))
    if img is None:
        return False
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = out_name if out_name else src.name
    cv2.imwrite(str(dest_dir / name), resized)
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
def build_unified_pool() -> dict:
    """
    STEP 1 – Merge all DataSet/ splits into data/all_images/.

    Brain tumor: Train_1, Train_2, Test_1, Test_2
    Pneumonia  : pneumonia_train, pneumonia_testing

    Images are renamed sequentially ({class}_{n:05d}.jpg) so there are no
    filename collisions between splits. Each class is capped at MAX_PER_CLASS.

    Returns {class_name: count}.
    """
    print("\n[Step 1] Building unified image pool → data/all_images/")
    print("         (brain: Train_1+Train_2+Test_1+Test_2 | pneumonia: train+testing)")

    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    UNIFIED_DIR.mkdir(parents=True)

    counters: dict[str, int] = {c: 0 for c in ALL_CLASSES}

    # ── Brain Tumor ───────────────────────────────────────────────────────────
    for split_rel in SOURCE_SPLITS:
        split_dir = DATASET_DIR / split_rel
        if not split_dir.exists():
            print(f"  [WARN] Not found: {split_dir}  — skipping")
            continue
        for cls in BRAIN_CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            images = [p for p in sorted(cls_dir.iterdir())
                      if p.suffix.lower() in IMG_EXTS]
            added = 0
            for img_path in images:
                if counters[cls] >= MAX_PER_CLASS:
                    break
                counters[cls] += 1
                added += 1
                out_name = f"{cls}_{counters[cls]:05d}.jpg"
                preprocess_and_save(img_path, UNIFIED_DIR / cls, out_name)
            print(f"  Pooled  {split_rel:35s}/{cls:12s}: {added:5d}  "
                  f"(total={counters[cls]}  cap={MAX_PER_CLASS})")

    # ── Pneumonia ─────────────────────────────────────────────────────────────
    for split_rel in PNEUMONIA_SPLITS:
        split_dir = DATASET_DIR / split_rel
        if not split_dir.exists():
            print(f"  [WARN] Not found: {split_dir}  — skipping")
            continue
        for folder_name, cls_name in PNEUMONIA_MAP.items():
            cls_dir = split_dir / folder_name
            if not cls_dir.exists():
                continue
            images = [p for p in sorted(cls_dir.iterdir())
                      if p.suffix.lower() in IMG_EXTS]
            added = 0
            for img_path in images:
                if counters[cls_name] >= MAX_PER_CLASS:
                    break
                counters[cls_name] += 1
                added += 1
                out_name = f"{cls_name}_{counters[cls_name]:05d}.jpg"
                preprocess_and_save(img_path, UNIFIED_DIR / cls_name, out_name)
            print(f"  Pooled  {split_rel:35s}/{folder_name:12s} → {cls_name}: "
                  f"{added:5d}  (total={counters[cls_name]}  cap={MAX_PER_CLASS})")

    print()
    total = sum(counters.values())
    print(f"  Unified pool total: {total} images across {len(ALL_CLASSES)} classes")
    for cls in ALL_CLASSES:
        pct = counters[cls] / total * 100 if total else 0
        print(f"    {cls:12s}: {counters[cls]:5d}  ({pct:.1f}%)")
    return counters


# ─────────────────────────────────────────────────────────────────────────────
def collect_from_unified() -> tuple:
    """
    STEP 2 — Collect all image paths from data/all_images/ (the unified pool).
    Returns (paths, labels) ready for train_test_split.
    """
    paths, labels = [], []
    for cls in ALL_CLASSES:
        cls_dir = UNIFIED_DIR / cls
        if not cls_dir.exists():
            continue
        found = [p for p in sorted(cls_dir.iterdir())
                 if p.suffix.lower() in IMG_EXTS]
        paths.extend(found)
        labels.extend([cls] * len(found))
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    BANNER = "=" * 65
    print(BANNER)
    print("  FL Project -- Preprocessing (6 classes: brain tumor + pneumonia)")
    print("  Brain: Train_1+Train_2+Test_1+Test_2 | Pneumonia: train+testing")
    print(BANNER)

    if not DATASET_DIR.exists():
        print(f"[ERROR] DataSet/ not found at {DATASET_DIR}")
        sys.exit(1)

    # ── Step 1: Merge all original splits into data/all_images/ ──────────────
    pool_counts = build_unified_pool()
    n_total = sum(pool_counts.values())
    if n_total == 0:
        print("[ERROR] No images found. Check DataSet/ structure.")
        sys.exit(1)

    # ── Step 2: Collect paths from the unified pool ───────────────────────────
    print(f"\n[Step 2] Reading unified pool from data/all_images/ ...")
    all_paths, all_labels = collect_from_unified()
    print(f"  Collected {len(all_paths)} images from data/all_images/")

    # ── Step 3: Carve out global test set ────────────────────────────────────
    print(f"\n[Step 3] Carving out global test set ({int(TEST_RATIO*100)}%) ...")
    train_p, test_p, train_l, test_l = train_test_split(
        all_paths, all_labels,
        test_size=TEST_RATIO,
        stratify=all_labels,
        random_state=RANDOM_STATE,
    )
    print(f"  Training pool : {len(train_p)}")
    print(f"  Global test   : {len(test_p)}")

    # ── Step 4: Partition training pool into client splits ───────────────────
    print(f"\n[Step 4] Partitioning into {NUM_CLIENTS} client splits ...")
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

    # ── Step 5: Write partitions ──────────────────────────────────────────────
    print(f"\n[Step 5] Writing partitions to {PART_DIR} ...")
    print("         (images are already preprocessed in all_images/ — copying)")

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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BANNER}")
    print("  Preprocessing complete -- Partition Summary")
    print(f"  Source  : DataSet/  \u2192  pooled {n_total} images (capped at {MAX_PER_CLASS}/class, 6 classes)")
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
    print(f"\n  data/partitions/ — FL client splits (training-ready)")
    print(f"  DataSet/         — original source images (untouched)")
    print(BANNER)

    # ── Cleanup: remove intermediate all_images/ pool ────────────────────────
    # Partitions are already written — the unified pool is no longer needed.
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
        print(f"\n  Cleaned up intermediate pool: data/all_images/ (removed)")


if __name__ == "__main__":
    main()

