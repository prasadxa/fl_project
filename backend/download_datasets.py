"""
Multi-Source Medical Dataset Downloader
========================================
Downloads and scrapes additional training data for all 6 classes:
  Brain Tumor : glioma, meningioma, notumor, pituitary
  Chest X-Ray : normal, pneumonia

Sources (in order of priority):
  1. HuggingFace — Hemg/Brain-Tumor-MRI-Dataset          (~7k brain MRI images)
  2. HuggingFace — hf-vision/chest-xray-pneumonia         (~5.8k chest X-ray images)
  3. HuggingFace — yashshinde0080/chest_xray_images_pneumonia (~11.7k chest X-ray)
  4. DuckDuckGo  — image scraping for all 6 classes        (~50 images/class)
  5. Open-access URLs — direct downloads from public repos

All images are saved into DataSet/ under the correct class folders
so the existing preprocess.py pipeline can absorb them automatically.

Usage:
    python src/download_datasets.py                    # download everything
    python src/download_datasets.py --hf-only          # HuggingFace only
    python src/download_datasets.py --scrape-only      # DuckDuckGo scraping only
    python src/download_datasets.py --max-per-class 500 # cap per class
    python src/download_datasets.py --no-hf            # skip HuggingFace
    python src/download_datasets.py --no-scrape        # skip scraping
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJ_ROOT / "DataSet"

# Target folders inside DataSet/ where new images will land
# (preprocess.py already knows how to pick these up)
HF_BRAIN_EXTRA = DATASET_DIR / "hf_brain_extra"  # new split from HF brain dataset
HF_PNEUMONIA_EXTRA = DATASET_DIR / "hf_pneumonia_extra"  # new split from HF pneumonia
SCRAPED_DIR = DATASET_DIR / "scraped_extra"  # DuckDuckGo scraped images

BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PNEUMONIA_CLASSES = ["normal", "pneumonia"]
ALL_CLASSES = BRAIN_CLASSES + PNEUMONIA_CLASSES

# Mapping from HuggingFace label strings → our class names
HF_BRAIN_LABEL_MAP = {
    # Hemg/Brain-Tumor-MRI-Dataset  (integer labels 0-3)
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
    # string variants
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "notumor",
    "no tumor": "notumor",
    "no_tumor": "notumor",
    "pituitary": "pituitary",
    "glioblastoma": "glioma",  # treat glioblastoma as glioma subtype
    "Glioblastoma": "glioma",
    "Glioma": "glioma",
    "Meningioma": "meningioma",
    "Notumor": "notumor",
    "NoTumor": "notumor",
    "Pituitary": "pituitary",
}

HF_PNEUMONIA_LABEL_MAP = {
    0: "normal",
    1: "pneumonia",
    "NORMAL": "normal",
    "PNEUMONIA": "pneumonia",
    "normal": "normal",
    "pneumonia": "pneumonia",
}

# DuckDuckGo search queries per class
SCRAPE_QUERIES = {
    "glioma": [
        "brain MRI glioma tumor scan",
        "glioma MRI axial slice medical",
        "brain glioma T1 T2 MRI",
    ],
    "meningioma": [
        "brain MRI meningioma tumor scan",
        "meningioma MRI brain scan medical",
        "meningioma contrast MRI",
    ],
    "notumor": [
        "normal brain MRI no tumor",
        "healthy brain MRI scan axial",
        "normal brain T1 MRI",
    ],
    "pituitary": [
        "pituitary tumor brain MRI scan",
        "pituitary adenoma MRI",
        "pituitary gland MRI medical",
    ],
    "normal": [
        "normal chest xray no pneumonia",
        "healthy lungs chest radiograph PA",
        "normal chest X-ray frontal",
    ],
    "pneumonia": [
        "pneumonia chest xray radiograph",
        "lung pneumonia X-ray consolidation",
        "bacterial pneumonia chest X-ray",
    ],
}

# Direct download URLs for open-access public datasets (GitHub / zenodo / etc.)
# These are ZIP/TAR archives or individual image URLs from open repositories
DIRECT_SOURCES = [
    # BrainTumorMRI open dataset on GitHub
    {
        "url": "https://github.com/sartajbhuvaji/brain-tumor-classification-mri/archive/refs/heads/master.zip",
        "type": "zip",
        "label": "brain_github",
        "mapping": {
            "Training/glioma_tumor": "glioma",
            "Training/meningioma_tumor": "meningioma",
            "Training/no_tumor": "notumor",
            "Training/pituitary_tumor": "pituitary",
            "Testing/glioma_tumor": "glioma",
            "Testing/meningioma_tumor": "meningioma",
            "Testing/no_tumor": "notumor",
            "Testing/pituitary_tumor": "pituitary",
        },
        "dest": DATASET_DIR / "github_brain_extra",
    },
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    prefix = {"INFO": "✓", "WARN": "⚠", "ERROR": "✗", "HEAD": "═"}.get(level, "·")
    print(f"  [{prefix}] {msg}", flush=True)


def banner(title: str) -> None:
    line = "═" * 65
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ── HuggingFace downloader ─────────────────────────────────────────────────────
def _pil_to_gray_jpg(pil_img: Image.Image, dest_path: Path) -> bool:
    """Convert PIL image to grayscale JPEG and save."""
    try:
        gray = pil_img.convert("L")
        gray.save(str(dest_path), format="JPEG", quality=90)
        return True
    except Exception as e:
        log(f"Save failed {dest_path.name}: {e}", "WARN")
        return False


def download_hf_brain_dataset(
    max_per_class: int = 2000,
    dest_dir: Optional[Path] = None,
) -> dict[str, int]:
    """
    Download brain tumor MRI images from Hemg/Brain-Tumor-MRI-Dataset on HuggingFace.
    Returns {class_name: count} of downloaded images.
    """
    if dest_dir is None:
        dest_dir = HF_BRAIN_EXTRA

    banner("HuggingFace — Hemg/Brain-Tumor-MRI-Dataset  (~7k images)")

    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets library not found. Install: pip install datasets", "ERROR")
        return {}

    counts = {cls: 0 for cls in BRAIN_CLASSES}

    # Create destination directories
    for cls in BRAIN_CLASSES:
        (dest_dir / cls).mkdir(parents=True, exist_ok=True)

    try:
        log("Loading Hemg/Brain-Tumor-MRI-Dataset from HuggingFace hub …")
        ds = load_dataset(
            "Hemg/Brain-Tumor-MRI-Dataset",
            split="train",
            trust_remote_code=True,
        )
        log(f"Dataset loaded: {len(ds)} rows")

        for i, row in enumerate(ds):
            if i % 500 == 0:
                log(f"  Processing row {i}/{len(ds)} …")

            raw_label = row.get("label", row.get("labels", None))
            cls_name = HF_BRAIN_LABEL_MAP.get(raw_label)
            if cls_name is None:
                continue
            if counts[cls_name] >= max_per_class:
                continue

            img = row.get("image")
            if img is None:
                continue

            idx = counts[cls_name]
            out_path = dest_dir / cls_name / f"hf_hemg_{cls_name}_{idx:05d}.jpg"
            if _pil_to_gray_jpg(img, out_path):
                counts[cls_name] += 1

    except Exception as e:
        log(f"Hemg dataset error: {e}", "ERROR")

    for cls, n in counts.items():
        log(f"  {cls:12s}: {n} images saved → {dest_dir / cls}")

    return counts


def download_hf_chest_xray_dataset(
    max_per_class: int = 3000,
    dest_dir: Optional[Path] = None,
) -> dict[str, int]:
    """
    Download chest X-ray images from hf-vision/chest-xray-pneumonia on HuggingFace.
    Returns {class_name: count} of downloaded images.
    """
    if dest_dir is None:
        dest_dir = HF_PNEUMONIA_EXTRA

    banner("HuggingFace — hf-vision/chest-xray-pneumonia  (~5.8k images)")

    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets library not found. Install: pip install datasets", "ERROR")
        return {}

    counts = {cls: 0 for cls in PNEUMONIA_CLASSES}

    for cls in PNEUMONIA_CLASSES:
        (dest_dir / cls).mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "test", "validation"]:
        try:
            log(f"Loading hf-vision/chest-xray-pneumonia split='{split_name}' …")
            ds = load_dataset(
                "hf-vision/chest-xray-pneumonia",
                split=split_name,
                trust_remote_code=True,
            )
            log(f"  {split_name}: {len(ds)} rows")

            for i, row in enumerate(ds):
                if i % 500 == 0:
                    log(f"    Row {i}/{len(ds)} …")

                raw_label = row.get("label", row.get("labels", None))
                cls_name = HF_PNEUMONIA_LABEL_MAP.get(raw_label)
                if cls_name is None:
                    continue
                if counts[cls_name] >= max_per_class:
                    continue

                img = row.get("image")
                if img is None:
                    continue

                idx = counts[cls_name]
                out_path = (
                    dest_dir
                    / cls_name
                    / f"hf_cxr_{split_name}_{cls_name}_{idx:05d}.jpg"
                )
                if _pil_to_gray_jpg(img, out_path):
                    counts[cls_name] += 1

        except Exception as e:
            log(f"  hf-vision/{split_name} error: {e}", "WARN")

    for cls, n in counts.items():
        log(f"  {cls:12s}: {n} images saved → {dest_dir / cls}")

    return counts


def download_hf_chest_xray_extra(
    max_per_class: int = 2000,
    dest_dir: Optional[Path] = None,
) -> dict[str, int]:
    """
    Download extra chest X-ray images from yashshinde0080/chest_xray_images_pneumonia.
    Returns {class_name: count} of downloaded images.
    """
    if dest_dir is None:
        dest_dir = DATASET_DIR / "hf_pneumonia_extra2"

    banner("HuggingFace — yashshinde0080/chest_xray_images_pneumonia  (~11.7k images)")

    try:
        from datasets import load_dataset
    except ImportError:
        log("datasets library not found.", "ERROR")
        return {}

    counts = {cls: 0 for cls in PNEUMONIA_CLASSES}

    for cls in PNEUMONIA_CLASSES:
        (dest_dir / cls).mkdir(parents=True, exist_ok=True)

    # This dataset uses 'image' column but no explicit label column
    # Images are in subdirs named NORMAL/PNEUMONIA
    for split_name in ["train", "test"]:
        try:
            log(
                f"Loading yashshinde0080/chest_xray_images_pneumonia split='{split_name}' …"
            )
            ds = load_dataset(
                "yashshinde0080/chest_xray_images_pneumonia",
                split=split_name,
                trust_remote_code=True,
            )
            log(f"  {split_name}: {len(ds)} rows, features: {list(ds.features.keys())}")

            # Try to determine label from available columns
            label_col = None
            for col in ["label", "labels", "category", "class", "target"]:
                if col in ds.features:
                    label_col = col
                    break

            for i, row in enumerate(ds):
                if i % 500 == 0:
                    log(f"    Row {i}/{len(ds)} …")

                raw_label = row.get(label_col) if label_col else None
                cls_name = HF_PNEUMONIA_LABEL_MAP.get(raw_label)
                if cls_name is None:
                    # Try to infer from image filename metadata
                    img_obj = row.get("image")
                    if img_obj is not None and hasattr(img_obj, "filename"):
                        fname = str(img_obj.filename).upper()
                        if "NORMAL" in fname:
                            cls_name = "normal"
                        elif (
                            "PNEUMONIA" in fname
                            or "BACTERIA" in fname
                            or "VIRUS" in fname
                        ):
                            cls_name = "pneumonia"
                if cls_name is None:
                    continue
                if counts[cls_name] >= max_per_class:
                    continue

                img = row.get("image")
                if img is None:
                    continue

                idx = counts[cls_name]
                out_path = (
                    dest_dir
                    / cls_name
                    / f"hf_cxr2_{split_name}_{cls_name}_{idx:05d}.jpg"
                )
                if _pil_to_gray_jpg(img, out_path):
                    counts[cls_name] += 1

        except Exception as e:
            log(f"  yashshinde/{split_name} error: {e}", "WARN")

    for cls, n in counts.items():
        log(f"  {cls:12s}: {n} images saved → {dest_dir / cls}")

    return counts


# ── DuckDuckGo Scraper ─────────────────────────────────────────────────────────
def _fetch_image(url: str, timeout: int = 8) -> Optional[Image.Image]:
    """Download image from URL and return as PIL Image, or None on failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        return img
    except Exception:
        return None


def scrape_duckduckgo(
    max_per_class: int = 80,
    dest_dir: Optional[Path] = None,
    delay: float = 0.5,
) -> dict[str, int]:
    """
    Scrape medical images from DuckDuckGo image search for all 6 classes.
    Returns {class_name: count} of saved images.
    """
    if dest_dir is None:
        dest_dir = SCRAPED_DIR

    banner("DuckDuckGo Image Scraping — All 6 Classes")

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log(
            "duckduckgo_search not found. Install: pip install duckduckgo-search",
            "ERROR",
        )
        return {}

    counts = {cls: 0 for cls in ALL_CLASSES}

    for cls in ALL_CLASSES:
        (dest_dir / cls).mkdir(parents=True, exist_ok=True)

    for cls_name, queries in SCRAPE_QUERIES.items():
        log(f"\n  Scraping class: {cls_name}")
        cls_dir = dest_dir / cls_name
        per_query = max(1, max_per_class // len(queries))

        for query in queries:
            if counts[cls_name] >= max_per_class:
                break

            log(f"    Query: '{query}'  (want ~{per_query} images)")
            try:
                with DDGS() as ddgs:
                    results = list(
                        ddgs.images(
                            query,
                            max_results=per_query
                            + 10,  # overfetch to account for failures
                            type_image="photo",
                            size="Medium",
                            safesearch="off",
                        )
                    )
            except Exception as e:
                log(f"    DDG search error: {e}", "WARN")
                results = []

            saved_this_query = 0
            for r in results:
                if counts[cls_name] >= max_per_class:
                    break
                img_url = r.get("image", "")
                if not img_url:
                    continue

                img = _fetch_image(img_url)
                if img is None:
                    continue

                # Basic quality checks
                w, h = img.size
                if w < 64 or h < 64:
                    continue  # too small
                if w > 4096 or h > 4096:
                    continue  # unreasonably large

                idx = counts[cls_name]
                out_path = cls_dir / f"ddg_{cls_name}_{idx:05d}.jpg"
                try:
                    gray = img.convert("L")
                    gray.save(str(out_path), format="JPEG", quality=85)
                    counts[cls_name] += 1
                    saved_this_query += 1
                except Exception:
                    pass

                time.sleep(delay)

            log(f"    Saved {saved_this_query} images (total {counts[cls_name]})")
            time.sleep(delay * 2)  # be polite between queries

    log("")
    for cls, n in counts.items():
        log(f"  {cls:12s}: {n} images scraped → {dest_dir / cls}")

    return counts


# ── GitHub / Direct URL Downloader ────────────────────────────────────────────
def download_github_brain_dataset(
    max_per_class: int = 1500,
) -> dict[str, int]:
    """
    Download the sartajbhuvaji brain tumor dataset from GitHub (open MIT licence).
    Extracts ZIP and saves images into DataSet/github_brain_extra/.
    Returns {class_name: count}.
    """
    import tempfile
    import zipfile

    banner("GitHub — sartajbhuvaji/brain-tumor-classification-mri")

    source = DIRECT_SOURCES[0]
    url = source["url"]
    mapping = source["mapping"]
    dest = source["dest"]

    counts = {cls: 0 for cls in BRAIN_CLASSES}
    for cls in BRAIN_CLASSES:
        (dest / cls).mkdir(parents=True, exist_ok=True)

    log(f"Downloading ZIP from GitHub …")
    log(f"URL: {url}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "brain_github.zip"

            # Stream download with progress
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(
                            f"\r    {pct:5.1f}%  ({downloaded // 1024 // 1024} MB)",
                            end="",
                            flush=True,
                        )
            print()

            log(f"Downloaded {downloaded // 1024 // 1024} MB — extracting …")

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                log(f"ZIP contains {len(members)} entries")

                for member in members:
                    suffix = Path(member).suffix.lower()
                    if suffix not in IMG_EXTENSIONS:
                        continue

                    # Match against our folder mapping
                    cls_name = None
                    for folder_key, mapped_cls in mapping.items():
                        # Normalize: replace backslashes, lower
                        norm_member = member.replace("\\", "/").lower()
                        norm_key = folder_key.replace("\\", "/").lower()
                        if f"/{norm_key}/" in norm_member or norm_member.endswith(
                            f"/{norm_key}/" + Path(member).name.lower()
                        ):
                            cls_name = mapped_cls
                            break
                        # Also try without leading dir (archive root varies)
                        parts = norm_member.split("/")
                        for j in range(len(parts) - 1):
                            partial = "/".join(parts[j:])
                            if partial.startswith(norm_key + "/"):
                                cls_name = mapped_cls
                                break
                        if cls_name:
                            break

                    if cls_name is None:
                        continue
                    if counts[cls_name] >= max_per_class:
                        continue

                    idx = counts[cls_name]
                    out_path = dest / cls_name / f"gh_{cls_name}_{idx:05d}{suffix}"

                    try:
                        img_data = zf.read(member)
                        img = Image.open(io.BytesIO(img_data)).convert("L")
                        img.save(str(out_path), format="JPEG", quality=90)
                        counts[cls_name] += 1
                    except Exception:
                        pass

    except Exception as e:
        log(f"GitHub download failed: {e}", "ERROR")
        log("This is non-fatal — continuing with other sources.", "WARN")

    for cls, n in counts.items():
        log(f"  {cls:12s}: {n} images saved → {dest / cls}")

    return counts


# ── Update preprocess.py to include new folders ───────────────────────────────
def patch_preprocess_for_new_sources() -> None:
    """
    Dynamically registers new DataSet sub-folders into preprocess.py's SOURCE_SPLITS
    and PNEUMONIA_SPLITS so they are automatically picked up on next preprocess run.
    We do this by writing a config JSON that preprocess.py can optionally read,
    OR simply by printing the folders the user should add.
    """
    banner("Integration — New DataSet Folders")

    brain_dirs = []
    pneumonia_dirs = []

    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir():
            continue
        subdirs = [s.name for s in d.iterdir() if s.is_dir()]
        has_brain = any(c in subdirs for c in BRAIN_CLASSES)
        has_pneumonia = any(
            c in ["normal", "pneumonia", "NORMAL", "PNEUMONIA"] for c in subdirs
        )

        if has_brain:
            brain_dirs.append(d.name)
        if has_pneumonia:
            pneumonia_dirs.append(d.name)

    log("New brain tumor folders detected:")
    for d in brain_dirs:
        log(f"    DataSet/{d}")
    log("New pneumonia / chest X-ray folders detected:")
    for d in pneumonia_dirs:
        log(f"    DataSet/{d}")

    # Write a manifest file that the updated preprocess.py can read
    manifest_path = PROJ_ROOT / "data" / "extra_datasets_manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        f.write("# Auto-generated by download_datasets.py\n")
        f.write("# brain_dirs — each sub-folder must contain class-named sub-dirs\n")
        for d in brain_dirs:
            f.write(f"brain:{d}\n")
        f.write(
            "# pneumonia_dirs — sub-folders must contain NORMAL/ and/or PNEUMONIA/\n"
        )
        for d in pneumonia_dirs:
            f.write(f"pneumonia:{d}\n")

    log(f"Manifest written → {manifest_path}")


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(all_counts: dict[str, dict[str, int]]) -> None:
    banner("Download Summary")

    combined = {cls: 0 for cls in ALL_CLASSES}

    # Header
    col_w = 14
    sources = list(all_counts.keys())
    header = (
        f"  {'Class':<14}" + "".join(f"{s:>12}" for s in sources) + f"{'TOTAL':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cls in ALL_CLASSES:
        row = f"  {cls:<14}"
        row_total = 0
        for src in sources:
            n = all_counts[src].get(cls, 0)
            row += f"{n:>12}"
            row_total += n
            combined[cls] += n
        row += f"{row_total:>10}"
        print(row)

    print("  " + "=" * (len(header) - 2))
    total_row = f"  {'TOTAL':<14}"
    grand = 0
    for src in sources:
        s = sum(all_counts[src].values())
        total_row += f"{s:>12}"
        grand += s
    total_row += f"{grand:>10}"
    print(total_row)
    print()
    log(f"Grand total new images downloaded: {grand}")
    log("")
    log("Next step — run preprocessing to absorb new data:")
    log("    python src/preprocess.py")
    log("")
    log("Then retrain with:")
    log("    python src/train_sim.py --rounds 30 --epochs 2 --fresh")


# ── Preprocess integration: patch SOURCE_SPLITS in preprocess.py ──────────────
def auto_patch_preprocess() -> None:
    """
    Read preprocess.py and insert new DataSet sub-folders into SOURCE_SPLITS
    and PNEUMONIA_SPLITS if they are not already there.
    """
    preprocess_path = PROJ_ROOT / "src" / "preprocess.py"
    if not preprocess_path.exists():
        log("preprocess.py not found — skipping auto-patch", "WARN")
        return

    content = preprocess_path.read_text(encoding="utf-8")

    new_brain_dirs = []
    new_pneumo_dirs = []

    for d in sorted(DATASET_DIR.iterdir()):
        if not d.is_dir():
            continue
        # Skip original dirs that are already in preprocess.py
        if d.name in (
            "brain_tumer_train",
            "brain_tumer_testing",
            "pneumonia_train",
            "pneumonia_testing",
        ):
            continue

        subdirs = [s.name for s in d.iterdir() if s.is_dir()]
        has_brain = any(c in subdirs for c in BRAIN_CLASSES)
        has_pneumonia = any(
            c.upper() in ["NORMAL", "PNEUMONIA"] or c in ["normal", "pneumonia"]
            for c in subdirs
        )

        if has_brain and d.name not in content:
            new_brain_dirs.append(d.name)
        if has_pneumonia and d.name not in content:
            new_pneumo_dirs.append(d.name)

    if not new_brain_dirs and not new_pneumo_dirs:
        log("preprocess.py already up to date — no patching needed")
        return

    banner("Auto-patching preprocess.py")

    # Patch SOURCE_SPLITS for brain dirs
    for d in new_brain_dirs:
        old_marker = '    "brain_tumer_testing/Test_2",'
        new_entry = f'    "{d}",'
        if new_entry not in content:
            content = content.replace(
                old_marker,
                f"{old_marker}\n    # Extra brain MRI from download_datasets.py\n{new_entry}",
            )
            log(f"  Added brain dir: {d}")

    # Patch PNEUMONIA_SPLITS for pneumonia dirs
    for d in new_pneumo_dirs:
        old_marker = '    "pneumonia_testing",'
        new_entry = f'    "{d}",'
        if new_entry not in content:
            content = content.replace(
                old_marker,
                f"{old_marker}\n    # Extra chest X-ray from download_datasets.py\n{new_entry}",
            )
            log(f"  Added pneumonia dir: {d}")

    preprocess_path.write_text(content, encoding="utf-8")
    log(f"preprocess.py patched successfully → {preprocess_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download & scrape additional medical imaging datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hf-only", action="store_true", help="Only run HuggingFace downloads"
    )
    p.add_argument(
        "--scrape-only", action="store_true", help="Only run DuckDuckGo scraping"
    )
    p.add_argument(
        "--github-only", action="store_true", help="Only run GitHub download"
    )
    p.add_argument("--no-hf", action="store_true", help="Skip HuggingFace downloads")
    p.add_argument("--no-scrape", action="store_true", help="Skip DuckDuckGo scraping")
    p.add_argument("--no-github", action="store_true", help="Skip GitHub download")
    p.add_argument(
        "--max-per-class",
        type=int,
        default=1500,
        help="Maximum images to download per class per source",
    )
    p.add_argument(
        "--scrape-max",
        type=int,
        default=80,
        help="Maximum images to scrape per class from DuckDuckGo",
    )
    p.add_argument(
        "--scrape-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between DuckDuckGo image fetches",
    )
    p.add_argument(
        "--skip-preprocess-patch",
        action="store_true",
        help="Do not auto-patch preprocess.py",
    )
    p.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Automatically run preprocess.py after downloading",
    )
    p.add_argument(
        "--run-train",
        action="store_true",
        help="Automatically run train_sim.py after preprocessing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    banner("Tecnomate FL — Multi-Source Medical Dataset Downloader")
    log(f"Project root : {PROJ_ROOT}")
    log(f"Dataset dir  : {DATASET_DIR}")
    log(f"Max per class: {args.max_per_class}")

    all_counts: dict[str, dict[str, int]] = {}

    run_hf = not args.no_hf and not args.scrape_only and not args.github_only
    run_scrape = not args.no_scrape and not args.hf_only and not args.github_only
    run_github = not args.no_github and not args.hf_only and not args.scrape_only

    # ── 1. HuggingFace ────────────────────────────────────────────────────────
    if run_hf:
        log("\nStarting HuggingFace downloads …")

        counts_hf_brain = download_hf_brain_dataset(
            max_per_class=args.max_per_class,
            dest_dir=HF_BRAIN_EXTRA,
        )
        all_counts["HF-Brain"] = counts_hf_brain

        counts_hf_cxr = download_hf_chest_xray_dataset(
            max_per_class=args.max_per_class,
            dest_dir=HF_PNEUMONIA_EXTRA,
        )
        all_counts["HF-CXR"] = counts_hf_cxr

        counts_hf_cxr2 = download_hf_chest_xray_extra(
            max_per_class=args.max_per_class,
            dest_dir=DATASET_DIR / "hf_pneumonia_extra2",
        )
        all_counts["HF-CXR2"] = counts_hf_cxr2

    # ── 2. GitHub / Direct download ───────────────────────────────────────────
    if run_github:
        log("\nStarting GitHub download …")
        counts_gh = download_github_brain_dataset(
            max_per_class=args.max_per_class,
        )
        all_counts["GitHub"] = counts_gh

    # ── 3. DuckDuckGo Scraping ────────────────────────────────────────────────
    if run_scrape:
        log("\nStarting DuckDuckGo image scraping …")
        counts_ddg = scrape_duckduckgo(
            max_per_class=args.scrape_max,
            dest_dir=SCRAPED_DIR,
            delay=args.scrape_delay,
        )
        all_counts["DDG"] = counts_ddg

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_counts:
        print_summary(all_counts)
    else:
        log("No download sources were run. Use --help to see options.", "WARN")
        sys.exit(0)

    # ── Auto-patch preprocess.py ──────────────────────────────────────────────
    if not args.skip_preprocess_patch:
        auto_patch_preprocess()
        patch_preprocess_for_new_sources()

    # ── Optionally run preprocess ─────────────────────────────────────────────
    if args.run_preprocess:
        banner("Running preprocess.py")
        import subprocess

        ret = subprocess.run(
            [sys.executable, str(PROJ_ROOT / "src" / "preprocess.py")],
            cwd=str(PROJ_ROOT),
        )
        if ret.returncode != 0:
            log("preprocess.py exited with errors", "ERROR")
            sys.exit(1)

        # ── Optionally run train ──────────────────────────────────────────────
        if args.run_train:
            banner("Running train_sim.py")
            ret = subprocess.run(
                [
                    sys.executable,
                    str(PROJ_ROOT / "src" / "train_sim.py"),
                    "--rounds",
                    "30",
                    "--epochs",
                    "2",
                    "--fresh",
                    "--eval",
                ],
                cwd=str(PROJ_ROOT),
            )
            if ret.returncode != 0:
                log("train_sim.py exited with errors", "ERROR")
                sys.exit(1)

    banner("All done!")


if __name__ == "__main__":
    main()
