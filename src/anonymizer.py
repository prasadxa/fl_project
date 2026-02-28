"""
Privacy Filter - anonymizer.py
================================
Strips ALL EXIF data, metadata, and hidden tags from uploaded medical images
before they are saved to the local dataset.

This is a mandatory gatekeeper in the Tecnomate edge-collection pipeline.
No patient metadata (timestamps, device info, GPS, etc.) is ever retained.

Usage:
    from anonymizer import clean_image
    clean_image("uploads/scan.jpg", "data/new_collected_data/pneumonia/scan.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image


def clean_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
) -> bool:
    """
    Open an image, strip ALL metadata/EXIF, and save a clean copy.

    Parameters
    ----------
    image_path  : path to the uploaded source image (JPEG or PNG)
    output_path : destination path for the anonymised, metadata-free image

    Returns
    -------
    True  if the image was saved successfully
    False if an unrecoverable error occurred (logged to stdout)

    Privacy guarantee
    -----------------
    Pillow rebuilds the image from raw pixel data only.  Any EXIF block,
    XMP sidecar, IPTC record, ICC profile, thumbnail, GPS tags, or device
    fingerprint that was embedded in the original file is silently discarded.
    """
    image_path  = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        print(f"[Anonymizer] ERROR - source file not found: {image_path}")
        return False

    try:
        # Open and decode pixel data
        with Image.open(image_path) as img:
            # Convert to RGB (handles palette / RGBA / greyscale JPEG edge-cases)
            mode = img.mode
            if mode not in ("L", "RGB"):
                img = img.convert("RGB")
                mode = "RGB"

            # Rebuild from raw pixel array — all metadata paths are severed
            clean = Image.fromarray(__import__("numpy").array(img), mode=mode)

        # Ensure destination directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Infer save format from extension
        suffix = output_path.suffix.lower()
        fmt    = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"

        # Save with no extra info dict (no EXIF, no comment blocks)
        clean.save(str(output_path), format=fmt)

        print(f"[Anonymizer] Clean image saved -> {output_path.name}  "
              f"(metadata stripped, format={fmt})")
        return True

    except OSError as exc:
        # Covers truncated files, unsupported formats, disk errors
        print(f"[Anonymizer] ERROR reading image '{image_path.name}': {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[Anonymizer] Unexpected error for '{image_path.name}': {exc}")
        return False




def strip_metadata_and_save(raw_image: Image.Image, save_path: Union[str, Path]) -> None:
    """
    Accept a PIL Image object, strip all metadata/EXIF, and save to save_path.

    This is the privacy-gateway called by the Streamlit frontend when a doctor
    confirms or overrides a diagnosis.

    Parameters
    ----------
    raw_image : PIL.Image.Image  — the image as opened from the uploader
    save_path : str | Path       — full destination path (file will be JPEG)

    Privacy guarantee
    -----------------
    Image is rebuilt from raw pixel data only; all EXIF, XMP, IPTC, ICC
    profiles, GPS tags, and thumbnail blocks are discarded.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild from raw pixel array — severs all metadata channels
    import numpy as _np
    mode  = raw_image.mode if raw_image.mode in ("L", "RGB") else "RGB"
    img   = raw_image.convert(mode)
    clean = Image.fromarray(_np.array(img), mode)

    suffix = save_path.suffix.lower()
    fmt    = "JPEG" if suffix in (".jpg", ".jpeg") else "PNG"
    clean.save(str(save_path), format=fmt)
    print(f"[Anonymizer] strip_metadata_and_save -> {save_path.name}  (format={fmt})")
