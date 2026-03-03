"""
scrape_and_train.py
====================
Step 1 — Parallel scraping
    Spins up one worker thread per (label, query) pair using
    ThreadPoolExecutor.  Each worker downloads images from DuckDuckGo and
    saves them under DataSet/scraped_extra/<label>/.

Step 2 — Preprocessing
    Calls preprocess.py::main() in-process to rebuild the unified pool and
    re-partition data across clients.

Step 3 — Training
    Launches train_sim.py as a subprocess so its argument parser, signal
    handling, and stdout flushing all work correctly end-to-end.

Usage (run from project root):
    python src/scrape_and_train.py                         # default settings
    python src/scrape_and_train.py --images 15             # 15 images per label
    python src/scrape_and_train.py --workers 6             # 6 parallel threads
    python src/scrape_and_train.py --rounds 10 --epochs 5  # training params
    python src/scrape_and_train.py --fresh                 # ignore checkpoint
    python src/scrape_and_train.py --scrape-only           # skip training
    python src/scrape_and_train.py --train-only            # skip scraping
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import requests

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).parent.parent
SCRAPE_DIR = PROJ_ROOT / "DataSet" / "scraped_extra"
SRC_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Queries  —  one entry per (label, query) pair.
# Add or remove rows freely; each row becomes its own worker thread.
# ---------------------------------------------------------------------------
SCRAPE_QUERIES: List[Tuple[str, str]] = [
    ("glioma", "brain MRI glioma tumor scan"),
    ("glioma", "glioma brain cancer radiology"),
    ("meningioma", "meningioma brain MRI scan"),
    ("meningioma", "meningioma radiology imaging"),
    ("notumor", "normal brain MRI no tumor"),
    ("pituitary", "pituitary tumor brain MRI"),
    ("pituitary", "pituitary adenoma MRI radiology"),
    ("normal", "normal chest X-ray healthy lungs"),
    ("pneumonia", "chest X-ray pneumonia infection"),
    ("pneumonia", "bacterial pneumonia chest radiograph"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_print_lock = Lock()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _log(worker: str, msg: str) -> None:
    with _print_lock:
        print(f"[{_ts()}] [{worker}] {msg}", flush=True)


def _banner(title: str) -> None:
    line = "=" * 65
    with _print_lock:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)


# ---------------------------------------------------------------------------
# Worker — runs inside a thread
# ---------------------------------------------------------------------------
def _scrape_worker(
    label: str,
    query: str,
    out_dir: Path,
    max_images: int,
    request_timeout: int,
    worker_id: str,
) -> Dict:
    """
    Download up to *max_images* for a single (label, query) pair.
    Returns a summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0
    skipped = 0

    # Build a filename prefix that encodes the query so files from
    # different workers for the same label don't collide.
    safe_query = query.replace(" ", "_")[:40]
    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Per-worker jitter: stagger requests so all threads don't hammer DDG simultaneously
    jitter = random.uniform(0.5, 3.0)
    time.sleep(jitter)

    _log(
        worker_id,
        f"Starting  |  query='{query}'  max={max_images}  out={out_dir.relative_to(PROJ_ROOT)}",
    )

    try:
        with DDGS() as ddgs:
            results = ddgs.images(
                query, max_results=max_images * 3
            )  # over-fetch to hit target
            for r in results:
                if downloaded >= max_images:
                    break
                url = r.get("image", "")
                if not url:
                    skipped += 1
                    continue
                try:
                    resp = requests.get(url, timeout=request_timeout)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    if "image" not in content_type and not url.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    ):
                        skipped += 1
                        _log(worker_id, f"Skipped (not an image): {url[:60]}")
                        continue

                    ext = ".jpg"
                    if "png" in content_type or url.lower().endswith(".png"):
                        ext = ".png"

                    fname = f"{ts_prefix}_{safe_query}_{downloaded:04d}{ext}"
                    fpath = out_dir / fname
                    fpath.write_bytes(resp.content)
                    downloaded += 1
                    _log(worker_id, f"  [{downloaded:>3}/{max_images}] Saved {fname}")

                except requests.exceptions.Timeout:
                    failed += 1
                    _log(worker_id, f"  Timeout: {url[:60]}")
                except requests.exceptions.RequestException as exc:
                    failed += 1
                    _log(worker_id, f"  Request error: {exc}")
                except OSError as exc:
                    failed += 1
                    _log(worker_id, f"  File write error: {exc}")

    except Exception as exc:
        _log(worker_id, f"  DDGS error: {exc}")

    _log(
        worker_id,
        f"Done  |  downloaded={downloaded}  failed={failed}  skipped={skipped}",
    )
    return {
        "worker_id": worker_id,
        "label": label,
        "query": query,
        "downloaded": downloaded,
        "failed": failed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Phase 1 — parallel scraping
# ---------------------------------------------------------------------------
def run_scraping(max_images: int, max_workers: int, request_timeout: int) -> None:
    _banner("Phase 1 — Parallel Scraping")
    print(f"  Queries     : {len(SCRAPE_QUERIES)}")
    print(f"  Images/query: {max_images}")
    print(f"  Threads     : {max_workers}")
    print("  Output root : DataSet/scraped_extra/")
    print("=" * 65)

    tasks = []
    for idx, (label, query) in enumerate(SCRAPE_QUERIES):
        out_dir = SCRAPE_DIR / label
        worker_id = f"W{idx + 1:02d}:{label[:10]}"
        tasks.append((label, query, out_dir, max_images, request_timeout, worker_id))

    wall_t0 = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scrape_worker, *t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                label, query = futures[fut][0], futures[fut][1]
                _log("MAIN", f"Worker for '{label}'/'{query}' raised: {exc}")

    wall_elapsed = time.perf_counter() - wall_t0

    # --- Summary -------------------------------------------------------
    _banner("Scraping Summary")
    total_dl = total_fail = 0
    by_label: Dict[str, int] = {}
    print(f"  {'Worker':<16} {'Label':<12} {'OK':>4} {'Fail':>5}  Query")
    print("  " + "-" * 62)
    for r in sorted(results, key=lambda x: x["worker_id"]):
        print(
            f"  {r['worker_id']:<16} {r['label']:<12} "
            f"{r['downloaded']:>4} {r['failed']:>5}  {r['query']}"
        )
        total_dl += r["downloaded"]
        total_fail += r["failed"]
        by_label[r["label"]] = by_label.get(r["label"], 0) + r["downloaded"]

    print()
    print(f"  Total downloaded : {total_dl}")
    print(f"  Total failed     : {total_fail}")
    print(f"  Wall-clock time  : {wall_elapsed:.1f}s")
    print()
    print("  Per-label totals:")
    for lbl, cnt in sorted(by_label.items()):
        print(f"    {lbl:<14} {cnt:>4} images  →  DataSet/scraped_extra/{lbl}/")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Phase 2 — preprocessing  (re-partition data)
# ---------------------------------------------------------------------------
def run_preprocessing() -> None:
    _banner("Phase 2 — Preprocessing / Re-partitioning")
    print("  Running src/preprocess.py ...")
    print("=" * 65)

    # Run in a subprocess so stdout streams live and any sys.exit() inside
    # preprocess.py doesn't kill our process.
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "preprocess.py")],
        cwd=str(PROJ_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(
            f"\n[ERROR] preprocess.py exited with code {result.returncode}. "
            "Check output above for details.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(result.returncode)

    print(f"\n[{_ts()}] Preprocessing complete.")


# ---------------------------------------------------------------------------
# Phase 3 — training
# ---------------------------------------------------------------------------
def run_training(
    rounds: int,
    epochs: int,
    batch: int,
    lr: float,
    mu: float,
    fresh: bool,
    do_eval: bool,
    device: str = "mps",
    num_workers: int = 4,
) -> None:
    _banner("Phase 3 — Federated Training  (train_sim.py)")

    cmd = [
        sys.executable,
        str(SRC_DIR / "train_sim.py"),
        "--rounds",
        str(rounds),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--lr",
        str(lr),
        "--mu",
        str(mu),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]
    if fresh:
        cmd.append("--fresh")
    if do_eval:
        cmd.append("--eval")

    print("  Command:", " ".join(cmd))
    print("=" * 65)

    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), check=False)
    if result.returncode != 0:
        print(
            f"\n[ERROR] train_sim.py exited with code {result.returncode}.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tecnomate — multi-worker scrape → preprocess → train pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Scraping options ---------------------------------------------------
    scrape = p.add_argument_group("Scraping")
    scrape.add_argument(
        "--images",
        type=int,
        default=10,
        help="Max images to download per (label, query) worker",
    )
    scrape.add_argument(
        "--workers",
        type=int,
        default=len(SCRAPE_QUERIES),
        help="Number of parallel download threads (default = one per query)",
    )
    scrape.add_argument(
        "--timeout",
        type=int,
        default=8,
        help="HTTP request timeout in seconds",
    )

    # -- Pipeline control ---------------------------------------------------
    ctrl = p.add_argument_group("Pipeline control")
    ctrl.add_argument(
        "--scrape-only",
        action="store_true",
        help="Run scraping only; skip preprocessing and training",
    )
    ctrl.add_argument(
        "--train-only",
        action="store_true",
        help="Skip scraping and preprocessing; run training only",
    )
    ctrl.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step (use existing partitions as-is)",
    )

    # -- Training options (forwarded verbatim to train_sim.py) ---------------
    train = p.add_argument_group("Training  (forwarded to train_sim.py)")
    train.add_argument("--rounds", type=int, default=30, help="Federated rounds")
    train.add_argument(
        "--epochs", type=int, default=2, help="Local epochs per client per round"
    )
    train.add_argument("--batch", type=int, default=64, help="Mini-batch size")
    train.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    train.add_argument(
        "--mu", type=float, default=0.01, help="FedProx proximal coefficient"
    )
    train.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Training device: mps | cuda | cpu  (default: mps for Apple Silicon)",
    )
    train.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes per client",
    )
    train.add_argument(
        "--fresh", action="store_true", help="Ignore checkpoint, start fresh"
    )
    train.add_argument(
        "--eval", action="store_true", help="Evaluate on test set every round"
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.scrape_only and args.train_only:
        print(
            "[ERROR] --scrape-only and --train-only are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    wall_start = time.perf_counter()

    _banner("Tecnomate — Scrape → Preprocess → Train Pipeline")
    print(
        f"  Timestamp  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    print(f"  Project    : {PROJ_ROOT}")
    print(f"  Scrape-only: {args.scrape_only}")
    print(f"  Train-only : {args.train_only}")
    print("=" * 65)

    # ── Phase 1: Scraping ─────────────────────────────────────────────────
    if not args.train_only:
        run_scraping(
            max_images=args.images,
            max_workers=args.workers,
            request_timeout=args.timeout,
        )
    else:
        print(f"[{_ts()}] [Skip] Scraping skipped (--train-only).")

    if args.scrape_only:
        print(f"\n[{_ts()}] --scrape-only set. Stopping after scraping.")
        return

    # ── Phase 2: Preprocessing ────────────────────────────────────────────
    if not args.train_only and not args.skip_preprocess:
        run_preprocessing()
    else:
        reason = "--train-only" if args.train_only else "--skip-preprocess"
        print(f"[{_ts()}] [Skip] Preprocessing skipped ({reason}).")

    # ── Phase 3: Training ─────────────────────────────────────────────────
    run_training(
        rounds=args.rounds,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        mu=args.mu,
        fresh=args.fresh,
        do_eval=args.eval,
        device=args.device,
        num_workers=args.num_workers,
    )

    total = time.perf_counter() - wall_start
    _banner("All Phases Complete")
    print(f"  Total pipeline wall-clock time: {total:.1f}s  ({total / 60:.1f} min)")
    print("=" * 65)


if __name__ == "__main__":
    main()
