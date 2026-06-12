#!/usr/bin/env python
"""
Standalone local test of the flower/fruit (reproductive) inference path.

This mirrors the core of the targets `annotations_repro` branch
(R/annotate_batch_inference.R) but runs on Apple Silicon (MPS) or CPU instead
of CUDA, and reads images straight from a local folder instead of the cluster
parquet metadata. It exists to answer one question: does flower/fruit inference
run end-to-end on local images and produce sensible probabilities?

What it reuses faithfully from the pipeline:
  - The same model: phenobase/phenovision (transformers AutoModelForImageClassification)
  - The same preprocessing: timm ViT-L/16 data transform (resolve_data_config + create_transform)
  - The same output mapping: sigmoid(logits) -> .pred_flower / .pred_fruit

What it deliberately changes for local use:
  - Device cuda:0 -> mps/cpu, and drops torch.cuda.amp.autocast (CUDA-only)
  - Image source: local folder, not cluster parquet (no thresholds/family stats,
    which require cluster metadata). A "Detected" flag is added using the default
    decision thresholds documented in R/ml_annotation_postprocess.R.

Usage:
  .venv/bin/python scripts/test_flowerfruit_local.py
"""

import csv
import glob
import os
import re
import sys

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Config -----------------------------------------------------------------
HF_REPO = "phenobase/phenovision"           # reproductive (flower/fruit) model
IMAGES_DIR = "inat_burnt_images/images"
METADATA_CSV = "inat_burnt_images/metadata.csv"
OUT_CSV = "output/flowerfruit_local_test/predictions.csv"
BATCH_SIZE = 16

# Default decision thresholds from R/ml_annotation_postprocess.R
# (middle value of each threshold triple). These are NOT the learned v1.1.0
# buffer params (which live on the cluster); they only make the output readable.
FLOWER_DECISION = 0.84
FRUIT_DECISION = 0.53


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_metadata(path):
    """photo_id -> metadata row (taxon_name, observed_on, lat/lon)."""
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            meta[str(row.get("photo_id", "")).strip()] = row
    return meta


def photo_id_from_name(fname):
    """Boronia_algida_319322334_577212490.jpg -> 577212490 (last numeric token)."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def build_transform():
    """Replicate the pipeline's timm ViT-L/16 inference transform."""
    import timm
    m = timm.create_model("vit_large_patch16_224", num_classes=2, pretrained=False)
    cfg = timm.data.resolve_data_config({}, model=m)
    return timm.data.create_transform(**cfg, is_training=False)


def label_order(model):
    """
    Return indices (idx_flower, idx_fruit) into the model's 2-logit output.

    Prefers the model's id2label if it names flower/fruit; otherwise falls back
    to the pipeline's convention in R/annotate_batch_inference.R:
    column V1 = fruit, V2 = flower.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    idx_flower = idx_fruit = None
    for i, name in labels.items():
        if "flower" in name:
            idx_flower = i
        if "fruit" in name:
            idx_fruit = i
    if idx_flower is not None and idx_fruit is not None:
        return idx_flower, idx_fruit, f"id2label {labels}"
    # Pipeline fallback: V1=fruit, V2=flower
    return 1, 0, "pipeline default (V1=fruit, V2=flower)"


def main():
    device = pick_device()
    print(f"Device: {device}")

    from transformers import AutoModelForImageClassification

    print(f"Loading model {HF_REPO} (first run downloads ~1.2 GB)...")
    try:
        model = AutoModelForImageClassification.from_pretrained(HF_REPO)
    except Exception as e:  # noqa: BLE001 - some custom repos need remote code
        print(f"  plain load failed ({e}); retrying with trust_remote_code=True")
        model = AutoModelForImageClassification.from_pretrained(
            HF_REPO, trust_remote_code=True
        )
    model.eval().to(device)

    n_labels = int(getattr(model.config, "num_labels", -1))
    print(f"num_labels: {n_labels}")
    if n_labels != 2:
        sys.exit(
            f"Expected 2 labels for flower/fruit, got {n_labels}. Wrong model?"
        )

    idx_flower, idx_fruit, how = label_order(model)
    print(f"Label mapping: flower=logit[{idx_flower}], fruit=logit[{idx_fruit}] via {how}")

    transform = build_transform()
    meta = load_metadata(METADATA_CSV)

    files = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
        + glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    )
    print(f"Found {len(files)} images in {IMAGES_DIR}")
    if not files:
        sys.exit(f"No images found in {IMAGES_DIR}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    n_corrupt = 0
    sig = torch.nn.Sigmoid()

    for start in range(0, len(files), BATCH_SIZE):
        chunk = files[start : start + BATCH_SIZE]
        tensors, valid = [], []
        for fpath in chunk:
            try:
                img = Image.open(fpath).convert("RGB")
                tensors.append(transform(img))
                valid.append((fpath, True))
            except Exception as e:  # noqa: BLE001
                print(f"  corrupt/unreadable: {fpath} ({e})")
                valid.append((fpath, False))
                n_corrupt += 1

        if tensors:
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = model(batch)
            logits = out.logits if hasattr(out, "logits") else out[0]
            logits = logits.float().cpu()
            probs = sig(logits)
        else:
            logits = probs = None

        ti = 0
        for fpath, ok in valid:
            pid = photo_id_from_name(fpath)
            m = meta.get(pid, {})
            base = {
                "photo_id": pid,
                "file_name": os.path.basename(fpath),
                "taxon_name": m.get("taxon_name", ""),
                "observed_on": m.get("observed_on", ""),
                "latitude": m.get("latitude", ""),
                "longitude": m.get("longitude", ""),
                "corrupt": not ok,
            }
            if ok:
                pf = float(probs[ti, idx_flower])
                pr = float(probs[ti, idx_fruit])
                base.update(
                    pred_flower=round(pf, 6),
                    pred_fruit=round(pr, 6),
                    logit_flower=round(float(logits[ti, idx_flower]), 6),
                    logit_fruit=round(float(logits[ti, idx_fruit]), 6),
                    flower_detected=pf >= FLOWER_DECISION,
                    fruit_detected=pr >= FRUIT_DECISION,
                )
                ti += 1
            else:
                base.update(
                    pred_flower="", pred_fruit="",
                    logit_flower="", logit_fruit="",
                    flower_detected="", fruit_detected="",
                )
            rows.append(base)
        print(f"  batch {start // BATCH_SIZE + 1}: {len(chunk)} images done")

    fieldnames = [
        "photo_id", "file_name", "taxon_name", "observed_on",
        "latitude", "longitude", "pred_flower", "pred_fruit",
        "logit_flower", "logit_fruit", "flower_detected", "fruit_detected",
        "corrupt",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if not r["corrupt"]]
    n_fl = sum(1 for r in ok_rows if r["flower_detected"] is True)
    n_fr = sum(1 for r in ok_rows if r["fruit_detected"] is True)
    print("\n=== Summary ===")
    print(f"images processed : {len(rows)}")
    print(f"corrupt/skipped  : {n_corrupt}")
    print(f"flower detected  : {n_fl}/{len(ok_rows)} (>= {FLOWER_DECISION})")
    print(f"fruit detected   : {n_fr}/{len(ok_rows)} (>= {FRUIT_DECISION})")
    print(f"output CSV       : {OUT_CSV}")


if __name__ == "__main__":
    main()
