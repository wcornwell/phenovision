# Local flower/fruit inference test

A standalone smoke test for the **flower/fruit (reproductive)** inference path,
runnable on a laptop (Apple Silicon / MPS or CPU) without the cluster GPU nodes
or the parquet metadata the full `_targets_inference.R` pipeline depends on.

It answers one question: *does flower/fruit inference run end-to-end on local
images and produce sensible probabilities?*

## What it reuses from the real pipeline

- **Same model**: `phenobase/phenovision` loaded via
  `transformers.AutoModelForImageClassification` (as in `R/model_loading.R`).
- **Same preprocessing**: the timm ViT-L/16 data transform
  (`resolve_data_config` + `create_transform`).
- **Same output mapping**: `sigmoid(logits)` → `.pred_flower` / `.pred_fruit`,
  with the pipeline's column convention (logit[0]=fruit, logit[1]=flower) used
  as a fallback when the model config doesn't name the labels.

## What it deliberately changes for local use

- **Device**: `cuda:0` → `mps`/`cpu`, and drops `torch.cuda.amp.autocast`
  (CUDA-only) — runs full precision.
- **Image source**: a local folder instead of the cluster parquet dataset.
- **Model revision**: loads the default `main` branch on HF rather than
  resolving the version DOI via `rdatacite` (e.g. v1.1.0 → `10.57967/hf/7952`).
  Pin `revision=` in the script if exact-version parity is needed.
- **Thresholding**: no learned buffer params / family stats (those live on the
  cluster). A simple `Detected` flag is added using the default decision
  thresholds documented in `R/ml_annotation_postprocess.R`
  (flower ≥ 0.84, fruit ≥ 0.53).

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch torchvision timm transformers pillow numpy safetensors huggingface_hub
```

## Get test images (optional)

`run_inat_download.R` downloads iNaturalist images for a set of observations
into `inat_burnt_images/`. It reads its seed list from
`burnt_only_observations.csv` (an iNaturalist/ALA export — **not tracked in
git**; supply your own) and uses `EcoacousticUtilities::get_inat_images`.

```bash
Rscript run_inat_download.R
```

Or point the harness at any folder of `.jpg`/`.jpeg`/`.png` images by editing
`IMAGES_DIR` in the script.

## Run

```bash
.venv/bin/python scripts/test_flowerfruit_local.py
```

## Output

`output/flowerfruit_local_test/predictions.csv` (gitignored — regenerable), one
row per image:

`photo_id, file_name, taxon_name, observed_on, latitude, longitude,
pred_flower, pred_fruit, logit_flower, logit_fruit, flower_detected,
fruit_detected, corrupt`

## Last verified run

825 burnt-area iNaturalist images, 0 corrupt, on an Apple M4 (MPS). Spot-checks
against the photos confirmed correct calls: strong flower, strong fruit, clean
foliage-only negatives, and appropriately uncertain mixed cases.
