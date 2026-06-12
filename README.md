# PhenoVision
[![DOI](https://zenodo.org/badge/660198130.svg)](https://doi.org/10.5281/zenodo.15182888)

Code for PhenoVision, a model for classifying phenology from plant images using computer vision models.

> **Note:** This is a fork of [Phenobase/phenovision](https://github.com/Phenobase/phenovision)
> maintained for local experimentation. Its purpose is the local flower/fruit
> inference test described below; changes here are not intended to merge back upstream.

## Local flower/fruit inference test

**Goal:** run the flower/fruit (reproductive) inference path on a laptop —
Apple Silicon (MPS) or CPU — against an arbitrary folder of images, *without*
the cluster GPU nodes or parquet metadata that the production
`_targets_inference.R` pipeline requires. It answers one question: does
flower/fruit inference run end-to-end on local images and produce sensible
probabilities? (Validated on burnt-area iNaturalist images.)

**What this fork adds:**

- `scripts/test_flowerfruit_local.py` — standalone harness. Loads
  `phenobase/phenovision` from Hugging Face, applies the pipeline's timm
  ViT-L/16 transform, runs a local image folder, and writes per-image
  `pred_flower`/`pred_fruit` (plus logits and default-threshold detection flags)
  to `output/flowerfruit_local_test/predictions.csv`.
- `run_inat_download.R` — fetches the iNaturalist test images.
- `docs/local_flowerfruit_test.md` — full setup and usage notes.
- `.gitignore` — excludes the local venv and test data.

**How it differs from the production pipeline** (changes confined to the local
harness, not the pipeline code):

- Device `cuda:0` → `mps`/`cpu`, and drops the CUDA-only
  `torch.cuda.amp.autocast` (runs full precision).
- Reads images from a local folder instead of the cluster parquet dataset.
- Loads the model's default `main` revision rather than resolving the version
  DOI via `rdatacite`.
- Adds a `Detected` flag from the default decision thresholds documented in
  `R/ml_annotation_postprocess.R` (flower ≥ 0.84, fruit ≥ 0.53), since the
  learned buffer params and family stats live on the cluster.

**Quick start:**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch torchvision timm transformers pillow numpy safetensors huggingface_hub
.venv/bin/python scripts/test_flowerfruit_local.py
```

See [`docs/local_flowerfruit_test.md`](docs/local_flowerfruit_test.md) for details.
