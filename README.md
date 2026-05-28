# BionicVision

This repository reproduces the Han et al. 2021 scene-simplification baseline on
EPIC-KITCHENS clips, evaluates those outputs against VISOR annotations, and adds
`combination1`: the same Han fusion scheme with temporal model inputs
(DeepGaze III, TCMonoDepth, and manual VISOR-guided DEVA).

Baseline reproduction is mandatory for this project. Run and evaluate the Han
baseline before comparing `combination1`; the comparison is only meaningful when
both outputs are produced from the same clips, frame rate, environment, and
evaluation code.

The full setup, data manifest, model weight locations, and run commands are in
[docs/SETUP.md](docs/SETUP.md).

Quick command map:

```bash
source .venv-models/bin/activate

# Mandatory baseline reproduction.
PYTHONPATH=src python -m pipelines.han_baseline \
  --clip-dir data/epic_kitchens/video_snippets/first_10s \
  --output-root outputs/han_baseline_data_first10 \
  --target-fps 20 \
  --device cuda

PYTHONPATH=src python scripts/evaluate_han_baseline.py \
  --data-root data/epic_kitchens \
  --output-root outputs/han_baseline_data_first10 \
  --results-dir outputs/evaluation/han_baseline_data_first10

# Combination1 after the baseline exists.
PYTHONPATH=src python scripts/run_combination1.py \
  --clip-dir data/epic_kitchens/video_snippets/first_10s \
  --output-root outputs/combination1_epic10 \
  --data-root data/epic_kitchens \
  --target-fps 20 \
  --device cuda

PYTHONPATH=src python scripts/evaluate_han_baseline.py \
  --data-root data/epic_kitchens \
  --output-root outputs/combination1_epic10 \
  --results-dir outputs/evaluation/combination1_epic10
```
