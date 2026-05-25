# Poster Evaluation Metrics

This project uses a compact metric set for an EPIC-KITCHENS/VISOR poster
evaluation. The goal is to test the whole simplification pipeline without
turning the poster into a general segmentation benchmark.

## Evaluation Setup

For each evaluated clip/window:

- **Foreground**: hands plus the currently active/contacted object.
- **Background interactable objects**: visible objects that are interacted with
  somewhere in the clip/window, but are not currently foreground.
- **Pipeline output**: grayscale simplified video and, when available, the
  corresponding pulse2percept SPV video.

Object representation must be compatible with both filled-mask and outline
rendering. A ground-truth object is counted as represented if the simplified
output overlaps either the object mask or a dilated boundary band by at least a
small fixed threshold.

## Supervised Metrics

These use VISOR hand/object masks, active/contact object annotations, and object
tracks.

### 1. Foreground Recall vs SPV Load

Measures whether the critical current interaction remains visible.

- Ground truth: hand masks plus currently active/contacted object masks.
- Prediction: represented foreground objects in the simplified output.
- Y-axis: foreground object recall.
- X-axis: SPV perceptual load.
- Summary: AUC under a fixed load cap, e.g. `AUC@load<=L`.

This is the main task-preservation metric.

### 2. Background Interactable Recall vs SPV Load

Measures useful context preservation under a perceptual-load budget.

- Ground truth: visible objects that are interacted with at some point in the
  clip/window but are not currently foreground.
- Prediction: represented background interactable objects.
- Y-axis: background interactable object recall.
- X-axis: SPV perceptual load.
- Summary: AUC under a fixed load cap, e.g. `AUC@load<=L`.

This should be interpreted as a tradeoff curve, not as "higher recall is always
better" independent of load.

### 3. Track Dropout Rate

Measures object-level temporal consistency.

For each foreground or background-interactable object track, count cases where
the ground-truth object remains visible but the simplified output representation
disappears and later reappears.

Lower is better.

## Unsupervised Metric

### 4. Flow-Compensated Flicker

Measures whole-output temporal instability after accounting for camera/user
motion.

Procedure:

1. Estimate optical flow between RGB frame `t` and frame `t+1`.
2. Warp simplified output `S_t` into frame `t+1`.
3. Compare warped `S_t` with `S_{t+1}` using mean absolute difference.
4. Average over the clip.

Lower is better.

## SPV Perceptual Load

SPV load is the x-axis/budget variable for the recall curves, not a standalone
success metric.

Recommended definition:

```text
SPV load = mean normalized intensity of the simulated percept
```

For simulated percept frames `P_t(x, y) in [0, 1]`:

```text
load = (1 / TWH) * sum_t sum_x sum_y P_t(x, y)
```

An optional alternate diagnostic is active percept area:

```text
active area = mean(1[P_t(x, y) > threshold])
```

Use a fixed video-level normalization, not per-frame normalization.

## Tiny User Study

Use a lightweight 2AFC preference task as perceptual validation.

Stimulus:

- Short SPV clips, approximately 2-4 seconds.
- Baseline and proposed method, randomized order.

Prompt:

```text
Which video better shows what the person is doing?
```

Report:

- Percent of trials where participants prefer the proposed method.
- Optional confidence rating.

This is not meant to replace the quantitative metrics. It verifies whether the
metric improvements correspond to perceived clarity.

