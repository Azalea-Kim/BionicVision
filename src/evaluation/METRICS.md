# Poster Evaluation Metrics

This project uses a compact metric set for an EPIC-KITCHENS/VISOR poster
evaluation. The goal is to test the whole simplification pipeline without
turning the poster into a general segmentation benchmark.

## Evaluation Setup

For each evaluated clip/window:

- **Foreground**: hands plus the currently active/contacted object.
- **Background/context objects**: visible non-hand objects that are not
  currently foreground.
- **Primary supervised target**: sparse VISOR masks plus dense VISOR
  `type=1` filtered ground-truth masks.
- **Pseudo target**: dense VISOR `type=0` automatic/interpolated masks, reported
  separately from the primary metrics.
- **Pipeline output**: grayscale simplified video. For the current benchmark
  phase, evaluation is performed before pulse2percept simulation.

Object representation must be compatible with both filled-mask and outline
rendering. A ground-truth object is counted as represented if the simplified
output overlaps either the object mask or a dilated boundary band by at least a
small fixed threshold.

The implemented entry point is `evaluation.evaluate_clip`. It accepts a sequence
of parsed VISOR `EpicFrame` annotations and the corresponding grayscale
simplified frames. By default, `EvaluationConfig.annotation_quality="gold"`,
which scores sparse masks and dense `type=1` filtered ground truth only.
`annotation_quality="pseudo"` scores dense `type=0` automatic/interpolated masks
as a secondary diagnostic, and `annotation_quality="all"` is available only for
debugging continuity with older mixed-target reports. Optional pulse2percept
frames can be supplied when load should be measured on simulated percepts
instead of the preprocessed simplified video.

Dense VISOR interpolation JSONs should be aligned to video/pipeline outputs by
explicit frame index. Some local benchmark clips have duplicate dense annotation
records for the same frame index, and some frame indices may be missing from the
dense JSON. Evaluation code should merge duplicate records for a frame and score
only frames that have annotations. Dense VISOR polygons are generated at
`854x480` and must be scaled to the evaluated frame resolution before
rasterization.

## Supervised Metrics

These use VISOR hand/object masks, active/contact object annotations, and object
tracks.

### 1. Foreground Recall vs Active-Area Load

Measures whether the critical current interaction remains visible.

- Ground truth: hand masks plus currently active/contacted object masks.
- Prediction: represented foreground objects in the simplified output.
- Y-axis: foreground object recall.
- X-axis: active-area load, the fraction of output pixels above a fixed
  activity threshold.
- Summary: AUC under a fixed active-area cap, e.g.
  `AUC@active_area<=L`.

This is the main task-preservation metric.

### 2. Background Context Recall vs Active-Area Load

Measures useful visible context preservation under a perceptual-load budget.

- Ground truth: visible non-hand objects that are not currently foreground.
- Prediction: represented background/context objects.
- Y-axis: background/context object recall.
- X-axis: active-area load.
- Summary: AUC under a fixed active-area cap, e.g.
  `AUC@active_area<=L`.

This should be interpreted as a tradeoff curve, not as "higher recall is always
better" independent of load.

### 3. Object-Frame Miss Rate and Track Fragmentation

Measures object-level temporal consistency.

Object-frame miss rate counts visible foreground/background object appearances
that are missing from the simplified output. This is the current
`track_dropout_rate` field for backward compatibility, but the more accurate
name is object-frame miss rate.

The score is normalized by visible target object-frame appearances, so it is in
`[0, 1]`. Lower is better: `0` means no visible target appearances were missed;
`1` means every visible target appearance was missed.

Track fragmentation is separate: for each represented visible track, count
whether it has an internal represented -> missing -> represented gap. This
captures unstable reacquisition that object-frame miss rate alone cannot.

## Unsupervised Metric

### 4. Flow-Compensated Flicker

Measures whole-output temporal instability after accounting for camera/user
motion.

Procedure:

1. Estimate optical flow between RGB frame `t` and frame `t+1`.
2. Warp simplified output `S_t` into frame `t+1`.
3. Compare warped `S_t` with `S_{t+1}` using mean absolute difference.
4. Average over the clip.

The score is mean absolute normalized intensity difference after warping, so it
is in `[0, 1]`. A value of `0.10` means that after compensating for camera/user
motion, the simplified output changes by about 10% of the full brightness range
per pixel on average. Lower is better.

## Active-Area Load

Active-area load is the x-axis/budget variable for the recall curves, not a
standalone success metric.

```text
active area = mean(1[P_t(x, y) > threshold])
```

For the current unpercepted benchmark, `P_t` is the simplified output frame
after fixed normalization. Use a fixed video-level threshold, not per-frame
normalization or adaptive brightness thresholds.

Active area is in `[0, 1]`: it is the average fraction of pixels above the fixed
activity threshold.

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
