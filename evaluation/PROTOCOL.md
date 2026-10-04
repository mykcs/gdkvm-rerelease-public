# GDKVM re-release evaluation protocol v1

Protocol ID: `gdkvm-rerelease-v1`

Status: **current metric-semantic contract when merged**

Dataset release, split, temporal sampling, and preprocessing are bound by separate dataset protocol identities. This file freezes metric meaning so those later bindings cannot silently redefine a metric.

This protocol defines metric semantics and reporting. Dataset acquisition, split construction, temporal sampling, and preprocessing are external protocol components and must be versioned separately.

## Semantic reference

Primary metric-semantics reference:

- Awesome Echocardiography Reference Metrics v1.1.0
- repository: `wangrui2025/awesome-echocardiography`
- pinned upstream commit: `25b0e5ec7ca72093330f6d30db0760ab0671fa7f`
- path: `reference/metrics_v1/`

The upstream reference is the mathematical semantic authority. GDKVM adds project-specific grouping/reporting rules below.

No source code is copied into this repository by this PR.

## Problem fingerprint

GDKVM is treated as 2-D echocardiographic semantic segmentation over video, with clinically relevant chamber boundaries and derived cardiac-function quantities.

Consequences:

- use complementary overlap and boundary metrics;
- keep per-patient hierarchy;
- keep physical units when spacing is available;
- separate segmentation accuracy from derived clinical agreement;
- do not treat temporal smoothness as a substitute for ground-truth segmentation accuracy.

## Segmentation metrics

### Core re-release set

Every accepted segmentation result reports:

1. **Dice**
2. **HD95**
3. **ASD**

Rationale:

- Dice captures region overlap;
- HD95 captures robust worst-boundary error;
- ASD captures average boundary error.

### Compatibility set

Also compute when needed for historical/published comparison:

4. **IoU**
5. **full Hausdorff distance (HD)**

IoU remains useful because the ICCV 2025 GDKVM paper reports mIoU.

Full HD remains useful because both the GDKVM paper and official CAMUS evaluation history use Hausdorff distance, even though HD95 is more robust to isolated outliers.

### Not in the mandatory v1 set

**Normalized/Surface Dice** is not mandatory in v1.

It may be added later only after a defensible tolerance is selected in physical or image units. An arbitrary tolerance chosen after looking at results is not acceptable.

A temporal-consistency metric may be added as a project-specific diagnostic, but it is not a primary accuracy metric in v1.

## Exact segmentation semantics

Use Awesome Echocardiography Reference Metrics v1.1 semantics:

- atomic item: one prediction mask + one reference mask for one patient/video, structure, view, and annotated phase/frame;
- Dice/IoU use binarized masks and no training-loss smoothing term;
- HD/HD95/ASD operate on surfaces;
- spacing order is `(y_spacing, x_spacing)`;
- physical spacing -> distance metrics in mm;
- unavailable spacing -> `(1, 1)` and explicit px units;
- both masks empty -> N/A for segmentation metrics and count `n_both_empty`;
- exactly one mask empty -> Dice/IoU = 0 and the finite field-of-view distance penalty defined by the reference standard;
- HD95 computes the 95th percentile separately in both directions, then takes the maximum;
- ASD uses the pooled symmetric surface-distance mean.

## Aggregation

Never reduce over arbitrary batch or GPU partitions.

Preserve these axes whenever available:

- dataset;
- patient/video;
- structure;
- view;
- phase/frame.

### CAMUS

Published group summaries should preserve:

- A2C / A4C;
- ED / ES or other explicitly annotated phases;
- LV endocardium;
- LV epicardium/myocardium representation as defined by the accepted data protocol;
- LA when evaluated.

Do not collapse all structures/views/phases into one headline number unless the macro rule is explicitly declared.

### EchoNet-Dynamic

Preserve:

- patient/video;
- ED / ES annotated frames;
- LV structure.

### Mean and uncertainty

For each reported group:

- report mean;
- report valid sample count;
- report `n_one_empty`;
- report `n_both_empty`.

For the re-release tables, add a patient-level 95% bootstrap confidence interval when computationally practical. Bootstrap resampling must happen at the patient/video unit, not at the pixel or batch level.

Historical ICCV tables are not retroactively assigned confidence intervals.

## Clinical / cardiac-function metrics

For accepted LVEF results report at least:

- MAE;
- Pearson `r`;
- bias = prediction − reference;
- sample SD of paired differences (`ddof=1`);
- Bland–Altman 95% limits of agreement.

Recommended secondary metrics:

- RMSE;
- R².

If EDV/ESV are reconstructed and scientifically meaningful, the same MAE/correlation/agreement family may be reported for EDV and ESV.

## LVEF upstream method

A metric result is invalid without naming how EDV/ESV or the surrogate quantity was obtained.

Do not mix:

- biplane Simpson / Method of Disks;
- single-plane volume estimation;
- area-change surrogate;
- dataset-provided clinical values.

If two methods are compared, treat them as different protocol variants.

## Pre/post-processing boundary

The following are outside metric definitions and must be recorded by the experiment protocol:

- resize resolution;
- image interpolation;
- mask interpolation;
- threshold;
- connected-component filtering;
- temporal frame selection;
- sequence sampling;
- spatial spacing update after resize;
- any contour smoothing;
- LVEF/volume reconstruction.

## Historical compatibility

The ICCV 2025 numbers remain historical evidence under `gdkvm-historical-iccv2025`.

The re-release protocol is `gdkvm-rerelease-v1`.

Do not overwrite an old table cell with a recomputed value unless the label clearly says it is a re-evaluation.

## Community-alignment notes

This contract intentionally keeps complementary overlap + boundary metrics, consistent with the problem-aware approach advocated by Metrics Reloaded.

Official CAMUS evaluation historically reports Dice, Hausdorff/mean contour distance, and clinical correlation/bias/agreement quantities. The re-release keeps those compatibility paths while using the more explicit modern reference semantics above.

## Qualification evidence

The independent implementation in `gdkvm_eval/metrics.py` is covered by dataset-free fixtures in `tests/test_metrics.py`.

`evaluation/reference/compatibility_receipt.json` records a deterministic comparison against the pinned Awesome Echocardiography Reference Metrics v1.1 implementation:

- 69 segmentation cases;
- exact status agreement;
- maximum absolute difference 0.0 for Dice, IoU, HD, HD95, and ASD;
- maximum absolute difference 0.0 for the compared regression and Bland–Altman quantities;
- acceptance tolerance 1e-12.

## Downstream bindings

The following remain intentionally outside this metric-semantic owner:

- exact CAMUS structure/phase mapping;
- exact EchoNet-Dynamic annotation/frame mapping;
- dataset split identity;
- temporal sampling;
- LVEF upstream volume/surrogate reconstruction;
- preprocessing/post-processing.

Those are bound by the dataset/evaluation experiment contract. They may choose inputs to these metrics but may not change the metric definitions under the same protocol ID.
