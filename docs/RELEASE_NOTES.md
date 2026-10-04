# GDKVM re-release notes

Status: **scientific + technical release candidate; publication awaits explicit owner approval**.

## Relationship to ICCV 2025

This work does not rewrite publication history. Keep distinct:

1. original ICCV 2025 paper results;
2. behavior of the previously public code;
3. any historical reproduction under an old evaluator/runtime;
4. new results produced by the rewritten project under `gdkvm-rerelease-v1`.

The new checkpoint and results below are **re-release results**, not a byte-for-byte recovery of the original paper environment.

## New authoritative CAMUS checkpoint

- `gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth`
- bytes: **421,645,549**
- SHA256: `61b27142ab7ce88cd118e2f11f80c692c4f5696d1d57a399a6d8408fb82d7b57`
- training code SHA: `264d3d64d7b102ec43c3b150cbf6695e4d125b8a`
- full CAMUS caches: train/val/test = **800/100/100** view-samples
- schedule: **3000** iterations, batch **10**, eager, contiguous, FP32, pretrained backbones, 23 GiB allocator envelope
- final train loss: **0.0859814882**
- non-finite losses: **0**

The checkpoint is durably stored in the private release artifact repository `miyuki17/gdkvm-rerelease` until publication is explicitly approved.

## New `gdkvm-rerelease-v1` CAMUS test result

Full test cache: 100 view-samples / 50 patients / 200 ED/ES atomic segmentation cases.

- Dice mean: **0.9367208311**
- IoU mean: **0.8827891934**
- HD95 mean: **3.7239407633 mm**
- ASD mean: **1.4222699904 mm**
- patient-bootstrap Dice 95% CI: **[0.9305751160, 0.9426554169]**

Clinical binding `single-view-area-length-v1` is a single-view surrogate and **not** biplane Simpson:

- LVEF MAE: **5.91285 percentage points**
- RMSE: **11.15788 percentage points**
- Pearson r: **0.75244**
- bias: **-1.46854 percentage points**

Machine-readable authorities:

- `repro/receipts/lyg-camus-formal-training-20261001.json`
- `repro/receipts/lyg-camus-formal-evaluation-20261001.json`
- `repro/receipts/lyg-camus-formal-artifact-registration-20261001.json`

## Data

CAMUS v1 uses the canonical 500-patient source and 400/50/50 patient split, expanded to 800/100/100 2CH/4CH view-samples. Temporal selection and contiguous-NPY cache construction are explicit.

EchoNet-Dynamic remains separately authorized; dataset bytes and restricted download links are not redistributed.

## Runtime

Reference/default v1: single GPU, eager, contiguous, FP32.

Qualified engineering evidence exists for PyTorch 2.14/CUDA 13.0 on RTX 5090. `torch.compile(mode="default")` remains optional. No new RTX 3090 throughput or DDP support claim is made.

## License boundary

- source code: Apache-2.0;
- CAMUS-trained checkpoint, if publication is approved: conservative CC BY-NC-SA 4.0 / non-commercial scientific-research policy with CAMUS attribution;
- datasets are not redistributed by this release.

## Remaining publication gate

The scientific and technical candidate is qualified. The remaining gate is an **explicit owner approval** to publish the clean GitHub snapshot and change the checkpoint artifact visibility. The private authority repository itself remains private.
