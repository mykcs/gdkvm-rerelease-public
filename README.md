# GDKVM

Re-release of **GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule**.

Paper: https://openaccess.thecvf.com/content/ICCV2025/html/Wang_GDKVM_Echocardiography_Video_Segmentation_via_Spatiotemporal_Key-Value_Memory_with_Gated_ICCV_2025_paper.html

## What this release is

This is a modern rewrite/retrain/re-evaluation line that preserves the GDKVM scientific lineage while making implementation, data contracts, training, runtime policy, and metric semantics explicit.

It is **not** a byte-for-byte reconstruction of the 2025 repository or environment.

## Authoritative re-release checkpoint

- file: `gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth`
- SHA256: `61b27142ab7ce88cd118e2f11f80c692c4f5696d1d57a399a6d8408fb82d7b57`
- bytes: `421645549`
- role: `gdkvm-rerelease-candidate`
- release artifact repository: `miyuki17/gdkvm-rerelease`

During private RC review that artifact repository remains private. After owner-approved publication, obtain the checkpoint from the release artifact repository and verify it:

```bash
python scripts/verify_checkpoint.py \
  --path gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth \
  --sha256 61b27142ab7ce88cd118e2f11f80c692c4f5696d1d57a399a6d8408fb82d7b57 \
  --bytes 421645549
```

## New CAMUS result (`gdkvm-rerelease-v1`)

Full test cache: 100 view-samples / 50 patients / 200 ED/ES atomic cases.

- Dice: **0.9367208311**
- IoU: **0.8827891934**
- HD95: **3.7239407633 mm**
- ASD: **1.4222699904 mm**
- patient-bootstrap Dice 95% CI: **[0.9305751160, 0.9426554169]**

These are **new re-release results**, not the original ICCV 2025 paper numbers.

The optional clinical readout uses `single-view-area-length-v1`; it is a single-view surrogate, not biplane Simpson.

## Reference runtime

The v1 reference contract is single GPU, eager execution, contiguous memory layout, and FP32.

PyTorch 2.14 has been runtime-qualified on RTX 5090 engineering fixtures. `torch.compile(mode="default")` is optional, not the scientific authority. RTX 3090 throughput is not inferred from a constrained RTX 5090.

## Install

See [docs/INSTALL.md](docs/INSTALL.md).

## Data

See [data/SOURCES.md](data/SOURCES.md).

- CAMUS is acquired from CREATIS / Human Heart Project and is not redistributed here.
- EchoNet-Dynamic requires each user to obtain access through Stanford's current Research Use Agreement path.
- Derived caches are reproducible, but dataset bytes are not part of this repository.

## Train

```bash
python scripts/train_gdkvm.py \
  --config training/configs/camus-v1.yaml \
  --train-cache /authorized/camus-cache/train \
  --val-cache /authorized/camus-cache/val \
  --test-cache /authorized/camus-cache/test \
  --output /path/to/run
```

Formal mode fails closed on partial or wrong-split caches.

## Evaluate

```bash
python scripts/evaluate_gdkvm_checkpoint.py \
  --config training/configs/camus-v1.yaml \
  --checkpoint gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth \
  --cache /authorized/camus-cache/test \
  --split test \
  --output checkpoint-eval.json
```

Metric protocol: `evaluation/PROTOCOL.md`

Protocol ID: `gdkvm-rerelease-v1`

Machine-readable reference receipts are included under `repro/receipts/`.

## Dataset-free smoke

```bash
python scripts/runtime_probe.py --profile eager-reference
python scripts/benchmark_gdkvm_runtime.py --device cpu --modes eager --size 64 --frames 2 --warmup 1 --iterations 2 --output /tmp/gdkvm-runtime.json
```

## License

Source code: Apache License 2.0.

The CAMUS-trained checkpoint has a separate conservative publication policy: CC BY-NC-SA 4.0 / non-commercial scientific research, with CAMUS attribution required. This project policy avoids granting broader rights than the CAMUS source terms; it is not a legal determination about model-weight derivative status.

Dataset terms remain separate from the source-code license.
