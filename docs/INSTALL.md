# Installation

## Reference release environment

Use Python **3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## PyTorch

Install PyTorch separately so the wheel matches the intended machine.

### Modern CUDA lane used for RTX 5090 qualification

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

This is evidence for the tested Blackwell lane, not a claim that every CUDA host should use cu130 or that a constrained RTX 5090 is an RTX 3090 benchmark.

### macOS / CPU smoke

```bash
pip install torch==2.14.0
```

### Historical evidence only

The old public environment captured PyTorch 2.6.0 + CUDA 11.8. It remains historical evidence and is not the new default.

## Project dependencies

```bash
pip install -r requirements-release.txt
```

`requirements-train.txt` records the same qualified core dependencies plus formal training-boundary notes. PyTorch remains installed separately.

Optional observability/cache tools:

```bash
pip install -r requirements-optional.txt
```

W&B is optional.

## Verify the clean environment

```bash
python scripts/runtime_probe.py --profile eager-reference
python -m unittest discover -s tests -p 'test_*.py'
```

## Checkpoint

Authoritative v1 candidate:

- file: `gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth`
- bytes: `421645549`
- SHA256: `61b27142ab7ce88cd118e2f11f80c692c4f5696d1d57a399a6d8408fb82d7b57`
- artifact repository: `miyuki17/gdkvm-rerelease`

During private RC review the artifact repository remains private. After owner-approved publication, download the checkpoint from that release repository and verify it:

```bash
python scripts/verify_checkpoint.py \
  --path gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth \
  --sha256 61b27142ab7ce88cd118e2f11f80c692c4f5696d1d57a399a6d8408fb82d7b57 \
  --bytes 421645549
```

## Prepare CAMUS

Follow `data/SOURCES.md` and acquire an authorized canonical source.

```bash
python scripts/prepare_camus_cache.py --source /authorized/CAMUS_public --output /cache/train --split train
python scripts/prepare_camus_cache.py --source /authorized/CAMUS_public --output /cache/val --split val
python scripts/prepare_camus_cache.py --source /authorized/CAMUS_public --output /cache/test --split test
```

The formal v1 gate expects 800 / 100 / 100 CAMUS view-samples.

## Train

```bash
python scripts/train_gdkvm.py \
  --config training/configs/camus-v1.yaml \
  --train-cache /cache/train \
  --val-cache /cache/val \
  --test-cache /cache/test \
  --output /path/to/run
```

## Evaluate

```bash
python scripts/evaluate_gdkvm_checkpoint.py \
  --config training/configs/camus-v1.yaml \
  --checkpoint gdkvm_rerelease_camus_v1_main_training_ckpt_3000.pth \
  --cache /cache/test \
  --split test \
  --output checkpoint-eval.json
```
