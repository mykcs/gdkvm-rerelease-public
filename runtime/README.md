# Runtime qualification

PR #6 owns runtime compatibility and optional acceleration. It does not redefine model, data, or evaluation semantics.

## Lanes

- **H / historical** — reproduce captured old environments where practical.
- **C / compatibility** — smallest support floor we are willing to maintain.
- **M / modern** — current optimized environment candidate.

The current candidate matrix lives in `profiles.json`.

## Qualification order

1. CUDA kernel smoke.
2. Small GDKVM model-forward fixture on eager.
3. Same weights / same input under compile modes.
4. Numerical comparison against eager.
5. Cold compile time, steady-state time, and peak VRAM.
6. Only after that: Trainer/DDP/AMP qualification.

A model-forward fixture is runtime evidence, not a paper-result benchmark.

## 3090 boundary

A runtime may be a compatibility **candidate** without being claimed as RTX 3090-qualified. A real 3090 execution receipt is required before the public support matrix says it is tested on 3090.

## Installation discipline

Do not modify host NVIDIA drivers or system CUDA for this workstream. Wheel-contained CUDA runtimes should be tested against the live driver first.


## v1 support claim

The current re-release support claim is deliberately narrow:

- reference/default: single-GPU eager, contiguous layout, FP32;
- tested modern GPU: RTX 5090 with PyTorch 2.14.0+cu130;
- compile-default: optional long-run candidate, not quickstart default;
- RTX 3090: historical environment evidence exists, but no new real-3090 qualification claim;
- DDP/multi-GPU: not yet a release support claim;
- full release environment: Python 3.12, to be fresh-install qualified by the release workstream.

Do not turn an untested compatibility lane into a public support statement.
