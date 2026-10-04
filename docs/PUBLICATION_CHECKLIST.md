# Publication checklist

Technical qualification and publication authorization are different gates.

## Must pass before public release

- [x] evaluation protocol frozen and tested;
- [x] source lineage frozen;
- [x] official dataset acquisition/split rules documented;
- [x] observability boundary qualified;
- [x] semantics-preserving source cleanup qualified;
- [x] runtime reference/optional acceleration policy qualified for declared v1 scope;
- [x] authoritative re-release checkpoint selected and SHA256 recorded;
- [x] checkpoint-bound `gdkvm-rerelease-v1` training/evaluation receipts recorded;
- [x] checkpoint stored durably in the private release artifact repository;
- [x] clean Python 3.12 GPU install path qualified from a fresh environment;
- [x] clean public snapshot qualification passes on the prior technical RC and must be refreshed on the merged scientific RC;
- [x] checkpoint publication policy reviewed against CAMUS terms: weights are released, if approved, under a conservative CC BY-NC-SA 4.0 / non-commercial scientific-research policy with CAMUS attribution;
- [ ] owner explicitly approves publication and visibility changes.

## Publication boundary

The source code remains Apache-2.0. The CAMUS-trained checkpoint has a separate, more restrictive publication policy. This is a project release policy chosen to avoid granting broader rights than the CAMUS source terms; it is not a legal determination about whether model weights are an adapted work.

The current GitHub authority repository and Hugging Face model repository remain private until the owner explicitly approves publication.

## Do not publish by changing this working repository's visibility

This private repository contains internal provenance and machine-specific evidence in Git history.

Build a clean public snapshot with:

```bash
python scripts/build_public_snapshot.py --output /tmp/gdkvm-public
```

Then publish/import the reviewed snapshot into the intended public repository only after the owner approval gate.

Making this private working repository public would expose historical internal evidence even if files were deleted in the latest commit.
