# Dataset source authority

This repository stores code, manifests, hashes, and receipts — never the datasets themselves.

## CAMUS

Canonical landing page: https://www.creatis.insa-lyon.fr/Challenge/camus/

The current CREATIS page routes downloads to the Human Heart Project Girder. The re-release downloader uses that public Girder API and records per-patient archive hashes.

The current NIfTI re-release contains 500 patients with an explicit patient-level split:

- 400 training
- 50 validation
- 50 testing

This is the default split for gdkvm-rerelease-v1-data.



### Owner-controlled Hugging Face recovery snapshots

The owner already maintains two Hugging Face snapshots of the same CAMUS lineage:

- `miyuki17/CAMUS_public`, revision `bd4fd12ae57e4b84259120e9845698f2031d00bf` — canonical unpacked NIfTI snapshot;
- `miyuki17/camus_public_20250706`, revision `6ace0dc4943314db5fccb67ccd443229cddb6830` — source-archive snapshot containing `CAMUS_public.zip`.

These snapshots are preferred for **owner-controlled recovery/materialization** when the official Girder path is slow or unreliable. They do not replace CREATIS as the scientific/source authority. A recovered snapshot must still pass patient-count, split-content, structure, and payload-integrity checks before use.

### Historical GDKVM discrepancy

The owner's 2025 camus_public_datasplit_20250706.json expands that same 400/50/50 patient split into 800/100/100 2CH/4CH sample IDs with no patient overlap.

A historical GDKVM paper draft, however, says CAMUS was split 7:1:2.

These facts are not silently reconciled. The re-release uses the verifiable official NIfTI split; historical reproduction must separately establish what the published training run actually consumed.

## EchoNet-Dynamic

Canonical landing page: https://echonet.github.io/dynamic/

The official site currently requires individual registration through Stanford AIMI/Redivis and states that users may not redistribute the dataset or share its download link.

Therefore this repository intentionally contains no EchoNet download URL or automated agreement acceptance.

After the owner places an individually authorized copy on the machine, validate it with:

    python scripts/validate_echonet_source.py --root /authorized/path/to/EchoNet-Dynamic

The source FileList.csv is the split authority. Expected counts are TRAIN 7465, VAL 1288, TEST 1277.



### Owner-controlled Hugging Face recovery snapshot

The owner also has pre-existing Hugging Face snapshots for internal recovery:

- `miyuki17/EchoNet-Dynamic-unzipped`, revision `e5eb1aa94d60be651b451d3b62b3e89d75a144a2`;
- `miyuki17/EchoNet-Dynamic-zip`, revision `cc64e45c68ba43b5856734c8dd1c30c4f4e9b866`, containing `EchoNet-Dynamic.zip`.

The ZIP snapshot records SHA256
`326939c379801daab839233a35613ca9e5d15529c758fe11e31f798b62a462b4`,
which matches the previously audited authorized archive candidate.

These existing owner-controlled Hub copies may be used to restore the owner's authorized research workspace. They are **not** a new public acquisition route for the re-release. Public instructions must continue to route each user through Stanford's current access/RUA process, and this project must not create additional redistribution links or mirrors.

### Verified local authorized archive candidate

A user-local private EchoNet-Dynamic archive was audited without publishing or redistributing it.

Observed identity:

- SHA256: `326939c379801daab839233a35613ca9e5d15529c758fe11e31f798b62a462b4`;
- 10,030 `.avi` video files;
- 10,030 `FileList.csv` rows;
- official split counts: TRAIN 7,465 / VAL 1,288 / TEST 1,277;
- 425,010 tracing rows covering 10,025 tracing filenames, each with exactly two annotated frames.

For segmentation supervision, six FileList videos have no tracing rows:

- `0X234005774F4CB5CD.avi` (TRAIN)
- `0X2DC68261CBCC04AE.avi` (TRAIN)
- `0X35291BE9AB90FB89.avi` (TRAIN)
- `0X5515B0BD077BE68A.avi` (TRAIN)
- `0X5DD5283AC43CCDD1.avi` (TEST)
- `0X6C435C1B417FDE8A.avi` (TRAIN)

One tracing filename, `0X4F8859C8AB4DA9CB.avi`, has no matching FileList/video entry in the audited archive.

Therefore the re-release segmentation cache contract expects **10,024 eligible videos**:

- TRAIN 7,460
- VAL 1,288
- TEST 1,276

These exclusions are explicit provenance, not silent loader drops. The private archive is evidence of an authorized local copy; public acquisition instructions still route each user to the official Stanford access path and Research Use Agreement.


## Rule

Mirrors may be used only as private recovery evidence after provenance checks. Public re-release instructions always route users back to the official dataset owners and their current terms.
