# Dev design — GDKVM public re-release

Development should minimize drift from the verified private release candidate while keeping the public tree understandable.

- validation should use committed tests/repro scripts;
- public release preparation must not depend on private credentials;
- publication is a separate decision from private remediation;
- no website Production role is implied.

A future CI gate should test public-repository invariants only and must not require private datasets, internal servers or secret state.
