#!/usr/bin/env python3
"""Probe the current official CAMUS Girder metadata without downloading patient data."""
from __future__ import annotations
import argparse, io, json, urllib.parse, urllib.request, zipfile
from hashlib import sha256
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from gdkvm_data.contract import CAMUS_EXPECTED_PATIENTS, validate_disjoint_splits
BASE="https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
PATIENT_PARENT_ID="63fde55f73e9f004868fb7ac"; SPLIT_FOLDER_ID="66e27d12961576b1bad4e4e1"
def request_bytes(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r: return r.read()
def list_patient_folders():
    q=urllib.parse.urlencode({"parentType":"folder","parentId":PATIENT_PARENT_ID,"limit":1000,"offset":0,"sort":"name","sortdir":1})
    return json.loads(request_bytes(f"{BASE}/folder?{q}").decode())
def parse_split_zip(payload):
    mapping={"subgroup_training.txt":"train","subgroup_validation.txt":"val","subgroup_testing.txt":"test"}
    splits={}; information=""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for member in z.namelist():
            name=Path(member).name
            if name in mapping:
                splits[mapping[name]]=[x.strip() for x in z.read(member).decode().splitlines() if x.strip()]
            elif name=="information.txt":
                information=z.read(member).decode().strip()
    if set(splits)!={"train","val","test"}: raise ValueError(f"incomplete split archive: {sorted(splits)}")
    validate_disjoint_splits(splits, expected_counts=CAMUS_EXPECTED_PATIENTS)
    return splits, information
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output", type=Path); a=p.parse_args()
    folders=sorted(list_patient_folders(), key=lambda r:r["name"]); names=[r["name"] for r in folders]
    if names != [f"patient{i:04d}" for i in range(1,501)]: raise SystemExit("unexpected CAMUS patient identities")
    payload=request_bytes(f"{BASE}/folder/{SPLIT_FOLDER_ID}/download", 120)
    splits, info=parse_split_zip(payload)
    canonical=json.dumps(splits,sort_keys=True,separators=(",",":")).encode("utf-8")
    content_sha = sha256(canonical).hexdigest()
    registry = json.loads((ROOT / "data" / "source_registry.json").read_text(encoding="utf-8"))
    expected = registry["datasets"]["camus"]
    if len(names) != int(expected["expected_patients"]):
        raise SystemExit("official CAMUS patient count changed from registry authority")
    if {k: len(v) for k, v in splits.items()} != expected["rerelease_split_counts"]:
        raise SystemExit("official CAMUS split counts changed from registry authority")
    if content_sha != expected["official_split_content_sha256"]:
        raise SystemExit(
            "official CAMUS canonical split content changed: "
            f"{content_sha} != {expected['official_split_content_sha256']}"
        )
    receipt={"source":"official-creatis-human-heart-girder","api_base":BASE,"patient_folder_count":len(names),"patient_first":names[0],"patient_last":names[-1],"split_counts":{k:len(v) for k,v in splits.items()},"split_content_sha256":content_sha,"split_archive_transport_sha256":sha256(payload).hexdigest(),"split_information":info}
    text=json.dumps(receipt,indent=2,sort_keys=True)+"\n"
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text,encoding="utf-8")
    print(text,end=""); return 0
if __name__=="__main__": raise SystemExit(main())
