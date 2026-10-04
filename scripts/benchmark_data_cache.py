#!/usr/bin/env python3
"""Dataset-free data-cache microbenchmark using the shared receipt plumbing."""
from __future__ import annotations
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from gdkvm_bench import PhaseRecorder, environment_snapshot, write_receipt
from gdkvm_data.npy_cache import NpyCacheReader, cache_nbytes, write_npy_cache
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True); p.add_argument("--code-sha",required=True)
    p.add_argument("--samples",type=int,default=128); p.add_argument("--sequence-length",type=int,default=10)
    p.add_argument("--size",type=int,default=128); p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--iterations",type=int,default=5); p.add_argument("--seed",type=int,default=7)
    p.add_argument("--cache-root",type=Path); a=p.parse_args()
    if min(a.samples,a.sequence_length,a.size,a.iterations)<1: raise SystemExit("positive sizes required")
    managed=a.cache_root is None
    tmp_parent=Path(tempfile.mkdtemp(prefix="gdkvm-data-cache-")) if managed else None
    root=(tmp_parent / "cache") if managed else a.cache_root
    assert root is not None
    if root.exists(): raise SystemExit(f"cache root must not already exist: {root}")
    rng=np.random.default_rng(a.seed)
    frames=rng.integers(0,256,size=(a.samples,a.sequence_length,a.size,a.size),dtype=np.uint8)
    masks=(frames>191).astype(np.uint8); meta=[{"sample_id":f"synthetic-{i:06d}"} for i in range(a.samples)]
    rec=PhaseRecorder()
    with rec.phase("cache_write"): write_npy_cache(root,frames,masks,meta,manifest_extra={"dataset_id":"synthetic-data-io-v1"})
    reader=NpyCacheReader(root,mmap=True); order=np.arange(len(reader),dtype=np.int64); checksum=0
    for _ in range(a.warmup):
        for idx in order[:min(len(order),16)]:
            f,m,_=reader[int(idx)]; checksum ^= int(f[0,0,0]); checksum ^= int(m[0,0,0])
    measured_bytes=0
    for iteration in range(a.iterations):
        iter_order=order if iteration%2==0 else rng.permutation(order)
        with rec.phase("cache_read_full_pass"):
            for idx in iter_order:
                f,m,_=reader[int(idx)]; checksum ^= int(f[0,0,0]); checksum ^= int(m[0,0,0]); measured_bytes += int(f.nbytes+m.nbytes)
    phases=rec.summary(); read_mean=float(phases["cache_read_full_pass"]["mean_s"])
    throughput=((measured_bytes/a.iterations)/max(read_mean,1e-12))/(1024**2)
    receipt={"schema_version":1,"kind":"data-io-benchmark","identity":{"code_sha":a.code_sha,"eval_protocol_id":"gdkvm-rerelease-v1","runtime_profile":"dataset-free-npy-mmap"},"workload":{"dataset_id":"synthetic-data-io-v1","data_cache_format":"contiguous-npy-mmap-v1","samples":a.samples,"sequence_length":a.sequence_length,"input_resolution":[a.size,a.size],"frames_dtype":"uint8","masks_dtype":"uint8"},"environment":environment_snapshot(),"measurement":{"warmup_iterations":a.warmup,"measured_iterations":a.iterations,"phases":phases,"cache_bytes":cache_nbytes(root),"mean_read_throughput_mib_s":throughput,"checksum":checksum}}
    write_receipt(a.output,receipt); print(json.dumps(receipt,indent=2,sort_keys=True))
    if managed and tmp_parent is not None: shutil.rmtree(tmp_parent,ignore_errors=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
