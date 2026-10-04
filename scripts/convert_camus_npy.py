#!/usr/bin/env python3
"""Convert authorized official CAMUS archives into the contiguous NPY cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from gdkvm_data.camus import build_npy_cache

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("train","val","test"),required=True)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--frames",type=int,default=10)
    p.add_argument("--patient-limit",type=int)
    a=p.parse_args()
    build_npy_cache(a.source_root,a.output,split=a.split,size=a.size,frame_count=a.frames,patient_limit=a.patient_limit)
    print(a.output)
    return 0
if __name__=="__main__": raise SystemExit(main())
