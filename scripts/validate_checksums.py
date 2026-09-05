#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, csv, argparse
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'MANIFEST_SHA256.csv'
def files():
    for p in ROOT.rglob('*'):
        if (
            p.is_file()
            and p != MANIFEST
            and '.git' not in p.parts
            and '__pycache__' not in p.parts
            and p.name != '.DS_Store'
            and p.suffix != '.tmp'
        ):
            yield p
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args=ap.parse_args()
    rows=[(str(p.relative_to(ROOT)),sha(p)) for p in sorted(files())]
    if args.write:
        with open(MANIFEST,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['path','sha256'])
            w.writerows(rows)
        print('wrote', MANIFEST)
        return
    saved={r['path']:r['sha256'] for r in csv.DictReader(open(MANIFEST))}
    now=dict(rows)
    bad=[k for k,v in saved.items() if now.get(k)!=v]
    extra=sorted(set(now)-set(saved))
    missing=sorted(set(saved)-set(now))
    if bad or extra or missing:
        print('checksum validation failed')
        print('bad',bad)
        print('extra',extra)
        print('missing',missing)
        raise SystemExit(1)
    print('checksum validation passed')
if __name__=='__main__': main()
