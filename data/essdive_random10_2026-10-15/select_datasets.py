"""Randomly select 10 recent public ESS-DIVE datasets with TROWEL-readable files.

Eligible: public packages with dateUploaded on or after CUTOFF and datePublished
year >= CUTOFF's year. Eligible packages are sorted by id, shuffled with SEED,
and taken in order, skipping any with no .csv/.tsv/.txt/.xls/.xlsx/.xml file,
until N are chosen.

Usage: python select_datasets.py  (writes selection.json and dois.txt here)

Note: the catalogue changes as datasets are added or revised, so a later run
may draw a different set even with the same seed.
"""
import json
import os
import random
import time
from urllib.parse import quote

import requests

API = "https://api.ess-dive.lbl.gov/packages"
CUTOFF = "2023-10-15"
SEED = 20261015
N = 10
READABLE = {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".xml"}
TIMEOUT = (30, 300)
HERE = os.path.dirname(os.path.abspath(__file__))


def get(url, **params):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(10)
    raise RuntimeError(f"Request failed: {url}")


def catalogue():
    rows, start, size = [], 1, 100
    while True:
        d = get(API, isPublic="true", pageSize=size, rowStart=start)
        for p in d["result"]:
            ds = p["dataset"]
            rows.append({"id": p["id"], "doi": ds.get("@id", ""), "name": ds.get("name", ""),
                         "dateUploaded": p.get("dateUploaded", ""),
                         "datePublished": str(ds.get("datePublished", ""))})
        start += size
        if start > d["total"]:
            return rows


def pub_year(r):
    try:
        return int(r["datePublished"][:4])
    except ValueError:
        return 0


def main():
    rows = catalogue()
    eligible = sorted((r for r in rows if r["dateUploaded"] >= CUTOFF
                       and pub_year(r) >= int(CUTOFF[:4])), key=lambda r: r["id"])
    random.Random(SEED).shuffle(eligible)

    chosen, skipped = [], []
    for r in eligible:
        p = get(f"{API}/{quote(r['id'], safe='')}")
        files = [f.get("name", "") for f in p["dataset"].get("distribution", [])]
        r["n_files"] = len(files)
        r["n_readable"] = sum(1 for f in files if os.path.splitext(f.lower())[1] in READABLE)
        (chosen if r["n_readable"] else skipped).append(r)
        if len(chosen) == N:
            break

    with open(os.path.join(HERE, "selection.json"), "w") as f:
        json.dump({"cutoff": CUTOFF, "seed": SEED, "catalogue_size": len(rows),
                   "pool_size": len(eligible), "chosen": chosen, "skipped": skipped}, f, indent=1)
    with open(os.path.join(HERE, "dois.txt"), "w") as f:
        f.writelines(r["doi"].removeprefix("doi:") + "\n" for r in chosen)


if __name__ == "__main__":
    main()
