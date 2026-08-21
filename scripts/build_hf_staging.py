#!/usr/bin/env python3
"""Assemble an HF dataset staging tree (cards/ + data/parquet + benchmark-metadata.json + README.md)
from a refresh-run cards dir, matching the v2 build mechanics byte-for-byte.

Usage:
  python scripts/build_hf_staging.py --cards <refresh_run>/cards --ref-v2 <v2_staging_dir> --out <v3_staging_dir>
"""
import argparse, copy, glob, json, os, shutil

def inner(c): return c.get("benchmark_card", c)

def normalize_for_parquet(card):
    """Inner body, markers dropped, 4 field-type normalizations. Matches v2 exactly."""
    c = copy.deepcopy(inner(card))
    c.pop("flagged_fields", None)
    c.pop("missing_fields", None)
    d = c.get("data", {})
    if isinstance(d.get("size_breakdown"), dict):
        d["size_breakdown"] = json.dumps(d["size_breakdown"], ensure_ascii=False)
    m = c.get("methodology", {})
    if "judge_num" in m:       m["judge_num"] = str(m["judge_num"])
    if "judge_uses_llm" in m:  m["judge_uses_llm"] = str(m["judge_uses_llm"])
    p = c.get("purpose_and_intended_users", {})
    if isinstance(p.get("out_of_scope_uses"), str):
        p["out_of_scope_uses"] = [p["out_of_scope_uses"]]
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--ref-v2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=516, help="expected card count")
    a = ap.parse_args()
    import datasets

    slugs = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.cards, "*.json")))
    assert len(slugs) == a.expect, len(slugs)

    # 1) cards/ : copy refreshed cards verbatim (byte-identical format)
    out_cards = os.path.join(a.out, "cards"); os.makedirs(out_cards, exist_ok=True)
    for s in slugs:
        shutil.copyfile(os.path.join(a.cards, f"{s}.json"), os.path.join(out_cards, f"{s}.json"))

    # 2) benchmark-metadata.json : {slug: inner_body_with_markers}, indent=2, ensure_ascii=False, NO trailing newline
    md = {s: inner(json.load(open(os.path.join(a.cards, f"{s}.json")))) for s in slugs}
    with open(os.path.join(a.out, "benchmark-metadata.json"), "w") as f:
        f.write(json.dumps(md, indent=2, ensure_ascii=False))

    # 3) data/train-00000-of-00001.parquet : built via HF datasets with the EXACT v2 Features schema
    v2feat = datasets.Dataset.from_parquet(os.path.join(a.ref_v2, "data", "train-00000-of-00001.parquet")).features
    records = [{"benchmark_card": normalize_for_parquet(json.load(open(os.path.join(a.cards, f"{s}.json"))))} for s in slugs]
    ds = datasets.Dataset.from_list(records, features=v2feat)
    assert ds.features == v2feat
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    ds.to_parquet(os.path.join(a.out, "data", "train-00000-of-00001.parquet"))

    # 4) README.md : reuse v2 verbatim
    shutil.copyfile(os.path.join(a.ref_v2, "README.md"), os.path.join(a.out, "README.md"))
    print("v3 staging built at", a.out)

if __name__ == "__main__":
    main()
