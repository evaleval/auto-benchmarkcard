"""Step 3 of the gold set: the faithfulness judge (deferred until now, runs with
FactReasoner v3). Judges, per card, whether filled fields are supported by the
benchmark's source text, whether 'Not specified' fields are genuinely absent
(completeness truth), and whether possible_risks are grounded.

No pipeline inference. The judge is a separate LLM judge (not the composer model,
to avoid self-preference bias), run via the documented prompt below (reproducible).
Three steps:

  prepare:  build per-card judge inputs from stored artifacts (deterministic)
  (judge):  run the judge over those inputs using JUDGE_PROMPT/JUDGE_SCHEMA
  ingest:   aggregate the structured verdicts into the report + calibration sample

Usage:
  python scripts/judge_gold_set.py prepare --label v31
  # -> /tmp/gold/judge_inputs_v31/<name>.json  (one file per card, with source_text)
  # run the judge workflow, save its {name: verdict} result to a json file
  python scripts/judge_gold_set.py ingest --label v31 --results /tmp/gold/judge_results_v31.json
  # -> gold_set/judge_<label>.json + gold_set/calibration_sample_<label>.csv (+ key)
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import (  # noqa: E402
    _DISPLAY_ONLY_FIELDS, GOLD_SECTIONS, extract_card, is_not_specified,
)

REPO = os.path.join(os.path.dirname(__file__), "..")
GOLD_DIR = os.path.join(REPO, "gold_set")

SOURCE_CAP = 280_000  # chars (~70k tokens); bounds the largest full-paper sources
_SECTION_SKIP = {"provenance", "flagged_fields", "missing_fields"}
# Skip deterministic fields plus link/tag metadata that are not paper-grounded claims.
# Explicit literal, decoupled from DETERMINISTIC_FIELDS so that a future
# DETERMINISTIC_FIELDS_V2 cannot silently change what the judge measures. Display-only
# fields are added set-driven (never hand-coded here).
JUDGE_SKIP = frozenset({
    "benchmark_details.languages",
    "benchmark_details.data_type",
    "ethical_and_legal_considerations.data_licensing",
    "data.size",
    "data.format",
    "methodology.metrics",
    "benchmark_details.resources",
    "benchmark_details.appears_in",
    # Additive (bv-to-0 item 7): structured/scalar v2 fields with no v31 baseline,
    # so adding them cannot change the judging of any existing field. NOT NLI-prose
    # (dict/bool/int/date), so they are not paper-grounded claims to faithfulness-judge.
    # No audience key here -- the two audience keys ARE judged as prose.
    "data.size_breakdown",
    "data.collection_date",
    "methodology.judge_uses_llm",
    "methodology.judge_num",
    "methodology.judge_models",
}) | _DISPLAY_ONLY_FIELDS

# Documented judge prompt (reproducibility). The workflow embeds this per card.
JUDGE_PROMPT_VERSION = "v2-eee"

JUDGE_PROMPT = """You are a careful faithfulness judge for an AI-benchmark documentation card.
You are NOT the model that wrote the card. Judge ONLY against the provided source text.

Read the file {input_path} (JSON). It has: name, source_text (the benchmark's own
paper/webpage/README, an [EEE EVALUATION DATA] block with machine-collected evaluation
results for this benchmark, plus retrieved context), fields (card fields to judge), risks.

For EACH entry in fields:
  - If is_ns is true (the field is 'Not specified'): set status='not_specified',
    specificity='na', and info_in_source='yes_primary' if a source OTHER than the
    [EEE EVALUATION DATA] block actually contains the information this field would
    hold (so the card MISSED it), 'yes_eee_only' if ONLY the [EEE EVALUATION DATA]
    block contains it, else 'no' (a correct abstention). Briefly say where in note.
  - If is_ns is false: judge whether the field VALUE is supported by the source:
    status='supported' (fully backed by the paper/webpage/README/retrieved context,
    i.e. WITHOUT needing the [EEE EVALUATION DATA] block),
    'supported_by_eee_only' (fully backed, but only when the [EEE EVALUATION DATA]
    block is counted -- without it the claim would not be fully backed),
    'partial' (partly backed / overreaches, even counting ALL sources including EEE),
    or 'unsupported' (not in any source / contradicted). Set info_in_source='na'.
    Also set specificity='specific' (concrete, benchmark-specific) or 'generic'
    (vague filler that could apply to any benchmark). Quote/point to the evidence in
    note; for 'supported_by_eee_only', point to the EEE evidence.

For EACH risk category: set relevant_and_grounded='yes' if the risk is both relevant
to this benchmark AND supported by the source (here the [EEE EVALUATION DATA] block
counts as source like any other), else 'no'.

Be strict: plausible-sounding claims with no source support are 'unsupported'. A claim
backed only by the [EEE EVALUATION DATA] block is 'supported_by_eee_only', never
'supported'. If the primary sources (paper/webpage/README/retrieved context) fully back
a claim, it is 'supported' even where the EEE data differs from them. Do not reward
fluency. Return the structured verdict; cover every field and risk."""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "field_verdicts", "risk_verdicts"],
    "properties": {
        "name": {"type": "string"},
        "field_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "status", "specificity", "info_in_source", "note"],
                "properties": {
                    "path": {"type": "string"},
                    "status": {"type": "string",
                               "enum": ["supported", "supported_by_eee_only", "partial",
                                        "unsupported", "not_specified"]},
                    "specificity": {"type": "string", "enum": ["specific", "generic", "na"]},
                    "info_in_source": {"type": "string",
                                       "enum": ["yes_primary", "yes_eee_only", "no", "na"]},
                    "note": {"type": "string"},
                },
            },
        },
        "risk_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "relevant_and_grounded", "note"],
                "properties": {
                    "category": {"type": "string"},
                    "relevant_and_grounded": {"type": "string", "enum": ["yes", "no"]},
                    "note": {"type": "string"},
                },
            },
        },
    },
}


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _first(g):
    hits = sorted(glob.glob(g))
    return hits[0] if hits else None


def _text(obj):
    if not obj:
        return ""
    if isinstance(obj, dict):
        return obj.get("text") or obj.get("content") or obj.get("markdown") or ""
    return str(obj)


def _rag_contexts(rag_dir):
    out = []
    for rp in sorted(glob.glob(os.path.join(rag_dir, "*"))):
        try:
            if rp.endswith(".jsonl"):
                for line in open(rp):
                    o = json.loads(line)
                    ctxs = o.get("contexts") or (o.get("formatted_rag_results") or {}).get("contexts") or []
                    out += [c if isinstance(c, str) else json.dumps(c) for c in ctxs]
            else:
                o = _load(rp) or {}
                ctxs = o.get("contexts") or []
                out += [c if isinstance(c, str) else json.dumps(c) for c in ctxs]
        except Exception:
            continue
    return out


def assemble_source(run_dir):
    """Primary = paper/doc + webpage + README, then the EEE artifact block (reserved
    budget, lowest trust tier), then retrieved context filling the rest."""
    to = os.path.join(run_dir, "tool_output")
    doc = _text(_load(_first(os.path.join(to, "docling", "*.json"))))
    html = _text(_load(_first(os.path.join(to, "html", "*.json"))))
    hf = _load(_first(os.path.join(to, "hf", "*.json"))) or {}
    readme = hf.get("readme_markdown") or hf.get("readme") or ""
    eee = _load(_first(os.path.join(to, "eee", "*.json")))

    parts = []
    if doc:
        parts.append("[PAPER / DOCUMENT]\n" + doc)
    if html:
        parts.append("[WEBPAGE]\n" + html)
    if readme:
        parts.append("[HF README]\n" + readme)
    primary = "\n\n".join(parts)

    # EEE artifacts are small (~3KB); reserve their budget so the primary
    # truncation can never drop the block.
    eee_block = "[EEE EVALUATION DATA]\n" + json.dumps(eee, indent=2, ensure_ascii=False) if eee else ""
    budget = max(0, SOURCE_CAP - (len(eee_block) + 2)) if eee_block else SOURCE_CAP
    if len(primary) >= budget:
        primary = primary[:budget]
        return (primary + "\n\n" + eee_block)[:SOURCE_CAP] if eee_block else primary
    if eee_block:
        primary = (primary + "\n\n" + eee_block) if primary else eee_block
    rag = "\n\n".join(_rag_contexts(os.path.join(to, "rag")))
    if rag:
        primary += "\n\n[RETRIEVED CONTEXT]\n" + rag
    return primary[:SOURCE_CAP]


def card_fields(card):
    fields = []
    for sec in GOLD_SECTIONS:
        sec_fields = card.get(sec, {})
        if not isinstance(sec_fields, dict):
            continue
        for k, v in sec_fields.items():
            if k in _SECTION_SKIP:
                continue
            path = f"{sec}.{k}"
            if path in JUDGE_SKIP:
                continue
            fields.append({"path": path,
                           "value": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False),
                           "is_ns": is_not_specified(v)})
    return fields


def card_risks(card):
    pr = card.get("possible_risks")
    if not isinstance(pr, list):
        return []
    out = []
    for r in pr:
        if isinstance(r, dict) and r.get("category"):
            out.append(r["category"])
    return out


def prepare(label):
    manifest = _load(os.path.join(GOLD_DIR, "manifest.json"))
    out_dir = f"/tmp/gold/judge_inputs_{label}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'name':18s} {'fields':>6s} {'NS':>3s} {'risks':>5s} {'src_chars':>9s}")
    for entry in manifest["cards"]:
        name = entry["name"]
        run_dir = os.path.join(REPO, os.path.dirname(os.path.dirname(entry["card_path"])))
        card = extract_card(_load(os.path.join(REPO, entry["card_path"])) or {})
        fields = card_fields(card)
        risks = card_risks(card)
        source = assemble_source(run_dir)
        n_ns = sum(f["is_ns"] for f in fields)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({"name": name, "fields": fields, "risks": risks, "source_text": source}, f)
        print(f"{name:18s} {len(fields):>6d} {n_ns:>3d} {len(risks):>5d} {len(source):>9d}")
    print(f"\nInputs -> {out_dir}")
    print("Run the judge workflow over these (JUDGE_PROMPT / JUDGE_SCHEMA), save {name: verdict} json,")
    print(f"then: python scripts/judge_gold_set.py ingest --label {label} --results <result.json>")


def _agg(verdicts):
    fvs = [fv for v in verdicts for fv in v.get("field_verdicts", [])]
    filled = [f for f in fvs if f["status"] != "not_specified"]
    ns = [f for f in fvs if f["status"] == "not_specified"]
    rvs = [rv for v in verdicts for rv in v.get("risk_verdicts", [])]

    def frac(n, d):
        return round(n / d, 3) if d else None

    n_sup = sum(f["status"] == "supported" for f in filled)
    n_eee = sum(f["status"] == "supported_by_eee_only" for f in filled)
    n_unsup = sum(f["status"] == "unsupported" for f in filled)
    n_partial = sum(f["status"] == "partial" for f in filled)
    n_specific = sum(f["specificity"] == "specific" for f in filled)
    # "yes" = legacy v31 two-way label (counts as primary)
    n_missed_primary = sum(f["info_in_source"] in ("yes_primary", "yes") for f in ns)
    n_missed_eee = sum(f["info_in_source"] == "yes_eee_only" for f in ns)
    n_missed = n_missed_primary + n_missed_eee
    n_grounded = sum(r["relevant_and_grounded"] == "yes" for r in rvs)
    return {
        "n_filled": len(filled), "n_ns": len(ns), "n_risks": len(rvs),
        "supported": n_sup, "supported_by_eee_only": n_eee,
        "partial": n_partial, "unsupported": n_unsup,
        "support_rate": frac(n_sup, len(filled)),
        "unsupported_rate": frac(n_unsup, len(filled)),
        "specific_rate": frac(n_specific, len(filled)),
        "ns_missed": n_missed, "ns_missed_primary": n_missed_primary,
        "ns_missed_eee_only": n_missed_eee,
        "ns_correct_abstain": len(ns) - n_missed,
        "completeness_miss_rate": frac(n_missed, len(ns)),
        "completeness_miss_rate_primary": frac(n_missed_primary, len(ns)),
        "risk_grounded_rate": frac(n_grounded, len(rvs)),
    }


_FIELD_STATUSES = {"supported", "supported_by_eee_only", "partial", "unsupported", "not_specified"}
_INFO_VALUES = {"yes_primary", "yes_eee_only", "yes", "no", "na"}  # "yes" = legacy v31


def ingest(label, results_path, judge_model=None, prompt_version=None):
    raw = _load(results_path)
    if isinstance(raw, dict) and "result" in raw and isinstance(raw["result"], dict):
        raw = raw["result"]  # tolerate a saved task-output wrapper
    if isinstance(raw, list):
        raw = {v["name"]: v for v in raw if isinstance(v, dict) and v.get("name")}
    if not isinstance(raw, dict):
        sys.exit(f"unrecognized results format in {results_path}")

    manifest = _load(os.path.join(GOLD_DIR, "manifest.json"))
    order = [c["name"] for c in manifest["cards"]]
    strata = {c["name"]: c["stratum"] for c in manifest["cards"]}
    per_card, missing = {}, []
    for name in order:
        v = raw.get(name)
        if not v:
            missing.append(name)
            continue
        if not v.get("field_verdicts"):
            sys.exit(f"{name}: empty or missing field_verdicts")
        # An off-enum value would silently fall out of every _agg bucket.
        bad = [fv.get("status") for fv in v.get("field_verdicts", [])
               if fv.get("status") not in _FIELD_STATUSES]
        bad += [fv.get("info_in_source") for fv in v.get("field_verdicts", [])
                if fv.get("info_in_source") not in _INFO_VALUES]
        # NS rows must say where the info was (or 'no'); 'na' would silently
        # count as a correct abstention in _agg.
        bad += [f"ns-na:{fv.get('path')}" for fv in v.get("field_verdicts", [])
                if fv.get("status") == "not_specified"
                and fv.get("info_in_source") not in ("yes_primary", "yes_eee_only", "yes", "no")]
        bad += [rv.get("relevant_and_grounded") for rv in v.get("risk_verdicts", [])
                if rv.get("relevant_and_grounded") not in ("yes", "no")]
        if bad:
            sys.exit(f"{name}: invalid enum values in verdicts: {sorted(set(map(str, bad)))}")
        per_card[name] = {"stratum": strata[name],
                          "field_verdicts": v.get("field_verdicts", []),
                          "risk_verdicts": v.get("risk_verdicts", []),
                          "aggregate": _agg([v])}

    out = {"label": label, "n_cards": len(per_card),
           "judge_info": {"model": judge_model, "prompt_version": prompt_version,
                          "judged_on": date.today().isoformat()},
           "aggregate": _agg(list(per_card.values())),
           "per_card": per_card}
    out_path = os.path.join(GOLD_DIR, f"judge_{label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _write_calibration(label, per_card)

    a = out["aggregate"]
    print(f"Judged {len(per_card)}/{len(order)} cards. Missing: {', '.join(missing) or 'none'}\n")
    print(f"  Support      supported={a['supported']} eee_only={a['supported_by_eee_only']} "
          f"partial={a['partial']} unsupported={a['unsupported']}"
          f"  -> support_rate={_pct(a['support_rate'])} unsupported_rate={_pct(a['unsupported_rate'])}")
    print(f"  Specificity  specific_rate={_pct(a['specific_rate'])} over {a['n_filled']} filled fields")
    print(f"  Completeness NS missed={a['ns_missed']}/{a['n_ns']} = {_pct(a['completeness_miss_rate'])}"
          f"  (primary={a['ns_missed_primary']}, eee_only={a['ns_missed_eee_only']})")
    print(f"  Risks        grounded_rate={_pct(a['risk_grounded_rate'])} over {a['n_risks']} risks")
    print(f"\nJudge -> {os.path.relpath(out_path, REPO)}")


def _write_calibration(label, per_card, target=50):
    """Stratified sample of field verdicts with the judge label HIDDEN (for an
    independent human rater study). Judge labels go to a separate key file."""
    rows = []
    for name, c in per_card.items():
        for fv in c["field_verdicts"]:
            rows.append((name, c["stratum"], fv))
    if not rows:
        return
    # deterministic stride sample across the (manifest-ordered) rows
    step = max(1, len(rows) // target)
    sample = rows[::step][:target]

    csv_path = os.path.join(GOLD_DIR, f"calibration_sample_{label}.csv")
    key_path = os.path.join(GOLD_DIR, f"calibration_key_{label}.json")
    key = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card", "stratum", "field_path", "kind", "field_value", "human_label"])
        for name, stratum, fv in sample:
            kind = "not_specified" if fv["status"] == "not_specified" else "filled"
            card = _load_card_value(name, fv["path"])
            w.writerow([name, stratum, fv["path"], kind, (card or "")[:300], ""])
            label_val = fv["info_in_source"] if kind == "not_specified" else fv["status"]
            key.append({"card": name, "field_path": fv["path"], "judge_label": label_val,
                        "specificity": fv.get("specificity")})
    with open(key_path, "w") as f:
        json.dump({"label": label,
                   "rubric": "filled: supported/supported_by_eee_only/partial/unsupported; "
                   "not_specified: info_in_source yes_primary/yes_eee_only/no", "key": key},
                  f, indent=2)
    print(f"Calibration  {len(sample)} fields -> {os.path.relpath(csv_path, REPO)} "
          f"(labels hidden; key in {os.path.relpath(key_path, REPO)})")


def _load_card_value(name, path):
    manifest = _load(os.path.join(GOLD_DIR, "manifest.json"))
    cp = next((c["card_path"] for c in manifest["cards"] if c["name"] == name), None)
    if not cp:
        return ""
    card = extract_card(_load(os.path.join(REPO, cp)) or {})
    sec, _, key = path.partition(".")
    v = card.get(sec, {}).get(key)
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Gold-set faithfulness judge (separate judge model; no pipeline inference).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--label", default="v31")
    i = sub.add_parser("ingest")
    i.add_argument("--label", default="v31")
    i.add_argument("--results", required=True)
    i.add_argument("--judge-model", default=None, help="judge model id recorded in judge_info")
    i.add_argument("--prompt-version", default=JUDGE_PROMPT_VERSION,
                   help="recorded in judge_info; set explicitly when re-ingesting results "
                        "judged with an older prompt")
    args = ap.parse_args()

    if args.cmd == "prepare":
        prepare(args.label)
    elif args.cmd == "ingest":
        ingest(args.label, args.results, args.judge_model, args.prompt_version)


if __name__ == "__main__":
    main()
