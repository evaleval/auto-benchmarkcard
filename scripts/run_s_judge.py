"""Headless batch runner for the frozen faithfulness judge.

Applies the frozen JUDGE_PROMPT / JUDGE_SCHEMA (imported from
scripts/judge_gold_set.py, never modified) to prepared judge inputs, one call
per card, model pinned to the judge_v31b judge_info id. Two transports:
  cli  one `claude -p` subprocess per card; no API key, uses the
       authenticated claude CLI; the agent reads the input file via tools
  api  one synchronous Anthropic Messages call per card, the prepared input
       file inlined verbatim (the exact md5-guarded bytes); ANTHROPIC_API_KEY
       from env or .env; requires --input-mode json; cumulative estimated
       cost guarded by --max-cost-usd
Resumable: a card is skipped when out/verdicts/<name>.json exists and
re-validates.

Input modes (D3-gated choice, recorded in run_meta):
  json  the original prepared input file (single-line JSON; agent reads via
        cat because the Read tool truncates very long lines)
  text  a content-identical text rendering (out/rendered/<name>.md) with the
        source_text newlines restored

Guard (orchestrator amendment 2): inputs_md5.json is re-verified at start,
on every resume, and after every completed card; any mismatch is a hard stop.

Usage:
  python scripts/run_s_judge.py --inputs /tmp/gold/judge_inputs_s150 \
      --inputs-md5 eval/s150/judge/inputs_md5.json --out eval/s150/judge \
      [--transport cli|api] [--max-cost-usd 400] [--max-tokens 16000] \
      [--only a,b,c] [--limit N] [--concurrency 2] [--input-mode json] \
      [--drift-compare gold_set/judge_v31b.json] [--partial-ok]
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import jsonschema

from check_frozen import check
from s_judge import load_frozen

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_MODEL = "claude-fable-5"  # judge_v31b.json judge_info pin
LIMIT_MARKERS = ("rate limit", "usage limit", "overloaded", "429", "quota")
LIMIT_SLEEP_S = 900

# USD per million tokens; a model missing here hard-stops the api transport
# (the cost tripwire must never silently count zero)
PRICING_USD_PER_MTOK = {
    "claude-fable-5": {"input": 10.0, "output": 50.0,
                       "cache_read": 1.0, "cache_write": 12.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0,
                          "cache_read": 0.3, "cache_write": 3.75},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0,
                        "cache_read": 0.5, "cache_write": 6.25},
}

# temperature=0 is attempted once per run; flipped off run-wide if the API
# rejects sampling params for the pinned model (recorded in run_meta)
API_STATE = {"temperature_ok": True}

OUTPUT_APPENDIX = (
    "\n\nOutput format: return ONLY the JSON verdict object (no prose, no code fences), "
    "conforming exactly to this JSON schema:\n{schema}"
)

INLINE_APPENDIX = (
    "\n\nThe complete contents of that file are reproduced verbatim below between the "
    "markers; use them directly instead of reading from disk.\n"
    "[BEGIN INPUT FILE]\n{raw}\n[END INPUT FILE]"
)


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(inputs_dir, md5_manifest, names):
    """Hard-stop if any prepared input no longer matches its recorded md5."""
    for name in names:
        rec = md5_manifest["cards"].get(name)
        p = os.path.join(inputs_dir, f"{name}.json")
        if rec is None:
            sys.exit(f"inputs guard: {name} not in inputs_md5.json")
        if not os.path.exists(p):
            sys.exit(f"inputs guard: missing input {p}")
        if _md5_file(p) != rec["md5"]:
            sys.exit(f"inputs guard: md5 mismatch for {name} - inputs changed under us")


def render_text(input_json, dest):
    """Content-identical text rendering: same fields/risks, source newlines restored."""
    with open(dest, "w") as f:
        f.write(f"name: {input_json['name']}\n\n")
        f.write("fields (JSON array):\n")
        json.dump(input_json["fields"], f, indent=2, ensure_ascii=False)
        f.write("\n\nrisks (JSON array):\n")
        json.dump(input_json["risks"], f, indent=2, ensure_ascii=False)
        f.write("\n\nsource_text:\n")
        f.write(input_json["source_text"])
        f.write("\n")


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def validate_verdict(verdict, input_json, schema):
    """Schema + coverage + the frozen ingest cross-rules. Returns list of problems."""
    problems = []
    try:
        jsonschema.validate(verdict, schema)
    except jsonschema.ValidationError as e:
        problems.append(f"schema: {e.message[:300]}")
        return problems
    if verdict.get("name") != input_json["name"]:
        problems.append(f"name mismatch: {verdict.get('name')!r}")
    want_path_counts = Counter(f["path"] for f in input_json["fields"])
    got_path_counts = Counter(fv["path"] for fv in verdict["field_verdicts"])
    want_paths = set(want_path_counts)
    got_paths = set(got_path_counts)
    duplicate_paths = sorted(path for path, n in got_path_counts.items() if n != 1)
    if duplicate_paths:
        problems.append(f"duplicate field verdicts: {duplicate_paths}")
    if want_paths - got_paths:
        problems.append(f"missing field verdicts: {sorted(want_paths - got_paths)}")
    if got_paths - want_paths:
        problems.append(f"unknown field paths: {sorted(got_paths - want_paths)}")
    want_risk_counts = Counter(input_json["risks"])
    got_risk_counts = Counter(rv["category"] for rv in verdict["risk_verdicts"])
    if want_risk_counts != got_risk_counts:
        problems.append(
            "risk coverage mismatch: "
            f"want {dict(sorted(want_risk_counts.items()))}, "
            f"got {dict(sorted(got_risk_counts.items()))}"
        )
    for fv in verdict["field_verdicts"]:
        if fv["status"] == "not_specified" and fv["info_in_source"] not in (
                "yes_primary", "yes_eee_only", "no"):
            problems.append(f"ns-na rule: {fv['path']} info_in_source={fv['info_in_source']}")
    return problems


def judge_one(name, inputs_dir, out_dir, jgs, args):
    """Run the judge for one card. Returns a meta dict."""
    input_path = os.path.join(inputs_dir, f"{name}.json")
    input_json = _read_json(input_path)
    if args.input_mode == "text":
        rendered_dir = os.path.join(out_dir, "rendered")
        os.makedirs(rendered_dir, exist_ok=True)
        judge_path = os.path.join(rendered_dir, f"{name}.md")
        render_text(input_json, judge_path)
        allowed = ["Read", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)"]
    else:
        judge_path = input_path
        allowed = ["Read", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)",
                   "Bash(grep:*)"]

    prompt = jgs.JUDGE_PROMPT.format(input_path=os.path.abspath(judge_path))
    prompt += OUTPUT_APPENDIX.format(schema=json.dumps(jgs.JUDGE_SCHEMA))

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLAUDE", "CLAUDECODE"))}
    meta = {"name": name, "retries": 0, "limit_sleeps": 0}
    attempt_prompt = prompt
    for attempt in range(args.max_retries + 1):
        t0 = time.time()
        cmd = [args.claude_bin, "-p", "--output-format", "json",
               "--model", args.model, "--max-turns", "25",
               "--allowedTools", *allowed]
        try:
            proc = subprocess.run(cmd, input=attempt_prompt, capture_output=True,
                                  text=True, timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            meta["retries"] = attempt + 1
            continue
        wall = time.time() - t0
        blob = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and any(m in blob.lower() for m in LIMIT_MARKERS):
            meta["limit_sleeps"] += 1
            time.sleep(LIMIT_SLEEP_S)
            continue  # does not consume a retry
        envelope = extract_json(proc.stdout or "")
        result_text = None
        if isinstance(envelope, dict) and "result" in envelope:
            result_text = envelope.get("result")
            meta["cost_usd"] = envelope.get("total_cost_usd")
            meta["num_turns"] = envelope.get("num_turns")
            meta["session_id"] = envelope.get("session_id")
            meta["cli_model_usage"] = sorted((envelope.get("modelUsage") or {}).keys())
        verdict = extract_json(result_text) if isinstance(result_text, str) else (
            envelope if isinstance(envelope, dict) and "field_verdicts" in envelope else None)
        problems = validate_verdict(verdict, input_json, jgs.JUDGE_SCHEMA) if verdict \
            else [f"no parseable verdict (rc={proc.returncode}, stderr={proc.stderr[:200]!r})"]
        meta["wall_s"] = round(wall, 1)
        if not problems:
            os.makedirs(os.path.join(out_dir, "verdicts"), exist_ok=True)
            with open(os.path.join(out_dir, "verdicts", f"{name}.json"), "w") as f:
                json.dump(verdict, f, indent=2, ensure_ascii=False)
            meta["ok"] = True
            return meta
        meta["retries"] = attempt + 1
        meta["last_problems"] = problems
        attempt_prompt = (prompt + "\n\nYour previous answer was invalid: "
                          + "; ".join(problems)[:1500]
                          + "\nReturn a corrected verdict covering every field and risk.")
    os.makedirs(os.path.join(out_dir, "failed"), exist_ok=True)
    with open(os.path.join(out_dir, "failed", f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    meta["ok"] = False
    return meta


def make_api_client(args):
    """Anthropic SDK client for --transport api. Lazy imports keep cli path clean."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"), override=False)
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--transport api requires ANTHROPIC_API_KEY (env or .env)")
    if args.model not in PRICING_USD_PER_MTOK:
        sys.exit(f"no pricing entry for model {args.model}; add it to PRICING_USD_PER_MTOK")
    # explicit timeout also lifts the SDK's non-streaming large-max_tokens guard
    return anthropic, anthropic.Anthropic(timeout=float(args.timeout), max_retries=2)


def estimate_cost(usage, model):
    """Estimated USD for one response's token usage, from the pinned pricing table."""
    p = PRICING_USD_PER_MTOK[model]
    return (usage["input_tokens"] * p["input"]
            + usage["output_tokens"] * p["output"]
            + usage["cache_read_input_tokens"] * p["cache_read"]
            + usage["cache_creation_input_tokens"] * p["cache_write"]) / 1e6


def run_card_api(name, inputs_dir, out_dir, jgs, args, anthropic_mod, client):
    """API-transport twin of judge_one: one Messages call per attempt, same
    validation, retry-feedback, and limit-sleep semantics as the cli path."""
    input_path = os.path.join(inputs_dir, f"{name}.json")
    input_json = _read_json(input_path)
    with open(input_path) as f:
        raw = f.read()  # the exact md5-guarded bytes the cli agent would read
    prompt = jgs.JUDGE_PROMPT.format(input_path=os.path.abspath(input_path))
    prompt += INLINE_APPENDIX.format(raw=raw)
    prompt += OUTPUT_APPENDIX.format(schema=json.dumps(jgs.JUDGE_SCHEMA))

    meta = {"name": name, "retries": 0, "limit_sleeps": 0, "cost_usd": 0.0,
            "usage": {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
    attempt_prompt = prompt
    attempt = 0
    while attempt <= args.max_retries:
        t0 = time.time()
        kwargs = dict(model=args.model, max_tokens=args.max_tokens,
                      messages=[{"role": "user", "content": attempt_prompt}])
        if API_STATE["temperature_ok"]:
            kwargs["temperature"] = 0.0
        try:
            resp = client.messages.create(**kwargs)
        except anthropic_mod.BadRequestError as e:
            if API_STATE["temperature_ok"] and "temperature" in str(e).lower():
                API_STATE["temperature_ok"] = False
                continue  # drop temperature run-wide, does not consume an attempt
            meta["retries"] = attempt + 1
            meta["last_problems"] = [f"api bad request: {str(e)[:300]}"]
            attempt += 1
            continue
        except anthropic_mod.RateLimitError:
            meta["limit_sleeps"] += 1
            time.sleep(LIMIT_SLEEP_S)
            attempt += 1  # mirrors the cli limit path: an attempt, not a retry
            continue
        except anthropic_mod.APIStatusError as e:
            if getattr(e, "status_code", None) == 529 or "overloaded" in str(e).lower():
                meta["limit_sleeps"] += 1
                time.sleep(LIMIT_SLEEP_S)
                attempt += 1
                continue
            meta["retries"] = attempt + 1
            meta["last_problems"] = [f"api status {getattr(e, 'status_code', '?')}: {str(e)[:300]}"]
            attempt += 1
            continue
        except anthropic_mod.APIConnectionError as e:  # includes APITimeoutError
            meta["retries"] = attempt + 1
            meta["last_problems"] = [f"api connection: {str(e)[:300]}"]
            attempt += 1
            continue
        meta["wall_s"] = round(time.time() - t0, 1)
        u = resp.usage
        usage = {"input_tokens": u.input_tokens or 0,
                 "output_tokens": u.output_tokens or 0,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        for k, v in usage.items():
            meta["usage"][k] += v
        meta["cost_usd"] = round(meta["cost_usd"] + estimate_cost(usage, args.model), 4)
        meta["stop_reason"] = resp.stop_reason
        meta["request_id"] = getattr(resp, "_request_id", None)
        if resp.stop_reason == "refusal":
            # no fallback model: that would break the judge_v31b pin
            verdict, problems = None, ["stop_reason=refusal (safety classifier)"]
        elif resp.stop_reason == "max_tokens":
            verdict, problems = None, [f"truncated at max_tokens={args.max_tokens}"]
        else:
            result_text = "".join(b.text for b in resp.content if b.type == "text")
            verdict = extract_json(result_text)
            problems = validate_verdict(verdict, input_json, jgs.JUDGE_SCHEMA) if verdict \
                else ["no parseable verdict in API response"]
        if not problems:
            os.makedirs(os.path.join(out_dir, "verdicts"), exist_ok=True)
            with open(os.path.join(out_dir, "verdicts", f"{name}.json"), "w") as f:
                json.dump(verdict, f, indent=2, ensure_ascii=False)
            meta["ok"] = True
            return meta
        meta["retries"] = attempt + 1
        meta["last_problems"] = problems
        attempt_prompt = (prompt + "\n\nYour previous answer was invalid: "
                          + "; ".join(problems)[:1500]
                          + "\nReturn a corrected verdict covering every field and risk.")
        attempt += 1
    os.makedirs(os.path.join(out_dir, "failed"), exist_ok=True)
    with open(os.path.join(out_dir, "failed", f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    meta["ok"] = False
    return meta


def drift_compare(out_dir, ref_path, names):
    ref = _read_json(os.path.join(REPO, ref_path))
    rows, agree_total, n_total = [], 0, 0
    pos_total, pos_flipped = 0, 0
    for name in names:
        vp = os.path.join(out_dir, "verdicts", f"{name}.json")
        if not os.path.exists(vp) or name not in ref["per_card"]:
            continue
        new = {f["path"]: f["status"] for f in _read_json(vp)["field_verdicts"]}
        old = {f["path"]: f["status"] for f in ref["per_card"][name]["field_verdicts"]}
        common = sorted(set(new) & set(old))
        agree = sum(new[p] == old[p] for p in common)
        flips = [(p, old[p], new[p]) for p in common if new[p] != old[p]]
        pos = [p for p in common if old[p] == "unsupported"]
        flipped_pos = [p for p in pos if new[p] != "unsupported"]
        agree_total += agree
        n_total += len(common)
        pos_total += len(pos)
        pos_flipped += len(flipped_pos)
        rows.append((name, agree, len(common), flips, flipped_pos))
    print("\nDrift vs", ref_path)
    for name, agree, n, flips, flipped_pos in rows:
        print(f"  {name:20s} {agree}/{n} agree"
              + (f"  pos-flips: {flipped_pos}" if flipped_pos else ""))
        for p, o, nw in flips:
            print(f"    {p}: {o} -> {nw}")
    if n_total:
        print(f"  OVERALL field-status agreement {agree_total}/{n_total} "
              f"= {100 * agree_total / n_total:.1f}%")
        print(f"  ref-unsupported positives present: {pos_total}, flipped: {pos_flipped}")
    return {"agreement": round(agree_total / n_total, 4) if n_total else None,
            "n_common": n_total, "pos_present": pos_total, "pos_flipped": pos_flipped}


def main():
    ap = argparse.ArgumentParser(description="Headless batch judge (frozen prompt/schema; cli or api transport).")
    ap.add_argument("--inputs", required=True, help="prepared judge inputs dir")
    ap.add_argument("--inputs-md5", required=True, help="inputs_md5.json written by s_judge prepare")
    ap.add_argument("--out", required=True, help="output dir (verdicts/, failed/, run_meta.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--transport", choices=["cli", "api"], default="cli")
    ap.add_argument("--max-cost-usd", type=float, default=400.0,
                    help="api transport: abort when cumulative estimated cost exceeds this")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="api transport: response budget (thinking bills against it)")
    ap.add_argument("--input-mode", choices=["json", "text"], default="json")
    ap.add_argument("--only", default=None, help="comma-separated card names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--claude-bin", default=shutil.which("claude") or "claude")
    ap.add_argument("--label", default=None, help="write results_<label>.json when complete")
    ap.add_argument("--partial-ok", action="store_true")
    ap.add_argument("--drift-compare", default=None, help="reference judge_<label>.json")
    args = ap.parse_args()
    check()

    anthropic_mod = client = None
    if args.transport == "api":
        if args.input_mode != "json":
            sys.exit("--transport api supports --input-mode json only "
                     "(the prepared file is inlined verbatim)")
        anthropic_mod, client = make_api_client(args)

    inputs_dir = os.path.abspath(args.inputs)
    out_dir = os.path.abspath(os.path.join(REPO, args.out))
    os.makedirs(out_dir, exist_ok=True)
    md5_manifest = _read_json(os.path.join(REPO, args.inputs_md5))

    all_names = sorted(md5_manifest["cards"])
    names = [n.strip() for n in args.only.split(",")] if args.only else all_names
    unknown = [n for n in names if n not in md5_manifest["cards"]]
    if unknown:
        sys.exit(f"unknown cards (not in inputs_md5): {unknown}")
    if args.limit:
        names = names[:args.limit]

    verify_inputs(inputs_dir, md5_manifest, names)

    jgs = load_frozen()
    todo, done = [], []
    for name in names:
        vp = os.path.join(out_dir, "verdicts", f"{name}.json")
        if os.path.exists(vp):
            v = _read_json(vp)
            if not validate_verdict(v, _read_json(os.path.join(inputs_dir, f"{name}.json")),
                                    jgs.JUDGE_SCHEMA):
                done.append(name)
                continue
        todo.append(name)
    print(f"{len(done)} already judged, {len(todo)} to run "
          f"(mode={args.input_mode}, model={args.model}, concurrency={args.concurrency})")

    started = datetime.now(timezone.utc).isoformat()
    metas = []
    total_cost = 0.0
    aborted = None
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            if args.transport == "api":
                futs = {ex.submit(run_card_api, n, inputs_dir, out_dir, jgs, args,
                                  anthropic_mod, client): n for n in todo}
            else:
                futs = {ex.submit(judge_one, n, inputs_dir, out_dir, jgs, args): n
                        for n in todo}
            for fut in concurrent.futures.as_completed(futs):
                m = fut.result()
                metas.append(m)
                verify_inputs(inputs_dir, md5_manifest, [m["name"]])
                total_cost += m.get("cost_usd") or 0.0
                status = "ok" if m.get("ok") else "FAILED"
                print(f"  {m['name']:24s} {status} wall={m.get('wall_s')}s "
                      f"cost=${m.get('cost_usd')} turns={m.get('num_turns')} "
                      f"retries={m['retries']}")
                if args.transport == "api" and total_cost > args.max_cost_usd:
                    aborted = (f"cost tripwire: est ${total_cost:.2f} exceeds "
                               f"--max-cost-usd ${args.max_cost_usd:.2f}")
                    print(f"\nABORT {aborted} (completed verdicts persist; resumable)")
                    for f2 in futs:
                        f2.cancel()
                    break
    verify_inputs(inputs_dir, md5_manifest, names)

    ok = done + [m["name"] for m in metas if m.get("ok")]
    failed = [m["name"] for m in metas if not m.get("ok")]

    run_meta_path = os.path.join(out_dir, "run_meta.json")
    prior = _read_json(run_meta_path) if os.path.exists(run_meta_path) else {"batches": []}
    prior["model"] = args.model
    prior["transport"] = args.transport
    if args.transport == "api":
        prior["execution"] = "anthropic-api-sync"
        prior["sdk_version"] = anthropic_mod.__version__
        prior["request_params"] = {
            "model": args.model, "max_tokens": args.max_tokens,
            "temperature": 0.0 if API_STATE["temperature_ok"] else "rejected_by_api_omitted",
            "thinking": "omitted (always-on for this model)",
            "stream": False, "sdk_timeout_s": args.timeout, "sdk_max_retries": 2}
        prior["pricing_usd_per_mtok"] = PRICING_USD_PER_MTOK[args.model]
        prior["inline_appendix_md5"] = hashlib.md5(INLINE_APPENDIX.encode()).hexdigest()
        totals = prior.get("total_usage") or {"input_tokens": 0, "output_tokens": 0,
                                              "cache_read_input_tokens": 0,
                                              "cache_creation_input_tokens": 0}
        for m in metas:
            for k in totals:
                totals[k] += m.get("usage", {}).get(k, 0)
        prior["total_usage"] = totals
        prior["total_cost_est_usd"] = round(
            (prior.get("total_cost_est_usd") or 0.0) + total_cost, 4)
        prior["max_cost_usd"] = args.max_cost_usd
    else:
        prior["execution"] = "claude-cli-headless"
        prior["claude_version"] = subprocess.run(
            [args.claude_bin, "--version"], capture_output=True, text=True).stdout.strip()
    prior["prompt_md5"] = hashlib.md5(jgs.JUDGE_PROMPT.encode()).hexdigest()
    prior["schema_md5"] = hashlib.md5(
        json.dumps(jgs.JUDGE_SCHEMA, sort_keys=True).encode()).hexdigest()
    prior["prompt_version"] = jgs.JUDGE_PROMPT_VERSION
    prior["input_mode"] = args.input_mode
    if aborted:
        prior["aborted"] = aborted
    else:
        prior.pop("aborted", None)
    prior["batches"].append({"started": started,
                             "finished": datetime.now(timezone.utc).isoformat(),
                             "cards": metas})
    drift = None
    if args.drift_compare:
        drift = drift_compare(out_dir, args.drift_compare, ok)
        prior["drift"] = {"ref": args.drift_compare, **drift}
    with open(run_meta_path, "w") as f:
        json.dump(prior, f, indent=2)
    if aborted:
        sys.exit(1)

    if failed:
        print(f"\nFAILED: {failed} (details in {os.path.relpath(out_dir, REPO)}/failed/)")
    if args.label:
        missing = [n for n in names if n not in ok]
        if missing and not args.partial_ok:
            print(f"not writing results_{args.label}.json: {len(missing)} missing")
        else:
            results = {n: _read_json(os.path.join(out_dir, "verdicts", f"{n}.json"))
                       for n in ok}
            rp = os.path.join(out_dir, f"results_{args.label}.json")
            with open(rp, "w") as f:
                json.dump(results, f)
            print(f"results -> {os.path.relpath(rp, REPO)} ({len(results)} cards)")
    if failed and not args.partial_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
