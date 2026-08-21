"""Headless batch runner for the S-sample web-grounded adversarial screen.

Ports s_screen_review.js to a reproducible raw-API runner: pinned model,
recorded request params, persisted per-card verdicts, web_search tool loop.

Usage:
  python scripts/run_s_screen.py \
      --config eval/s150/screen/screen_config.json \
      --model claude-sonnet-4-6 \
      --concurrency 2 \
      --max-cost-usd 120 \
      [--only slug1,slug2] [--max-uses 8] [--timeout 600]
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import jsonschema

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Pricing USD per million tokens for models expected here
PRICING_USD_PER_MTOK = {
    "claude-fable-5": {"input": 10.0, "output": 50.0,
                       "cache_read": 1.0, "cache_write": 12.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0,
                          "cache_read": 0.3, "cache_write": 3.75},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0,
                        "cache_read": 0.5, "cache_write": 6.25},
}

# Per web_search request billing
WEB_SEARCH_COST_USD = 0.01

LIMIT_MARKERS = ("rate limit", "usage limit", "overloaded", "429", "quota")
LIMIT_SLEEP_S = 900

# temperature=0 attempted once per run; if BadRequestError names temperature,
# drop it run-wide and record "rejected_by_api_omitted" in run_meta
API_STATE = {"temperature_ok": True}

# Schema byte-equal copy of collect_s_screen.py SCHEMA (contract: keep in sync)
SCHEMA = {
    "type": "object",
    "required": ["card", "real_benchmark", "identity_correct", "paper_assessment",
                 "hf_repo_assessment", "content_accurate", "fabricated_fact",
                 "verdict", "findings", "citations", "reviewing_model", "summary"],
    "properties": {
        "card": {"type": "string"},
        "real_benchmark": {"type": "string"},
        "identity_correct": {"type": ["boolean", "null"]},
        "paper_assessment": {"enum": ["correct", "wrong", "plausible-unverified",
                                      "correctly-none", "missing-should-exist"]},
        "hf_repo_assessment": {"enum": ["correct-kept", "correctly-rejected",
                                        "wrong-kept-CONTAMINATION", "wrongly-rejected-lost-data",
                                        "no-repo-ok", "no-repo-but-one-exists"]},
        "content_accurate": {"type": ["boolean", "null"]},
        "fabricated_fact": {"type": ["boolean", "null"]},
        "verdict": {"enum": ["clean", "minor", "needs-fix"]},
        "findings": {"type": "array", "items": {
            "type": "object",
            "required": ["severity", "category", "field", "issue", "ground_truth"],
            "properties": {
                "severity": {"enum": ["needs-fix", "minor", "note"]},
                "category": {"enum": ["wrong-identity", "wrong-paper", "fabricated-fact",
                                      "wrong-section-splice", "thin", "other"]},
                "field": {"type": "string"},
                "issue": {"type": "string"},
                "ground_truth": {"type": "string"}}}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "reviewing_model": {"type": "string"},
        "summary": {"type": "string"},
    },
}

# Static system prefix: ported rubric from s_screen_review.js (same text for
# every card; cached with cache_control ephemeral to save input token cost)
SYSTEM_PREFIX = (
    "You are an independent, adversarial reviewer of ONE generated AI-benchmark card. "
    "Establish real-world ground truth from the WEB, then check the card against it. "
    "Be skeptical and specific. WEB-VERIFY before calling anything \"fabricated\" -- "
    "some scores/models that look anachronistic are real and current (the live web is "
    "the arbiter, not your priors).\n\n"
    "STEP 1 - READ the inlined card files provided in the user message.\n\n"
    "STEP 2 - Establish GROUND TRUTH from the web (use the web_search tool): "
    "what the benchmark IS, who introduced it, the introducing paper (arxiv id), "
    "the canonical HF repo if any, and a few hard facts (size, metrics, languages, license). "
    "Do NOT trust the card for ground truth.\n\n"
    "STEP 3 - Adversarially check and categorize:\n"
    "  identity_correct: right benchmark, not a same-named other\n"
    "  paper_assessment: 'wrong'=wrong paper (category wrong-paper); "
    "'correctly-none'; 'missing-should-exist'=recall/thin gap\n"
    "  hf_repo_assessment: a kept different entity = wrong-kept-CONTAMINATION; "
    "a rejected repo with no spliced content = correctly-rejected\n"
    "  content_accurate + fabricated_fact: set true for any invented/untraceable concrete fact "
    "-- invented leaderboard, score for a non-existent model, size contradicting the breakdown, "
    "wrong authors -> category fabricated-fact; a right-benchmark one-wrong-detail = "
    "wrong-section-splice; a sparse card = thin\n\n"
    "verdict: clean / minor (presentation/thin/recall, no wrong facts/identity/paper) / "
    "needs-fix (wrong fact, wrong identity, wrong paper, fabrication, contamination).\n\n"
    "Recording requirements (mandatory):\n"
    "  - set citations = the URLs you actually consulted and that your verdict rests on "
    "(never empty unless you used no web sources at all, which itself is a defect)\n"
    "  - set reviewing_model to the exact model string (it will be overwritten server-side)\n\n"
    "End your reply with ONLY the JSON verdict object (no prose before or after it, "
    "no code fences), conforming exactly to this JSON schema:\n"
    "{schema}"
)


def _build_system_prefix():
    return SYSTEM_PREFIX.format(schema=json.dumps(SCHEMA, indent=2))


SYSTEM_PREFIX_RENDERED = _build_system_prefix()


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def _inline_file(path, label, cap=60000):
    """Read a file and return a labelled section, truncated with a note if needed."""
    if not os.path.exists(path):
        return f"[{label}: not present]\n"
    try:
        with open(path) as f:
            text = f.read()
    except Exception as e:
        return f"[{label}: read error: {e}]\n"
    truncated = ""
    if len(text) > cap:
        text = text[:cap]
        truncated = f"\n[TRUNCATED at {cap} chars]"
    return f"[{label}]\n{text}{truncated}\n"


def _build_user_prompt(slug, staged_dir):
    """Inline the staged card files into the user prompt."""
    parts = [f"CARD: \"{slug}\"\n\n"]

    card_path = os.path.join(staged_dir, "benchmarkcard", f"benchmark_card_{slug}.json")
    parts.append(_inline_file(card_path, f"benchmarkcard/benchmark_card_{slug}.json"))

    paper_path = os.path.join(staged_dir, "tool_output", "paper_resolver", "paper-verification.json")
    parts.append(_inline_file(paper_path, "tool_output/paper_resolver/paper-verification.json"))

    hf_path = os.path.join(staged_dir, "tool_output", "hf_verifier", "hf-verification.json")
    parts.append(_inline_file(hf_path, "tool_output/hf_verifier/hf-verification.json"))

    eee_dir = os.path.join(staged_dir, "tool_output", "eee")
    if os.path.isdir(eee_dir):
        for fname in sorted(os.listdir(eee_dir)):
            if fname.endswith(".json"):
                parts.append(_inline_file(
                    os.path.join(eee_dir, fname),
                    f"tool_output/eee/{fname}"))
    else:
        parts.append("[tool_output/eee/: not present]\n")

    return "".join(parts)


def extract_json(text):
    """Try to extract a JSON object from model text output."""
    if not text:
        return None
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


def validate_verdict(verdict):
    """Validate against SCHEMA. Returns list of problems (empty = ok)."""
    problems = []
    try:
        jsonschema.validate(verdict, SCHEMA)
    except jsonschema.ValidationError as e:
        problems.append(f"schema: {e.message[:300]}")
    return problems


def estimate_cost(usage, model, web_searches=0):
    """Estimated USD for one response, including any web_search requests."""
    p = PRICING_USD_PER_MTOK[model]
    cost = (usage.get("input_tokens", 0) * p["input"]
            + usage.get("output_tokens", 0) * p["output"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6
    cost += web_searches * WEB_SEARCH_COST_USD
    return cost


def make_client(args):
    """Load .env and construct the Anthropic client."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"), override=False)
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not found (env or .env)")
    if args.model not in PRICING_USD_PER_MTOK:
        sys.exit(f"no pricing entry for model {args.model!r}; add it to PRICING_USD_PER_MTOK")
    return anthropic, anthropic.Anthropic(timeout=float(args.timeout), max_retries=2)


def _accumulate_usage(total, u):
    total["input_tokens"] += u.get("input_tokens", 0)
    total["output_tokens"] += u.get("output_tokens", 0)
    total["cache_read_input_tokens"] += u.get("cache_read_input_tokens", 0)
    total["cache_creation_input_tokens"] += u.get("cache_creation_input_tokens", 0)


def screen_one(slug, staged_dir, out_dir, args, anthropic_mod, client):
    """Run the web-grounded screen for one card. Returns a meta dict."""
    user_prompt = _build_user_prompt(slug, staged_dir)

    meta = {
        "slug": slug,
        "retries": 0,
        "limit_sleeps": 0,
        "cost_usd": 0.0,
        "web_searches": 0,
        "wall_s": 0.0,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }

    tools = [{"type": "web_search_20260209", "name": "web_search",
               "max_uses": args.max_uses}]

    # initial user message
    base_messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    attempt = 0
    current_messages = list(base_messages)
    verdict = None
    problems = []

    t0_outer = time.time()

    while attempt <= 1:  # at most one retry-feedback attempt
        t0 = time.time()
        kwargs = dict(
            model=args.model,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PREFIX_RENDERED,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=tools,
            messages=current_messages,
        )
        if API_STATE["temperature_ok"]:
            kwargs["temperature"] = 0.0

        resp = None
        try:
            resp = client.messages.create(**kwargs)
        except anthropic_mod.BadRequestError as e:
            if API_STATE["temperature_ok"] and "temperature" in str(e).lower():
                API_STATE["temperature_ok"] = False
                continue  # retry without temperature, does not consume an attempt
            problems = [f"api bad request: {str(e)[:300]}"]
            break
        except anthropic_mod.RateLimitError:
            meta["limit_sleeps"] += 1
            time.sleep(LIMIT_SLEEP_S)
            # limit sleeps do not consume an attempt
            continue
        except anthropic_mod.APIStatusError as e:
            if getattr(e, "status_code", None) == 529 or "overloaded" in str(e).lower():
                meta["limit_sleeps"] += 1
                time.sleep(LIMIT_SLEEP_S)
                continue
            problems = [f"api status {getattr(e, 'status_code', '?')}: {str(e)[:300]}"]
            attempt += 1
            continue
        except anthropic_mod.APIConnectionError as e:
            problems = [f"api connection: {str(e)[:300]}"]
            attempt += 1
            continue

        # accumulate usage and cost across pause_turn continuations
        continuations = 0
        while resp.stop_reason == "pause_turn" and continuations < 8:
            u = resp.usage
            usage_dict = {
                "input_tokens": u.input_tokens or 0,
                "output_tokens": u.output_tokens or 0,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            }
            searches_this = getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0
            meta["web_searches"] += searches_this
            _accumulate_usage(meta["usage"], usage_dict)
            meta["cost_usd"] = round(
                meta["cost_usd"] + estimate_cost(usage_dict, args.model, searches_this), 6)

            # append assistant content verbatim (including encrypted web search blocks)
            current_messages = current_messages + [
                {"role": "assistant", "content": resp.content}
            ]
            # re-request to let the tool loop continue
            try:
                resp = client.messages.create(**kwargs | {"messages": current_messages})
            except Exception as e:
                problems = [f"pause_turn continuation error: {str(e)[:300]}"]
                resp = None
                break
            continuations += 1

        if resp is None:
            attempt += 1
            continue

        # final response accounting
        u = resp.usage
        usage_dict = {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        searches_this = getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0
        meta["web_searches"] += searches_this
        _accumulate_usage(meta["usage"], usage_dict)
        meta["cost_usd"] = round(
            meta["cost_usd"] + estimate_cost(usage_dict, args.model, searches_this), 6)
        meta["stop_reason"] = resp.stop_reason
        meta["request_id"] = getattr(resp, "_request_id", None)

        if resp.stop_reason == "refusal":
            problems = ["stop_reason=refusal (safety classifier)"]
            break  # no fallback model
        elif resp.stop_reason == "max_tokens":
            problems = ["truncated at max_tokens=16000"]
            attempt += 1
            continue

        result_text = "".join(b.text for b in resp.content if hasattr(b, "text") and b.type == "text")
        verdict = extract_json(result_text)
        if verdict is None:
            problems = ["no parseable JSON verdict in API response"]
            attempt += 1
            # feedback retry: rebuild from base messages with correction note
            current_messages = list(base_messages) + [
                {"role": "assistant", "content": result_text or ""},
                {"role": "user", "content": (
                    "Your previous answer did not contain a valid JSON verdict. "
                    "Return ONLY the JSON verdict object, no prose, no code fences."
                )},
            ]
            continue

        problems = validate_verdict(verdict)

        # enforce citations non-empty
        if not problems and not verdict.get("citations"):
            problems = ["citations array is empty; model must cite the URLs used"]

        if not problems:
            # overwrite reviewing_model with the authoritative --model string
            verdict["reviewing_model"] = args.model
            break

        # one retry-feedback pass on schema/citations failure
        attempt += 1
        error_summary = "; ".join(problems)[:1500]
        current_messages = list(base_messages) + [
            {"role": "assistant", "content": result_text},
            {"role": "user", "content": (
                f"Your previous verdict was invalid: {error_summary}\n"
                "Return a corrected JSON verdict object only."
            )},
        ]

    meta["wall_s"] = round(time.time() - t0_outer, 1)

    if not problems and verdict is not None:
        verdicts_dir = os.path.join(out_dir, "verdicts")
        os.makedirs(verdicts_dir, exist_ok=True)
        with open(os.path.join(verdicts_dir, f"{slug}.json"), "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        meta["ok"] = True
        return meta

    meta["ok"] = False
    meta["last_problems"] = problems
    failed_dir = os.path.join(out_dir, "failed")
    os.makedirs(failed_dir, exist_ok=True)
    with open(os.path.join(failed_dir, f"{slug}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(
        description="Headless batch web-screen runner (API transport, web_search tool).")
    ap.add_argument("--config", required=True,
                    help="screen_config.json from gen_screen_config.py")
    ap.add_argument("--model", required=True,
                    help="Anthropic model id (e.g. claude-sonnet-4-6)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-cost-usd", type=float, default=120.0)
    ap.add_argument("--only", default=None, help="comma-separated slugs")
    ap.add_argument("--max-uses", type=int, default=8,
                    help="max web_search calls per card")
    ap.add_argument("--timeout", type=int, default=600,
                    help="SDK request timeout in seconds")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from check_frozen import check
    check()

    cfg_path = os.path.join(REPO, args.config)
    if not os.path.exists(cfg_path):
        sys.exit(f"config not found: {cfg_path}")
    cfg = _read_json(cfg_path)

    cfg_md5 = hashlib.md5(open(cfg_path, "rb").read()).hexdigest()
    prompt_prefix_md5 = _md5(SYSTEM_PREFIX_RENDERED)

    all_cards = {c["slug"]: c["staged_dir"] for c in cfg["cards"]}
    if args.only:
        slugs = [s.strip() for s in args.only.split(",")]
        unknown = [s for s in slugs if s not in all_cards]
        if unknown:
            sys.exit(f"unknown slugs (not in config): {unknown}")
    else:
        slugs = list(all_cards)

    out_dir = cfg["outdir"]
    os.makedirs(out_dir, exist_ok=True)
    verdicts_dir = os.path.join(out_dir, "verdicts")

    # resume: skip slugs whose verdict file already exists
    todo = []
    done = []
    for slug in slugs:
        vp = os.path.join(verdicts_dir, f"{slug}.json")
        if os.path.exists(vp):
            done.append(slug)
        else:
            todo.append(slug)

    print(f"{len(done)} already screened, {len(todo)} to run "
          f"(model={args.model}, concurrency={args.concurrency})")

    anthropic_mod, client = make_client(args)

    import anthropic as _ant
    sdk_version = _ant.__version__

    started = datetime.now(timezone.utc).isoformat()
    metas = []
    total_cost = 0.0
    aborted = None

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(screen_one, slug, all_cards[slug], out_dir, args,
                          anthropic_mod, client): slug
                for slug in todo
            }
            for fut in concurrent.futures.as_completed(futs):
                m = fut.result()
                metas.append(m)
                total_cost += m.get("cost_usd") or 0.0
                status = "ok" if m.get("ok") else "FAILED"
                print(f"  {m['slug']:30s} {status} "
                      f"wall={m.get('wall_s')}s "
                      f"cost=${m.get('cost_usd'):.4f} "
                      f"searches={m.get('web_searches', 0)}")
                if total_cost > args.max_cost_usd:
                    aborted = (f"cost tripwire: est ${total_cost:.2f} exceeds "
                               f"--max-cost-usd ${args.max_cost_usd:.2f}")
                    print(f"\nABORT {aborted} (completed verdicts persist; resumable)")
                    for f2 in futs:
                        f2.cancel()
                    break

    ok_slugs = done + [m["slug"] for m in metas if m.get("ok")]
    failed_slugs = [m["slug"] for m in metas if not m.get("ok")]

    # build run_meta
    run_meta_path = os.path.join(out_dir, "run_meta.json")
    prior = _read_json(run_meta_path) if os.path.exists(run_meta_path) else {"batches": []}

    totals = prior.get("total_usage") or {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }
    total_searches = prior.get("total_web_searches") or 0
    for m in metas:
        for k in totals:
            totals[k] += m.get("usage", {}).get(k, 0)
        total_searches += m.get("web_searches", 0)

    prior.update({
        "transport": "anthropic-api-sync",
        "model": args.model,
        "sdk_version": sdk_version,
        "request_params": {
            "max_tokens": 16000,
            "temperature": 0.0 if API_STATE["temperature_ok"] else "rejected_by_api_omitted",
            "tools": [{"type": "web_search_20260209", "name": "web_search",
                       "max_uses": args.max_uses}],
            "stream": False,
            "sdk_timeout_s": args.timeout,
            "sdk_max_retries": 2,
        },
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK[args.model],
        "web_search_cost_usd_per_request": WEB_SEARCH_COST_USD,
        "prompt_prefix_md5": prompt_prefix_md5,
        "config_path": args.config,
        "config_md5": cfg_md5,
        "total_usage": totals,
        "total_web_searches": total_searches,
        "total_cost_est_usd": round(
            (prior.get("total_cost_est_usd") or 0.0) + total_cost, 4),
        "max_cost_usd": args.max_cost_usd,
    })
    if aborted:
        prior["aborted"] = aborted
    else:
        prior.pop("aborted", None)

    prior.setdefault("batches", []).append({
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "cards": metas,
    })

    with open(run_meta_path, "w") as f:
        json.dump(prior, f, indent=2)

    if failed_slugs:
        print(f"\nFAILED: {failed_slugs} (details in {os.path.relpath(out_dir, REPO)}/failed/)")
    print(f"\ntotal cost est: ${total_cost:.4f}  searches: {sum(m.get('web_searches', 0) for m in metas)}")

    if aborted:
        sys.exit(1)
    if failed_slugs:
        sys.exit(1)


if __name__ == "__main__":
    main()
