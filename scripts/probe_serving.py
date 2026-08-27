"""Probe the pinned composer serving route before a run.

Makes N tiny json_schema calls through the same OpenRouterEngine the composer uses and
fails loudly when the provider or quantization drifts from the pin, when structured
decoding breaks, or when any call errors. Run it immediately before every launch; a
green window hours earlier proves nothing (2026-08-23 lesson).

Usage:
  OPENROUTER_MODEL_ID=deepseek/deepseek-v4-flash OPENROUTER_PROVIDER_ORDER=Baidu \
      OPENROUTER_QUANTIZATION=fp8 .venv/bin/python scripts/probe_serving.py [-n 3]
Defaults for the env pins are the recorded phase 3 configuration, so a bare invocation
probes exactly the pinned route.
"""

import argparse
import json
import os
import sys
import time

PIN_DEFAULTS = {
    "OPENROUTER_MODEL_ID": "deepseek/deepseek-v4-flash",
    "OPENROUTER_PROVIDER_ORDER": "Baidu",
    "OPENROUTER_QUANTIZATION": "fp8",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["quotes"],
}

PROMPT = (
    "Return a JSON object with a key 'quotes' holding exactly three short verbatim "
    "quotes copied from this text: The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=3, help="number of probe calls")
    args = ap.parse_args()

    for key, val in PIN_DEFAULTS.items():
        os.environ.setdefault(key, val)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from auto_benchmarkcard.llm_handler import LLMHandler

    expected_provider = os.environ["OPENROUTER_PROVIDER_ORDER"].split(",")[0].strip()
    handler = LLMHandler(
        engine_type="openrouter",
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        parameters={"temperature": 0.15, "max_completion_tokens": 512},
    )
    print(f"probing {handler.engine.model_id} via {expected_provider} "
          f"(quant {os.environ['OPENROUTER_QUANTIZATION']}), {args.n} calls")

    failures = 0
    total_cost = 0.0
    for i in range(args.n):
        t0 = time.monotonic()
        try:
            outs = handler.engine.generate([PROMPT], response_format=SCHEMA)
            out = outs[0]
            wall = time.monotonic() - t0
            parsed = json.loads(out.prediction)
            quotes = parsed.get("quotes")
            ok_schema = isinstance(quotes, list) and len(quotes) >= 1
            ok_provider = out.provider == expected_provider
            total_cost += out.cost or 0.0
            status = "OK" if (ok_schema and ok_provider) else "FAIL"
            if status == "FAIL":
                failures += 1
            print(f"  call {i+1}: {status} provider={out.provider} "
                  f"rf={out.response_format_mode} quotes={len(quotes) if quotes else 0} "
                  f"wall={wall:.1f}s cost=${out.cost or 0:.6f}")
            if not ok_provider:
                print(f"    provider drift: expected {expected_provider}, got {out.provider}")
            if not ok_schema:
                print(f"    schema break: {out.prediction[:120]!r}")
        except Exception as exc:
            failures += 1
            print(f"  call {i+1}: ERROR {type(exc).__name__}: {str(exc)[:160]}")

    print(f"total cost ${total_cost:.6f}")
    if failures:
        print(f"SERVING RED: {failures}/{args.n} probes failed")
        return 1
    print("SERVING GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
