"""Per-call usage records belong to the card that made the call.

Failure class: usage_log_crosstalk. The sink was one process-wide path, so a batch runner
composing several cards at once had every worker append to whichever file was set most
recently: per-card cost was wrong and the per-card spend tripwire watched the wrong card.
"""

from __future__ import annotations

import json
import threading

from auto_benchmarkcard import llm_handler as H


class _Out:
    prediction = "x"
    stop_reason = "stop"
    input_tokens = 3
    output_tokens = 5
    serving = "openrouter"
    provider = "Together"
    quantization = None
    response_format_mode = "json_schema"
    cost = 0.001


class _Handler:
    model_name = "m"
    engine_type = "OPENROUTER"
    _record_usage = H.LLMHandler._record_usage


def test_each_thread_records_into_its_own_usage_log(tmp_path):
    def worker(index: int) -> None:
        H.set_usage_log_path(str(tmp_path / f"{index}.jsonl"), this_thread_only=True)
        _Handler()._record_usage(_Out(), "generate", 0.1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(4):
        rows = [json.loads(line) for line in
                (tmp_path / f"{i}.jsonl").read_text().splitlines() if line.strip()]
        assert len(rows) == 1 and rows[0]["cost"] == 0.001


def test_the_process_default_still_serves_a_single_threaded_caller(tmp_path):
    log = tmp_path / "single.jsonl"
    try:
        H.set_usage_log_path(str(log))
        _Handler()._record_usage(_Out(), "generate", 0.1)
        assert len(log.read_text().splitlines()) == 1
    finally:
        H.clear_usage_log_path()
    _Handler()._record_usage(_Out(), "generate", 0.1)
    assert len(log.read_text().splitlines()) == 1
