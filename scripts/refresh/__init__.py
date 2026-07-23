"""Refresh-from-cache harness for the card quality round.

Reconstructs composer/risk inputs from cached tool_output, re-invokes the real
composer offline, then applies an improve-only field merge onto the published
card and emits a per-field changelog. Phase A/B/C register their targeted fields
through registry.FieldSpec; they never edit this harness.

The NULL-REFRESH acceptance gate (empty registry + compose skipped) proves the
reconstruct -> merge path is byte-identical to the live cards without depending
on LLM determinism. See scripts/refresh_from_cache.py for the CLI.
"""
