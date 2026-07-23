"""Tests for the EEE-path card_info stamp (_stamp_card_info).

The risk_enhanced_card fallback (used when the FactReasoner worker did not run) carries no
card_info; the EEE workflow stamps it before save. Network-free.
"""

from datetime import datetime

from auto_benchmarkcard.config import Config
from auto_benchmarkcard.eee_workflow import _stamp_card_info


def test_stamps_card_info_when_absent():
    card = {"methodology": {}}
    _stamp_card_info(card)
    ci = card["card_info"]
    assert ci["llm"] == Config.COMPOSER_MODEL
    assert ci["schema_version"] == "v2"
    # created_at is an ISO-8601 timestamp (parses without error)
    datetime.fromisoformat(ci["created_at"])


def test_does_not_overwrite_existing_card_info():
    existing = {"created_at": "2026-01-01T00:00:00", "llm": "x"}
    card = {"card_info": existing}
    _stamp_card_info(card)
    # a real FactReasoner stamp is preserved untouched
    assert card["card_info"] is existing
