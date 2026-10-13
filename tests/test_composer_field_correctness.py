"""F2 composer field-correctness fixes (composer-redesign v2, phase F2).

Network-free: HTTP and the LLM handler are monkeypatched. Each fix is asserted with small
inline fixtures, not by depending on output/ run dirs.

  F2-1 logo: org-only, no person photos, no identicons; fetched logo not source-verified.
  F2-2 org_url: only for a confirmed HF org; a user handle -> NS.
  F2-3 name: mixed-case subtitle stripped when the pre-colon token is a short identity.
  F2-5 size_breakdown: explicit-% rejection; split-named breakdowns consistency-checked;
       heterogeneous statistic breakdowns kept.
  F2-6 collection_date: paper year preferred; HF-upload year low-confidence; prose -> NS.
  F2-7 audience: intra-list exact-duplicate dedup only (never blanks consumers).
  F2-8 name: single-EEE fallback to the EEE benchmark_name when otherwise NS.
"""

import json

from auto_benchmarkcard.tools.composer import composer_tool as ct
from auto_benchmarkcard.tools.composer import composer_tool as C  # full-compose alias


class _Resp:
    def __init__(self, *, json_data=None, status=200):
        self._json, self.status_code = json_data, status

    def json(self):
        return self._json


def _clear_logo_caches():
    ct._logo_cache.clear()
    ct._org_overview_cache.clear()


# ------------------------------------------------------------------ F2-1 logo / identicons
def test_f2_1_identicon_detection():
    assert ct._is_identicon_avatar("/avatars/6f9b.svg")
    assert ct._is_identicon_avatar("https://huggingface.co/avatars/6f9b.svg")
    assert not ct._is_identicon_avatar("https://cdn-avatars.huggingface.co/v1/x.png")
    assert not ct._is_identicon_avatar(None)
    assert not ct._is_identicon_avatar("")


def test_f2_1_org_avatar_identicon_rejected(monkeypatch):
    _clear_logo_caches()
    monkeypatch.setattr(ct.requests, "get", lambda url, **k: (
        _Resp(json_data={"avatarUrl": "https://huggingface.co/avatars/zzz.svg"})
        if "organizations/realorg/overview" in url else _Resp(status=404)))
    monkeypatch.setattr(ct.requests, "head", lambda *a, **k: _Resp(status=404))
    assert ct._fetch_org_logo("realorg") is None


def test_f2_1_fetched_logo_provenance_not_verified(monkeypatch):
    # A fetched org logo keeps its value but is never marked source-verified.
    monkeypatch.setattr(ct, "_hf_org_overview", lambda owner: {})  # confirmed org
    monkeypatch.setattr(ct, "_fetch_org_logo",
                        lambda org: "https://cdn-avatars.huggingface.co/v1/x.png")
    card, prov = {"benchmark_details": {}}, {}
    ct.apply_display_fields(card, hf_metadata=None,
                            extracted_ids={"hf_repo": "bigcode/bigcodebench"}, provenance=prov)
    bd = card["benchmark_details"]
    assert bd["logo"] == "https://cdn-avatars.huggingface.co/v1/x.png"
    assert prov["benchmark_details"]["logo"]["verified"] is False
    assert prov["benchmark_details"]["logo"]["status"] == "derived"
    # org_url (a structured derivation, not a fetch) stays verified
    assert prov["benchmark_details"]["org_url"]["verified"] is True


# ------------------------------------------------------------------ F2-2 org_url guard
def test_f2_2_org_url_set_for_confirmed_org(monkeypatch):
    monkeypatch.setattr(ct, "_hf_org_overview", lambda owner: {})  # confirmed org
    monkeypatch.setattr(ct, "_fetch_org_logo", lambda org: None)
    card, prov = {"benchmark_details": {}}, {}
    ct.apply_display_fields(card, hf_metadata=None,
                            extracted_ids={"hf_repo": "bigcode/bigcodebench"}, provenance=prov)
    assert card["benchmark_details"]["org_url"] == "https://huggingface.co/bigcode"


def test_f2_2_org_url_none_for_user(monkeypatch):
    # A user-owned dataset gets no org_url (no personal HF profile) and no logo.
    monkeypatch.setattr(ct, "_hf_org_overview", lambda owner: None)  # user handle (404)
    card, prov = {"benchmark_details": {}}, {}
    ct.apply_display_fields(card, hf_metadata=None,
                            extracted_ids={"hf_repo": "aps/superglue"}, provenance=prov)
    assert not card["benchmark_details"].get("org_url")
    assert "org_url" not in prov.get("benchmark_details", {})
    assert not card["benchmark_details"].get("logo")


# ------------------------------------------------------ FINISH-8 Fix 2: HF-README BibTeX identity gate
# A derivative dataset's README often carries its PARENT's bibtex (mgsm's README is the GSM8K bibtex
# cobbe2021gsm8k, 12 authors), which would leak the wrong authors independent of the paper binding.
# The bibtex authors are taken only when the bibtex key/title shares a DISTINCTIVE acronym/slug token
# with the EEE identity (an expanded name-word like "math" would false-pass the parent citation).

_GSM8K_BIBTEX = (
    "## Citation Information\n```bibtex\n@article{cobbe2021gsm8k,\n"
    "  title={Training Verifiers to Solve Math Word Problems},\n"
    "  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and "
    "Jun, Heewoo and Kaiser, Lukasz},\n  year={2021}\n}\n```\n")


def test_f2_bibtex_identity_extract():
    key, title = ct._bibtex_identity(_GSM8K_BIBTEX)
    assert key == "cobbe2021gsm8k"
    assert title == "Training Verifiers to Solve Math Word Problems"
    assert ct._bibtex_identity("no bibtex here") == ("", "")


def test_f2_distinctive_tokens_acronym_not_expanded_words():
    # mgsm identity -> distinctive token {mgsm}; the expanded words (multilingual/grade/school/math)
    # are NOT identity tokens (math appears in GSM8K's title -> would false-pass the parent).
    toks = ct._distinctive_identity_tokens(
        {"benchmark_name": "mgsm"},
        "Multilingual Grade School Math (MGSM) evaluates LLM reasoning over 250 problems from GSM8K.")
    assert "mgsm" in toks
    assert "math" not in toks and "grade" not in toks
    # hyphenated slug -> each part + the joined whole
    am = ct._distinctive_identity_tokens({"benchmark_name": "acemath-rewardbench"}, "")
    assert {"acemath", "rewardbench", "acemathrewardbench"} <= am


def test_f2_bibtex_mislabeled_parent_citation_authors_dropped():
    # mgsm card whose HF README is the GSM8K bibtex -> authors NOT taken (no GSM8K Cobbe-et-al leak).
    card = {"benchmark_details": {"overview": "Multilingual Grade School Math (MGSM): 250 problems "
                                  "translated from GSM8K into 10 languages."}}
    hf = {"tags": [], "readme_markdown": _GSM8K_BIBTEX, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "mgsm"})
    assert not card["benchmark_details"].get("authors")


def test_f2_bibtex_own_citation_cosmetic_title_kept():
    # charxiv's own citation: title omits the literal "charxiv" but the key wang2024charxiv carries it
    # -> authors KEPT (the loose key match avoids a false-drop on a cosmetic-title own-citation).
    bib = ("@article{wang2024charxiv,\n  title={Charting Gaps in Realistic Chart Understanding in "
           "Multimodal LLMs},\n  author={Wang, Zirui and Xia, Mengzhou},\n  year={2024}\n}")
    card = {"benchmark_details": {"overview": "CharXiv: realistic chart understanding."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "charxiv"})
    assert card["benchmark_details"]["authors"] == ["Zirui Wang", "Mengzhou Xia"]


def test_f2_bibtex_own_citation_hyphenated_key_kept():
    # acemath-rewardbench own citation (key 'acemath2024...') -> the 'acemath' slug token matches the
    # key -> authors KEPT. Proves no false-drop on a legit shortened/hyphenated own-citation key.
    bib = ("@article{acemath2024,\n  title={AceMath: Advancing Frontier Math Reasoning with "
           "Post-Training},\n  author={Liu, Zihan and Ping, Wei},\n  year={2024}\n}")
    card = {"benchmark_details": {"overview": "AceMath RewardBench evaluates math reward models."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "acemath-rewardbench"})
    assert card["benchmark_details"]["authors"] == ["Zihan Liu", "Wei Ping"]


def test_f2_bibtex_title_substring_collision_does_not_leak():
    # BLOCKER repro: a short identity acronym must NOT match as a TITLE substring. token "arc" inside
    # "architecture" would otherwise leak Neural Architecture Search authors onto an ARC card. The
    # title side requires a WHOLE-WORD match; the bibtex key here carries no "arc" -> authors NOT taken.
    bib = ("@article{zoph2017nas,\n  title={Neural Architecture Search with Reinforcement Learning},\n"
           "  author={Zoph, Barret and Le, Quoc V.},\n  year={2017}\n}")
    card = {"benchmark_details": {"overview": "ARC: AI2 Reasoning Challenge of grade-school science."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "arc"})
    assert not card["benchmark_details"].get("authors")


def test_f2_bibtex_title_wholeword_match_kept():
    # The flip side: a real WHOLE-WORD title match keeps the authors even when the key does not carry
    # the slug. "rewardbench" is a whole word in "AceMath-RewardBench" -> KEPT (no false-drop).
    bib = ("@article{badkey9999,\n  title={AceMath-RewardBench: A Benchmark for Math Reward Models},\n"
           "  author={Liu, Zihan and Ping, Wei},\n  year={2024}\n}")
    card = {"benchmark_details": {"overview": "AceMath RewardBench evaluates math reward models."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "rewardbench"})
    assert card["benchmark_details"]["authors"] == ["Zihan Liu", "Wei Ping"]


def test_f2_bibtex_matches_identity_wholeword_unit():
    # Direct: title substring collisions reject (arc/architecture, nli/online, ade/trade); key
    # substring and whole-word title matches accept.
    assert not ct._bibtex_matches_identity(
        "zoph2017nas", "Neural Architecture Search with Reinforcement Learning", {"arc"})
    assert not ct._bibtex_matches_identity("x2020", "Learning Online Representations", {"nli"})
    assert not ct._bibtex_matches_identity("x2020", "TRADE: a dialogue state tracker", {"ade"})
    assert ct._bibtex_matches_identity("badkey", "AceMath-RewardBench: A Benchmark", {"rewardbench"})
    assert ct._bibtex_matches_identity("badkey", "GSM8K: Math Word Problems", {"gsm8k"})
    assert ct._bibtex_matches_identity("wang2024charxiv", "Charting Gaps in Charts", {"charxiv"})


def test_f2_bibtex_short_token_key_substring_no_leak():
    # FINISH-8 fix-round 2: a SHORT acronym (len 3-4) must NOT corroborate via a KEY substring -- it
    # collides with common author-keyed words ('research'/'online'/'architecture'). Only the whole-word
    # title can corroborate a short token. Long slugs keep key-substring.
    assert not ct._bibtex_matches_identity("lee2020research", "Some Unrelated Title", {"arc"})
    assert not ct._bibtex_matches_identity("zhang2020online", "A Title", {"nli"})
    assert not ct._bibtex_matches_identity("xu2021hierarchical", "A Title", {"arc"})
    # short token survives via a whole-word TITLE match (a legit ARC own-citation)
    assert ct._bibtex_matches_identity(
        "clark2018arc", "Think You Have Solved Question Answering? Try ARC", {"arc"})
    # long slugs still corroborate via the key substring (slug recovery preserved)
    assert ct._bibtex_matches_identity("cobbe2021gsm8k", "Unrelated Title", {"gsm8k"})
    assert ct._bibtex_matches_identity("wang2024charxiv", "Unrelated Title", {"charxiv"})


def test_f2_bibtex_short_token_key_leak_e2e_dropped():
    # End-to-end: an ARC card whose HF README carries an unrelated bibtex whose KEY word merely
    # CONTAINS "arc" ('research') and whose title lacks "arc" -> authors NOT taken (leak closed).
    bib = ("@article{lee2020research,\n  title={A Study of Representation Learning},\n"
           "  author={Lee, Jane and Kim, Soo},\n  year={2020}\n}")
    card = {"benchmark_details": {"overview": "ARC: AI2 Reasoning Challenge of grade-school science."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "arc"})
    assert not card["benchmark_details"].get("authors")


def test_f2_bibtex_mmlu_pro_via_title_kept():
    # mmlu-pro: the 4-char token "mmlu" is excluded from the key side, but the title "MMLU-Pro"
    # carries "mmlu" as a whole word -> authors KEPT (no false-drop on a real own-citation).
    bib = ("@article{badkey9999,\n  title={MMLU-Pro: A More Robust and Challenging Benchmark},\n"
           "  author={Wang, Yubo and Ma, Xueguang},\n  year={2024}\n}")
    card = {"benchmark_details": {"overview": "MMLU-Pro: a robust multitask language understanding "
                                  "benchmark."}}
    hf = {"tags": [], "readme_markdown": bib, "card_data": {}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "mmlu-pro"})
    assert card["benchmark_details"]["authors"] == ["Yubo Wang", "Xueguang Ma"]


def test_f2_bibtex_gate_does_not_override_card_data_or_paper_authors():
    # The gate sits only on the HF-README BibTeX fallback: HF card_data person list and the resolved
    # paper authors still take precedence (unchanged ordering), even with a mismatched bibtex present.
    card = {"benchmark_details": {"overview": "MGSM benchmark."}}
    hf = {"tags": [], "readme_markdown": _GSM8K_BIBTEX,
          "card_data": {"authors": ["Real One", "Real Two"]}}
    ct.apply_display_fields(card, hf_metadata=hf, extracted_ids={},
                            provenance={}, eee_metadata={"benchmark_name": "mgsm"})
    assert card["benchmark_details"]["authors"] == ["Real One", "Real Two"]
    # paper authors path (no card_data) also wins over the bibtex
    card2 = {"benchmark_details": {"overview": "MGSM benchmark."}}
    ct.apply_display_fields(card2, hf_metadata={"tags": [], "readme_markdown": _GSM8K_BIBTEX,
                                                "card_data": {}},
                            extracted_ids={"paper_authors": ["Freda Shi"]},
                            provenance={}, eee_metadata={"benchmark_name": "mgsm"})
    assert card2["benchmark_details"]["authors"] == ["Freda Shi"]


# ------------------------------------------------------------------ F2-3 name subtitle
def test_f2_3_short_identity_helper():
    assert ct._is_short_identity("SuperGLUE")
    assert ct._is_short_identity("API-Bank")
    assert ct._is_short_identity("Big Bench Hard")           # 3 words
    assert not ct._is_short_identity("A Comprehensive Benchmark for Tools")  # > 3 words
    assert not ct._is_short_identity("")


def test_f2_3_mixed_case_subtitle_stripped():
    assert ct._clean_caps_name(
        "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding", ()
    ) == "SuperGLUE"
    assert ct._clean_caps_name(
        "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs", ()
    ) == "API-Bank"
    assert ct._clean_caps_name("BigCodeBench: Benchmarking Code Generation", ()) == "BigCodeBench"


def test_f2_3_long_precolon_not_truncated():
    # pre-colon is a long descriptive phrase (> 3 words), not a short identity -> unchanged.
    n = "A Benchmark for Evaluating Reasoning and Planning: Methods and Results"
    assert ct._clean_caps_name(n, ()) == n


def test_f2_3_candidate_propercase_preferred():
    assert ct._clean_caps_name("superglue: a stickier benchmark", ("SuperGLUE",)) == "SuperGLUE"


def test_f2_3_no_colon_unchanged():
    assert ct._clean_caps_name("BLINK", ()) == "BLINK"
    assert ct._clean_caps_name("AI2 Reasoning Challenge (ARC)", ()) == "AI2 Reasoning Challenge (ARC)"


# ------------------------------------------------------------------ F2-5 size_breakdown
def test_f2_5_rejects_percent_rows():
    md = "| Split | % |\n| --- | --- |\n| Train | 60% |\n| Test | 40% |\n"
    assert ct._size_breakdown_from_readme(md) is None


def test_f2_5_keeps_real_count_split():
    md = "| Split | Count |\n| --- | --- |\n| Train | 60 |\n| Test | 40 |\n"
    assert ct._size_breakdown_from_readme(md) == {"Train": 60, "Test": 40}


def test_f2_5_size_count_parse():
    assert ct._size_count("805 instruction-following tasks") == 805
    assert ct._size_count("1,200") == 1200
    assert ct._size_count(314) == 314
    assert ct._size_count("10K<n<100K") is None     # bucket range, not a count
    assert ct._size_count("n<1K") is None
    assert ct._size_count(None) is None


def test_f2_5_consistency_keeps_heterogeneous():
    # bigcodebench-style heterogeneous statistics are not summable totals -> always kept.
    bd = {"# Task": 1140, "# Domain": 7, "libraries": 200, "combinations": 1045}
    assert ct._size_breakdown_is_consistent(bd, 1140, "1K<n<10K") is True


def test_f2_5_consistency_drops_inconsistent_split():
    bd = {"train": 2000, "test": 202}                # sums 2202
    assert ct._size_breakdown_is_consistent(bd, 314, None) is False
    # train-only presented against the full total
    assert ct._size_breakdown_is_consistent({"train": 1000}, 1514, None) is False


def test_f2_5_consistency_keeps_matching_split():
    assert ct._size_breakdown_is_consistent({"train": 1200, "test": 314}, 1514, None) is True


def test_f2_5_consistency_keeps_when_total_unknown():
    # no authoritative total (data.size is a bucket) -> cannot disprove -> keep.
    assert ct._size_breakdown_is_consistent({"train": 60, "test": 40}, None, "1K<n<10K") is True


# ------------------------------------------------------------------ F2-6 collection_date
def test_f2_6_has_year():
    assert ct._has_year("2018")
    assert ct._has_year("collected in 2021 from public exams")
    assert not ct._has_year("drawn from grade-school science exams")
    assert not ct._has_year(None)


def test_f2_6_no_collection_date_from_upload_year(monkeypatch):
    # A publication / HF-upload year is NOT a data-collection date: collection_date is left
    # NS, never seeded from created_at (F6-2).
    monkeypatch.setattr(C, "get_llm_handler",
                        lambda *a, **k: _FakeHandler(_all_ns_response()))
    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    res = fn(query="TestBench",
             hf_metadata={"tags": [], "created_at": "2025-06-01T00:00:00Z"},
             extracted_ids={})
    card = res["benchmark_card"]
    assert card["data"]["collection_date"] == "Not specified"
    prov = (res.get("provenance") or {}).get("data", {})
    assert "collection_date" not in prov


def test_f2_6_prose_collection_date_coerced():
    # A grounded but non-year prose collection_date is dropped to NS; a year-bearing value
    # and the NS sentinel are left unchanged.
    assert ct._coerce_collection_date("drawn from grade-school science exams") == "Not specified"
    assert ct._coerce_collection_date("2018") == "2018"
    assert ct._coerce_collection_date("collected during 2018") == "collected during 2018"
    assert ct._coerce_collection_date("Not specified") == "Not specified"
    assert ct._coerce_collection_date(None) is None


def test_f4_collection_date_relative_dropped():
    # F4: a relative-time reference is not a real collection date, even with an embedded year
    assert ct._coerce_collection_date(
        "The collection date is reported as today (2025-08-12).") == "Not specified"
    assert ct._coerce_collection_date("data current as of 2024") == "Not specified"
    # a year-bearing date with no relative word is still kept
    assert ct._coerce_collection_date("collected in 2021 from public exams") == "collected in 2021 from public exams"


# ------------------------------------------------------------------ F2-7 audience dedup
def test_f2_7_dedup_preserve_order():
    assert ct._dedup_preserve_order(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]
    assert ct._dedup_preserve_order(["Not specified"]) == ["Not specified"]
    assert ct._dedup_preserve_order([]) == []


def test_f2_7_per_list_dedup_never_blanks():
    # The post-pass dedups each audience list independently and never empties one. Identical
    # evaluators/consumers therefore both survive as the same (deduped) list -- consumers is
    # never blanked to NS just because it equals evaluators.
    same = ["AI researchers", "AI researchers", "Practitioners"]
    evaluators = ct._dedup_preserve_order(list(same))
    consumers = ct._dedup_preserve_order(list(same))
    assert evaluators == ["AI researchers", "Practitioners"]
    assert consumers == evaluators           # identical cross-field relationship preserved
    assert consumers != []                   # consumers never blanked


# ------------------------------------------------------------------ F2-8 EEE name fallback
def test_f2_8_name_falls_back_to_eee(monkeypatch):
    monkeypatch.setattr(C, "get_llm_handler",
                        lambda *a, **k: _FakeHandler(_all_ns_response()))
    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    res = fn(query="alpacaeval-2.0", eee_metadata={"benchmark_name": "AlpacaEval 2.0"})
    assert res["benchmark_card"]["benchmark_details"]["name"] == "AlpacaEval 2.0"


def test_f2_8_real_name_not_overwritten(monkeypatch):
    resp = json.loads(_all_ns_response())
    resp["benchmark_details"]["name"] = "RealName"
    resp["benchmark_details"]["provenance"] = {
        "name": {"source": "paper", "evidence": "RealName"}}
    monkeypatch.setattr(C, "get_llm_handler", lambda *a, **k: _FakeHandler(json.dumps(resp)))
    fn = getattr(C.compose_benchmark_card, "func", C.compose_benchmark_card)
    res = fn(query="x", eee_metadata={"benchmark_name": "EEEName"})
    assert res["benchmark_card"]["benchmark_details"]["name"] == "RealName"


# --- shared compose fakes (mirror tests/test_composer_stage_b.py) ---
from auto_benchmarkcard.tools.composer.composer_tool import (  # noqa: E402
    BenchmarkDetails, DataInfo, EthicalAndLegalConsiderations,
    Methodology, PurposeAndIntendedUsers)

_SECTION_MODELS = {
    "benchmark_details": BenchmarkDetails,
    "purpose_and_intended_users": PurposeAndIntendedUsers,
    "data": DataInfo,
    "methodology": Methodology,
    "ethical_and_legal_considerations": EthicalAndLegalConsiderations,
}


def _all_ns_response():
    out = {}
    for sec, cls in _SECTION_MODELS.items():
        out[sec] = {f: "Not specified" for f in cls.model_fields if f != "provenance"}
        out[sec]["provenance"] = {}
    return json.dumps(out)


class _FakeHandler:
    model_name = "fake-model"

    def __init__(self, resp):
        self.resp = resp

    def generate(self, prompt, response_format=None):
        return self.resp

    def generate_with_meta(self, prompt, response_format=None, max_completion_tokens=None):
        return self.resp, None
