"""Tests for the post-Stage-B aboutness guard (judge_* + data.source).

The guard is structural and model-independent, so it is exercised directly with
synthetic evidence. Cases cover: classifier recall/precision (incl. the real aa-lcr
equality-checker phrasing that must NOT be demoted), the two judge demotion branches,
data.source subset rejection, the backfill-resurrection regression, and that the
frozen gate classifier buckets the outcome as grounded / NS (never ungrounded).
"""

import os
import sys

import pytest

from auto_benchmarkcard.card_utils import backfill_from_provenance, is_not_specified
from auto_benchmarkcard.tools.composer import aboutness as A
from auto_benchmarkcard.tools.composer import field_spec as fs

# The GO-gate classifier is the frozen instrument; import it read-only to assert the
# guard's output buckets correctly (sibling-script import mirrors the gate's own pattern).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from compute_gate_metrics import EXTRACTED as GATE_EXTRACTED  # noqa: E402
from compute_gate_metrics import NS as GATE_NS  # noqa: E402
from compute_gate_metrics import classify as gate_classify  # noqa: E402

# Real evidence strings from the artifacts (grounding the fixtures).
Q_ANNOTATORS = "We engage 13 authors as human annotators (including the lead annotator)"
Q_CODE_INTERP = "Specifically, we utilize the Code Interpreter session in the web-based GPT-4"
Q_PASS1 = "The main experiments report Pass@1 with greedy decoding in the zero-shot setting"
Q_SUBSET = ("leveraging the all-mpnet-base-v2 sentence embedding model to bridge query sources "
            "and BigCodeBench tasks, resulting in 6,895 queries and 626 BigCodeBench tasks after deduplication")
Q_EQUALITY_CHECKER = "Qwen3 235B A22B 2507 Non-reasoning is used as the equality checker model"
Q_EQUALITY_CHECKER2 = "We use an LLM-based equality checker to evaluate responses"
Q_GITHUB = "The data was collected from public GitHub repositories"
# attaq: a discriminative reward/ranking classifier (scalar scorer), NOT a generative LLM judge.
Q_ATTAQ_RANKING = ("In this paper, we use the ranking model released by OpenAssistant. This ranking model "
                   "provides scores indicating the likelihood of a response being seen as harmless and helpful.")
Q_ATTAQ_REWARD = "Our paper utilizes the 'OpenAssistant/reward-model-deberta-v3-large-v2' Reward model."
Q_ATTAQ_NORM = ("To ensure uniform scores and create a standardized range, we first limit the model's output "
                "scores to the range [-8,1], and subsequently employ min-max normalization to yield scores.")
# ace: the judge role and the model name are split across adjacent sentences (cross-sentence capture).
Q_ACE_GRADING = ("Model grading. For each response, we independently score the rubric's criterion, following "
                 "industry practice in using an LM judge (Gu et al., 2025; Zhu et al., 2025). "
                 "We use Gemini 2.5 Pro with Thinking = High and Temperature set to 0.0.")
# wordle: a deterministic game win rate is not an LLM judge.
Q_WORDLE_SCAN = ("Win rate (%) on Wordle Arena: guess a 5-letter word in 6 attempts with green/yellow/gray feedback")


# --------------------------------------------------------------- classifiers ----

@pytest.mark.parametrize("quote,expected", [
    # genuine output-scoring phrasings (must be recognized)
    ("scored by 3 LLM judges, majority vote", True),
    ("GPT-4 acts as the judge", True),
    ("we use an LLM-as-a-judge to grade answers", True),
    ("responses were scored by GPT-4 as an automatic evaluator on a 1-10 scale", True),
    # discriminative scalar scorers (reward/ranking model, classifier) are NOT generative LLM
    # judges: they emit a numeric score, not a verdict (attaq's DeBERTa reward model).
    ("a reward model assigns the final score", False),
    (Q_ATTAQ_RANKING, False),
    (Q_ATTAQ_REWARD, False),
    (Q_ATTAQ_NORM, False),
    ("Nemotron-4 340B Reward model is used as the judge.", True),  # generative judge survives the veto
    (Q_EQUALITY_CHECKER, True),          # aa-lcr: bare "equality checker model"
    (Q_EQUALITY_CHECKER2, True),         # aa-lcr: "evaluate responses"
    ("answers are graded against the reference answer", True),
    # construction / annotation / non-scoring (must NOT be recognized)
    (Q_ANNOTATORS, False),
    (Q_CODE_INTERP, False),
    ("13 authors as human annotators rated the difficulty of each task", False),  # scoring verb, wrong object + veto
    ("annotators used a rubric to label the data", False),                         # construction veto over a judge term
    ("", False),
])
def test_describes_output_scoring(quote, expected):
    assert A.describes_output_scoring(quote) is expected


@pytest.mark.parametrize("quote,expected", [
    (Q_PASS1, True),
    ("each solution is checked against a suite of unit tests in a sandbox", True),
    ("we measure exact match accuracy", True),
    (Q_ANNOTATORS, False),
    (Q_EQUALITY_CHECKER, False),
])
def test_describes_exec_scoring(quote, expected):
    assert A.describes_exec_scoring(quote) is expected


def test_subset_vs_provenance():
    assert A.is_subset_construction(Q_SUBSET) is True
    assert A.is_genuine_provenance(Q_SUBSET) is False
    assert A.is_genuine_provenance(Q_GITHUB) is True
    assert A.is_subset_construction(Q_GITHUB) is False


# ------------------------ shared EEE+HF LLM-judge detector (J1/J3/J4/J5) ----

@pytest.mark.parametrize("text,is_judge,model", [
    # named judge / equality checker -> True + the model string
    (Q_EQUALITY_CHECKER, True, "Qwen3 235B A22B 2507 Non-reasoning"),
    ("GPT-4 is used as the judge model", True, "GPT-4"),
    ("GPT-4 acts as the judge", True, "GPT-4"),
    ("GPT-4 served as the evaluator.", True, "GPT-4"),                 # past tense 'served'
    ("We use GPT-4 as the judge.", True, "GPT-4"),                     # bare 'use X as the judge'
    ("The judge model is GPT-4.", True, "GPT-4"),                      # 'judge model is X'
    ("We use Llama-3.3-70B to grade the answers", True, "Llama-3.3-70B"),
    ("We use a fine-tuned Llama-3.3-70B model to grade the answers.", True, "Llama-3.3-70B"),  # strip adj/trailing
    ("We compare against MT-Bench, where GPT-4 is used as the judge.", True, "GPT-4"),         # strip leading junk
    # judge signal but NO confident model name -> True + None (degrade to boolean-only)
    (Q_EQUALITY_CHECKER2, True, None),                       # "LLM-based equality checker" descriptor
    ("BioLP-Bench is a model-graded evaluation", True, None),
    ("Responses are graded by an LLM", True, None),
    # "LM judge" / "language model judge" now recognized (ace's phrasing)
    ("Responses are scored by an LM judge.", True, None),
    ("A language model judge assigns the verdict.", True, None),
    # a bare/game win rate is deterministic, not a judge; only a comparison-judge context
    # (pairwise / preference / judged) makes win rate an LLM-judge signal.
    (Q_WORDLE_SCAN, False, None),
    ("Win-rate is computed against the reference", False, None),
    ("Models are compared by pairwise win rate.", True, None),
    ("The judge model is unclear.", True, None),             # boolean fires, but no junk model
    # HUMAN evaluation is NOT an LLM judge (the brief's must-not-fire case)
    ("We use three human evaluators to judge the responses.", False, None),
    ("We use human annotators to score each answer.", False, None),
    ("We use Mechanical Turk workers to evaluate the answers.", False, None),
    ("We use a panel to judge the outputs.", False, None),
    ("We use crowdworkers to evaluate the outputs.", False, None),
    # a model used to BUILD / filter the dataset is construction, not output scoring
    ("We use GPT-4 to evaluate the difficulty of each task during dataset construction.", False, None),
    ("We use Mistral to score candidate items for inclusion.", False, None),
    ("We use GPT-4 to evaluate whether each generated question is answerable.", False, None),
    # exact-match / execution scoring is NOT an LLM judge (arc / api-bank / bigcode)
    ("Scored by exact match against the reference answer", False, None),
    ("Correctness is measured by exact string match of the API call", False, None),
    ("The main experiments report Pass@1 with greedy decoding", False, None),
    ("Each solution is checked against a suite of unit tests", False, None),
    # vetoes: citation context and answer-extractor mentions
    ("Outputs are graded by an LLM (Zheng et al., 2023)", False, None),
    ("We graded by an llm as described in [12]", False, None),
    ("An LLM-based equality checker is used to extract the final answer", False, None),
    ("", False, None),
])
def test_llm_judge_detector(text, is_judge, model):
    assert A.describes_llm_judge(text) is is_judge
    assert A.extract_judge_model(text) == model


@pytest.mark.parametrize("text,is_judge,model", [
    # a judge that scores RESPONSES is output scoring even if it mentions difficulty/filter/inclusion
    ("GPT-4 acts as a judge to evaluate the difficulty of each response.", True, "GPT-4"),
    ("We use GPT-4 as a judge to evaluate responses for inclusion in the final score.", True, "GPT-4"),
    ("GPT-4 acts as a judge to filter and score low-quality responses.", True, "GPT-4"),
    # an LLM applying a HUMAN-WRITTEN rubric is still an LLM judge (human is not the agent)
    ("GPT-4 is used as a judge to grade responses against a human-written rubric.", True, "GPT-4"),
    # a vetoed sibling clause (';'-joined) must not suppress the judge clause
    ("GPT-4 serves as a judge and scores each response; the dataset difficulty was rated by humans.", True, "GPT-4"),
    # extraction shapes: capitalized article, multi-word tier names, 'Reward model', connector words
    ("The GPT-4 model is used as the judge.", True, "GPT-4"),
    ("Mistral Large is used as the judge.", True, "Mistral Large"),
    ("Claude Opus acts as the judge.", True, "Claude Opus"),
    ("Nemotron-4 340B Reward model is used as the judge.", True, "Nemotron-4 340B"),
    ("GPT-4 Turbo is used as the judge.", True, "GPT-4 Turbo"),
    # still vetoes genuine construction (no output noun) and human evaluation
    ("We use GPT-4 to evaluate the difficulty of each task during dataset construction.", False, None),
    ("We use GPT-4 to evaluate whether each generated question is answerable.", False, None),
])
def test_llm_judge_detector_round2(text, is_judge, model):
    assert A.describes_llm_judge(text) is is_judge
    assert A.extract_judge_model(text) == model


@pytest.mark.parametrize("text", [
    "The Gold Standard is used as the judge.",
    "The Final Answer is used as the judge.",
    "Majority Voting is used as the judge.",
    "Our Method is used as the judge.",
    "The Best Response is used as the judge.",
])
def test_extract_does_not_fabricate_model_from_phrase(text):
    """A digitless capitalized PHRASE with no model-vendor token must not be mis-read as a model
    (a fabricated judge_models is worse than degrading to boolean-only)."""
    assert A.extract_judge_model(text) is None


def test_collect_judge_scan_text_merges_eee_and_hf():
    eee = {"metrics": {"m1": {"evaluation_description": "model-graded scoring"},
                       "m2": {"evaluation_description": "accuracy"}}}
    hf = {"readme_markdown": Q_EQUALITY_CHECKER}
    text = A.collect_judge_scan_text(eee, hf)
    assert "model-graded" in text and "equality checker" in text
    assert A.extract_judge_model(text) == "Qwen3 235B A22B 2507 Non-reasoning"
    assert A.collect_judge_scan_text(None, None) == ""


# --------------------- judge guard J4: named judge off the judge bucket ----

def test_named_judge_off_bucket_not_demoted():
    """A named equality-checker quote verified but routed to a NON-judge field (so it is not in
    the judge digest/candidates) keeps the cluster intact via the named-judge escape."""
    gs = _judge_sections(num="Not specified", models=("Qwen3 235B A22B 2507 Non-reasoning",))
    prov = _judge_prov(("E1",))  # judge fields cite a bland, non-scoring quote
    items = _items(("E1", "methodology.methods", "Scoring runs over the dataset"),
                   ("E2", "data.annotation", Q_EQUALITY_CHECKER))  # named judge, off the bucket
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)
    assert gs["methodology"]["judge_uses_llm"] is True
    assert gs["methodology"]["judge_models"] == ["Qwen3 235B A22B 2507 Non-reasoning"]
    assert t["judge_demoted"] is False and not flagged


def test_exec_only_still_demoted_despite_named_judge_escape():
    """The escape must NOT spare an execution-scored cluster: the exec evidence names no judge,
    so branch (a) still demotes to False (bigcode regression guard)."""
    gs = _judge_sections()
    prov = _judge_prov(("E1",))
    items = _items(("E1", "methodology.judge_setup", Q_ANNOTATORS),
                   ("E2", "methodology.methods", Q_PASS1))
    t = A.apply_aboutness_guard(gs, prov, items, {})
    assert gs["methodology"]["judge_uses_llm"] is False
    assert t["judge_branch"] == "a"


def test_exec_cluster_demotes_even_when_separate_quote_names_a_judge():
    """Exec evidence (Pass@1) + a NAMED judge in a SEPARATE off-bucket quote (e.g. related work)
    must still demote to False: the named-judge escape is gated on no exec evidence."""
    gs = _judge_sections()
    prov = _judge_prov(("E1",))
    items = _items(("E1", "methodology.judge_setup", Q_ANNOTATORS),
                   ("E2", "methodology.methods", Q_PASS1),
                   ("E3", "purpose_and_intended_users.tasks",
                    "Prior work uses GPT-4 to evaluate open-ended responses."))
    t = A.apply_aboutness_guard(gs, prov, items, {})
    assert gs["methodology"]["judge_uses_llm"] is False
    assert t["judge_branch"] == "a" and t["judge_demoted"] is True


# ----------------------------------------------------------------- builders ----

def _items(*specs):
    return [{"evidence_id": eid, "field": fld, "quote": q} for eid, fld, q in specs]


def _judge_sections(uses=True, num=13, models=("GPT-4",), source="Not specified"):
    return {
        "methodology": {"judge_uses_llm": uses, "judge_num": num,
                        "judge_models": list(models), "judge_score_consolidation": "Not specified"},
        "data": {"source": source},
    }


def _judge_prov(eids=("E1",)):
    return {"methodology": {f: {"evidence_ids": list(eids)} for f in fs.JUDGE_FLAT_FIELDS}}


def _all_ns_judge_sections():
    return {"methodology": {"judge_uses_llm": "Not specified", "judge_num": "Not specified",
                            "judge_models": ["Not specified"], "judge_score_consolidation": "Not specified"},
            "data": {"source": "Not specified"}}


# ----------------------------------------------------------- judge guard (a) ----

def test_judge_branch_a_false_with_derived_provenance():
    """Construction-only judge evidence + execution-scoring evidence -> judge_uses_llm=False,
    derived provenance linked to the Pass@1 quote (gate: GROUNDED-EXTRACTED)."""
    gs = _judge_sections()
    prov = _judge_prov(("E26",))
    items = _items(("E26", "methodology.judge_setup", Q_ANNOTATORS),
                   ("E10", "methodology.methods", Q_PASS1))
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)

    meth = gs["methodology"]
    assert meth["judge_uses_llm"] is False
    assert meth["judge_num"] == "Not specified"
    assert meth["judge_models"] == ["Not specified"]
    assert meth["judge_score_consolidation"] == "Not specified"
    assert t["judge_branch"] == "a" and t["judge_demoted"] is True

    jp = prov["methodology"]["judge_uses_llm"]
    assert jp["source"] == "derived" and jp["evidence_ids"] == ["E10"] and jp["verified"] is True
    # the misattributed E26 links on the other judge fields are cleared
    for f in ("judge_num", "judge_models", "judge_score_consolidation"):
        assert f not in prov["methodology"]
    # gate buckets the derived False as grounded (not ungrounded) via the evidence_ids link
    assert gate_classify(meth["judge_uses_llm"], jp) == GATE_EXTRACTED


# ----------------------------------------------------------- judge guard (b) ----

def test_judge_branch_b_not_specified_without_exec_evidence():
    """Construction-only judge evidence and NO programmatic-scoring evidence -> Not specified."""
    gs = _judge_sections()
    prov = _judge_prov(("E26",))
    items = _items(("E26", "methodology.judge_setup", Q_ANNOTATORS))  # no exec evidence
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)

    meth = gs["methodology"]
    assert meth["judge_uses_llm"] == "Not specified"
    assert meth["judge_num"] == "Not specified"
    assert meth["judge_models"] == ["Not specified"]
    assert t["judge_branch"] == "b" and t["judge_demoted"] is True
    assert "judge_uses_llm" not in prov["methodology"]  # provenance dropped, not derived
    # NS field with no provenance -> gate NS, never ungrounded
    assert gate_classify(meth["judge_uses_llm"], None) == GATE_NS


def test_annotation_with_scoring_verb_still_demoted():
    gs = _judge_sections()
    items = _items(("E1", "methodology.judge_setup",
                    "13 authors as human annotators rated the difficulty of each task"))
    t = A.apply_aboutness_guard(gs, _judge_prov(("E1",)), items, {})
    assert gs["methodology"]["judge_uses_llm"] == "Not specified"
    assert t["judge_branch"] == "b"


# --------------------------------------------------- judge guard: NOT demoted ----

@pytest.mark.parametrize("quote", [
    "scored by 3 LLM judges, majority vote",
    "responses were scored by GPT-4 as an automatic evaluator on a 1-10 scale",
    "we use an LLM-as-a-judge to grade the answers",
    Q_EQUALITY_CHECKER,    # aa-lcr regression: a real LLM-judge must survive
    Q_EQUALITY_CHECKER2,
])
def test_genuine_llm_judge_not_demoted(quote):
    gs = _judge_sections()
    prov = _judge_prov(("E1",))
    items = _items(("E1", "methodology.judge_setup", quote))
    t = A.apply_aboutness_guard(gs, prov, items, {})
    assert gs["methodology"]["judge_uses_llm"] is True
    assert gs["methodology"]["judge_models"] == ["GPT-4"]
    assert t["judge_demoted"] is False


def test_aa_lcr_real_card_not_demoted_branch_b_would_have_fired():
    """aa-lcr has NO exec-scoring evidence; without wide recall it would hit branch (b) and
    wrongly NS a correct LLM-judge. The equality-checker phrasing must keep it intact."""
    gs = _judge_sections(num="Not specified", models=("Qwen3 235B A22B 2507 Non-reasoning",))
    prov = _judge_prov(("E1",))
    items = _items(("E1", "methodology.judge_setup", Q_EQUALITY_CHECKER))  # no exec evidence anywhere
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)
    assert gs["methodology"]["judge_uses_llm"] is True
    assert gs["methodology"]["judge_models"] == ["Qwen3 235B A22B 2507 Non-reasoning"]
    assert t["judge_demoted"] is False and not flagged


# ------------------------------------------------------------ data.source ----

def test_data_source_subset_rejected_and_provenance_cleared():
    gs = _judge_sections(source=Q_SUBSET)
    prov = {"data": {"source": {"evidence_ids": ["E30"], "source": "paper", "evidence": Q_SUBSET}}}
    items = _items(("E30", "data.source", Q_SUBSET))
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)
    assert gs["data"]["source"] == "Not specified"
    assert "source" not in prov.get("data", {})            # cleared -> backfill cannot resurrect
    assert flagged["data.source"].startswith("[aboutness]")
    assert "schema_invalid" not in flagged["data.source"]  # must not feed the gate parse metric
    assert t["data_source_demoted"] is True
    assert gate_classify(gs["data"]["source"], None) == GATE_NS


def test_data_source_with_genuine_provenance_survives():
    gs = _judge_sections(source="Tasks are drawn from GitHub; the Hard subset keeps 626 after deduplication")
    prov = {"data": {"source": {"evidence_ids": ["E30", "E31"]}}}
    items = _items(("E30", "data.source", Q_SUBSET), ("E31", "data.source", Q_GITHUB))
    t = A.apply_aboutness_guard(gs, prov, items, {})
    assert gs["data"]["source"].startswith("Tasks are drawn from GitHub")
    assert t["data_source_demoted"] is False
    assert "source" in prov["data"]


# ------------------------------------------------------------- regressions ----

def test_backfill_does_not_resurrect_demoted_data_source():
    """After the guard drops data.source provenance, backfill_from_provenance (the prose-only
    validation pass) must NOT write the subset quote back as the field value."""
    gs = _judge_sections(source=Q_SUBSET)
    prov = {"data": {"source": {"evidence_ids": ["E30"], "source": "paper", "evidence": Q_SUBSET}}}
    A.apply_aboutness_guard(gs, prov, _items(("E30", "data.source", Q_SUBSET)), {})

    card = {"data": dict(gs["data"])}
    backfill_from_provenance(card, prov, [Q_SUBSET], prose_only=True)
    assert card["data"]["source"] == "Not specified"


def test_backfill_would_resurrect_if_provenance_left_intact():
    """Contrast that proves the provenance-clear is load-bearing: a leftover prose provenance
    entry with the grounded subset quote IS written back by backfill."""
    card = {"data": {"source": "Not specified", "size": "x", "format": "x", "annotation": "x"}}
    prov = {"data": {"source": {"source": "paper", "evidence": Q_SUBSET, "evidence_ids": ["E30"]}}}
    backfill_from_provenance(card, prov, [Q_SUBSET], prose_only=True)
    assert card["data"]["source"] == Q_SUBSET


def test_idempotent_on_already_ns_card():
    gs = _all_ns_judge_sections()
    prov = {}
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, [], flagged)
    assert gs["methodology"]["judge_uses_llm"] == "Not specified"
    assert gs["data"]["source"] == "Not specified"
    assert not flagged
    assert t["judge_demoted"] is False and t["data_source_demoted"] is False


# --------------------------------- discriminative scorer (attaq) / cross-sentence (ace) ----

def test_attaq_reward_classifier_demoted_to_false():
    """A reward/ranking classifier (DeBERTa, scalar score) is not a generative LLM judge: the
    judge candidates carry no output-scoring evidence, and the discriminative-scorer quote in a
    scoring field licenses the derived judge_uses_llm=False (branch a, gate GROUNDED-EXTRACTED)."""
    gs = _judge_sections(num=1, models=("OpenAssistant/reward-model-deberta-v3-large-v2",))
    prov = _judge_prov(("E1",))
    items = _items(("E1", "methodology.judge_uses_llm", Q_ATTAQ_RANKING),
                   ("E2", "methodology.methods", Q_ATTAQ_RANKING))
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)
    meth = gs["methodology"]
    assert meth["judge_uses_llm"] is False
    assert meth["judge_models"] == ["Not specified"]
    assert t["judge_branch"] == "a" and t["judge_demoted"] is True
    jp = prov["methodology"]["judge_uses_llm"]
    assert jp["source"] == "derived" and jp["evidence_ids"] == ["E2"] and jp["verified"] is True
    assert gate_classify(meth["judge_uses_llm"], jp) == GATE_EXTRACTED


def test_ace_named_judge_off_bucket_via_cross_sentence():
    """ace: the judge role ('using an LM judge') and the model name ('We use Gemini 2.5 Pro with
    ...') are split across sentences and routed off the judge bucket. The cross-sentence capture
    recovers the model so the named-judge escape keeps the cluster intact."""
    gs = _judge_sections(num="Not specified", models=("Gemini 2.5 Pro",))
    prov = _judge_prov(("E1",))  # judge fields cite a bland, non-scoring quote
    items = _items(("E1", "methodology.methods", "We use the mean score of the 8 runs for the leaderboard."),
                   ("E2", "benchmark_details.overview", Q_ACE_GRADING))  # grading passage, off the bucket
    flagged = {}
    t = A.apply_aboutness_guard(gs, prov, items, flagged)
    assert gs["methodology"]["judge_uses_llm"] is True
    assert gs["methodology"]["judge_models"] == ["Gemini 2.5 Pro"]
    assert t["judge_demoted"] is False and not flagged


def test_extract_judge_model_cross_sentence_naming():
    assert A.extract_judge_model(Q_ACE_GRADING) == "Gemini 2.5 Pro"
    # no grading context -> a 'we use X' model-under-test mention must NOT be read as the judge
    assert A.extract_judge_model(
        "We use GPT-4 with temperature 0.7 as the model under test. "
        "Accuracy is computed by exact match against the reference.") is None
