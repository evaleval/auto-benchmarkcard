"""FINISH-6 composer fixes (synthetic fixtures only; no network, no real cards).

M1 HF/source subject-coherence, M2 size internal-consistency extension, M3 text-artifact scrub.
Named benchmarks below are encoded as minimal synthetic inputs and serve as regression fixtures,
never as name-keyed logic.
"""

from auto_benchmarkcard.tools.composer import composer_tool as C
from auto_benchmarkcard import workers as W


# ---------------------------------------------------------------------------------------------
# M1 -- HF/source subject-coherence (actionbench wrong-repo splice)

# The matched repo facebook/actionbench describes ActionMesh (3D-mesh-from-video), a DIFFERENT
# benchmark from the EEE identity "ActionBench" (the ADI text-to-image action-customization
# benchmark whose metrics grade generated images).
_ACTIONMESH_README = (
    "---\nlicense: other\ntask_categories:\n  - image-to-3d\npretty_name: ActionBench\n---\n\n"
    "# ActionBench: Paired Video-3D Synthetic Benchmark\n\n## Overview\n\n"
    "ActionBench is a benchmark dataset of 128 paired video and animated point-cloud samples for "
    "evaluating animated 3D mesh generation from video. The dataset consists of synthetic scenes "
    "of animated objects from ObjaverseXL, rendered using Blender.")

_ACTIONBENCH_EEE = {
    "benchmark_name": "ActionBench",
    "metrics": {
        "total_accuracy_on_actionbench": {
            "evaluation_description": "A strict metric measuring the percentage of generated images "
            "where both the action and the subject are correctly generated, as judged by human "
            "evaluators, evaluating overall success in action customization."},
        "subject_accuracy_on_actionbench": {
            "evaluation_description": "Measures the percentage of generated images where the "
            "character corresponds with the specified textual name, evaluating subject fidelity "
            "while performing a customized action."},
    },
}

# A correctly-bound repo: the README's leading description and the EEE identity describe the same
# subject (General AI Assistants), so the overlap is high and no conflict fires.
_GAIA_README = (
    "# GAIA dataset\n\nGAIA is a benchmark which aims at evaluating next generation LLMs (LLMs with "
    "augmented capabilities due to added tooling, efficient prompting, access to search). GAIA "
    "questions are conceptually simple for humans yet challenging for most advanced AIs.")
_GAIA_EEE = {
    "benchmark_name": "GAIA",
    "metrics": {"score_on_gaia": {
        "evaluation_description": "Accuracy on GAIA, a benchmark evaluating general AI assistants on "
        "real-world questions requiring reasoning, tool use, and web browsing."}},
}


def test_m1_subject_conflict_fires_on_disjoint_readme():
    # actionbench: the ActionMesh README is near-disjoint from the ActionBench identity -> conflict.
    assert C._hf_subject_conflicts(
        _ACTIONMESH_README, C._eee_identity_subject(_ACTIONBENCH_EEE)) is True


def test_m1_no_conflict_on_aligned_readme():
    # gaia: README and identity share the subject (General AI Assistants) -> no conflict.
    assert C._hf_subject_conflicts(
        _GAIA_README, C._eee_identity_subject(_GAIA_EEE)) is False


def test_m1_no_conflict_when_either_text_thin():
    # Missing/thin README or identity is undecidable -> never fire.
    assert C._hf_subject_conflicts("", C._eee_identity_subject(_ACTIONBENCH_EEE)) is False
    assert C._hf_subject_conflicts(_ACTIONMESH_README, "") is False
    assert C._hf_subject_conflicts("# Tiny\n\nA short blurb.", "ActionBench accuracy") is False


def test_m1_overlap_threshold_separates_actionbench_from_aligned():
    # The wrong-repo overlap must sit clearly below the correctly-bound ones (cleanly separated).
    def overlap(readme, eee):
        lead = C._content_words(C._readme_lead(readme))
        ident = C._content_words(C._eee_identity_subject(eee))
        inter = len(lead & ident)
        return max(inter / len(lead), inter / len(ident)) if lead and ident else None
    action = overlap(_ACTIONMESH_README, _ACTIONBENCH_EEE)
    gaia = overlap(_GAIA_README, _GAIA_EEE)
    assert action < C._SUBJECT_OVERLAP_FLOOR <= gaia


def test_m1_eee_identity_subject_joins_name_and_descriptions():
    subj = C._eee_identity_subject(_ACTIONBENCH_EEE)
    assert "ActionBench" in subj and "generated images" in subj
    assert C._eee_identity_subject(None) == ""
    assert C._eee_identity_subject({"benchmark_name": "X"}) == "X"


def _prov(src):
    return {"source": src, "evidence": "", "quote": ""}


def test_m1_drop_blanks_wrong_repo_overview_and_wrong_paper_fields():
    # overview spliced from the README (source=hf); methodology/data from the WRONG paper
    # (source=paper, wrong_paper indicator set) -> all blanked. The EEE-grounded metric (source=
    # established) and the correct name (source=paper but not blanked: it is not in the drop set)
    # are kept.
    card = {
        "benchmark_details": {"name": "ActionBench",
                              "overview": "ActionBench is a benchmark of 128 paired video samples."},
        "methodology": {"metrics": ["Accuracy"],
                        "calculation": "chamfer distance CD-3D from the ActionMesh paper.",
                        "methods": ["Per-frame 3D reconstruction quality is evaluated."]},
        "data": {"size": "128 animated scenes",
                 "source": "128 animated scenes from Objaverse rendered with Blender."},
    }
    prov = {
        "benchmark_details": {"overview": _prov("hf"), "name": _prov("paper")},
        "methodology": {"metrics": _prov("established"), "calculation": _prov("paper"),
                       "methods": _prov("paper")},
        "data": {"size": _prov("paper"), "source": _prov("paper")},
    }
    W._drop_subject_incoherent_hf(card, prov, wrong_paper=True)
    assert card["benchmark_details"]["overview"] == "Not specified"
    assert card["methodology"]["calculation"] == "Not specified"
    assert card["methodology"]["methods"] == ["Not specified"]
    assert card["data"]["size"] == "Not specified"
    assert card["data"]["source"] == "Not specified"
    # EEE-grounded metric and the correct name are NOT from a suspect source -> kept.
    assert card["methodology"]["metrics"] == ["Accuracy"]
    assert card["benchmark_details"]["name"] == "ActionBench"


def test_m1_drop_keeps_paper_fields_when_paper_is_right():
    # Subject conflict on the README (overview blanked) but the resolved paper is the RIGHT one
    # (wrong_paper False) -> the paper-grounded methodology/data fields are kept.
    card = {
        "benchmark_details": {"overview": "spliced README text"},
        "methodology": {"calculation": "the real calculation from the correct paper"},
        "data": {"size": "448 questions"},
    }
    prov = {
        "benchmark_details": {"overview": _prov("hf")},
        "methodology": {"calculation": _prov("paper")},
        "data": {"size": _prov("paper")},
    }
    W._drop_subject_incoherent_hf(card, prov, wrong_paper=False)
    assert card["benchmark_details"]["overview"] == "Not specified"
    assert card["methodology"]["calculation"] == "the real calculation from the correct paper"
    assert card["data"]["size"] == "448 questions"


def test_m1_precision_aligned_repos_not_dropped():
    # gpqa (no README -> thin), mbpp (high overlap), gaia (high overlap) never conflict.
    mbpp_readme = ("# Dataset Card for Mostly Basic Python Problems (mbpp)\n\nThe benchmark consists "
                   "of around 1,000 crowd-sourced Python programming problems designed to be solvable "
                   "by entry level programmers, covering programming fundamentals.")
    mbpp_eee = {"benchmark_name": "mbpp", "metrics": {"pass_at_1": {
        "evaluation_description": "pass@1 on the Mostly Basic Python Problems benchmark of "
        "crowd-sourced entry-level Python programming problems."}}}
    assert C._hf_subject_conflicts(mbpp_readme, C._eee_identity_subject(mbpp_eee)) is False
    assert C._hf_subject_conflicts("", C._eee_identity_subject(
        {"benchmark_name": "GPQA", "metrics": {}})) is False
    assert C._hf_subject_conflicts(_GAIA_README, C._eee_identity_subject(_GAIA_EEE)) is False


# ---------------------------------------------------------------------------------------------
# M2 -- size internal-consistency extension

def test_m2_breakdown_product():
    # activitynet: 203 classes x 137 videos/class (skip total/hours/average-rate-1 keys).
    bd = {"203 activity classes": 203, "average untrimmed videos per class": 137,
          "average activity instances per video": 1, "total video hours": 849}
    assert C._breakdown_product(bd) == 203 * 137
    # no per-key -> no multiplicative structure.
    assert C._breakdown_product({"train": 60, "test": 40}) is None
    assert C._breakdown_product({"203 classes": 203}) is None
    assert C._breakdown_product("Not specified") is None


def test_m2_ruleA2_product_breakdown_disproves_bucket():
    card = {"data": {"size": "Less than 1K examples", "source": "",
                     "size_breakdown": {"203 activity classes": 203,
                                        "average untrimmed videos per class": 137,
                                        "average activity instances per video": 1,
                                        "total video hours": 849}},
            "benchmark_details": {"overview": "A large-scale video benchmark for human activity "
                                  "understanding with 203 activity classes."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"


def test_m2_ruleA2_no_per_key_bucket_kept():
    # A split breakdown (no per-token) does not multiply -> the bucket is left for the other rules.
    card = {"data": {"size": "Less than 1K examples", "source": "",
                     "size_breakdown": {"train": 300, "test": 200}},
            "benchmark_details": {"overview": "A small benchmark."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Less than 1K examples"


def test_m2_ruleA3_specific_size_contradicts_prose_count():
    # gpqa: size "546 questions" (Extended) vs the lead's "448 multiple-choice questions" (main).
    card = {"data": {"size": "546 questions", "source": "", "size_breakdown": "Not specified"},
            "benchmark_details": {"overview": "A challenging dataset of 448 multiple-choice "
                                  "questions written by domain experts in biology, physics, and "
                                  "chemistry. Questions are Google-proof."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert C._size_count(card["data"]["size"]) == 448


def test_m2_ruleA3_specific_size_agreeing_prose_kept():
    # A specific size that AGREES with the lead count (within tolerance) is untouched.
    card = {"data": {"size": "448 questions", "source": "", "size_breakdown": "Not specified"},
            "benchmark_details": {"overview": "A dataset of 448 multiple-choice questions."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "448 questions"


def test_m2_beyond_aime_correct_bucket_kept():
    # beyond-aime: 100 problems, no breakdown, no prose count in lead -> bucket stays.
    card = {"data": {"size": "Less than 1K examples", "source": "", "size_breakdown": "Not specified"},
            "benchmark_details": {"overview": "Beyond AIME is a difficult mathematical reasoning "
                                  "benchmark sourced from competitions."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Less than 1K examples"


def test_m2_agieval_unchanged_no_signal():
    # agieval (DEFERRED): size_breakdown NS, lead states no example count, real 8062 only in the
    # external paper -> no in-card signal -> the bucket is left UNCHANGED (not over-abstained).
    card = {"data": {"size": "Less than 1K examples", "source": "", "size_breakdown": "Not specified"},
            "benchmark_details": {"overview": "A human-centric benchmark for evaluating foundation "
                                  "models on standardized exams. Contains 20 tasks (18 "
                                  "multiple-choice, 2 cloze)."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Less than 1K examples"


def test_m2_finish5_c2_rules_still_green():
    # FINISH-5 C2 Rule A / D regressions must remain.
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": "consisting of 2,294 software engineering problems"}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"
    card = {"data": {"size_breakdown": {"llmbar-natural": 907, "math-prm": 1608, "donotanswer": 745}},
            "benchmark_details": {"overview": "It consists of 2,985 prompt-chosen-rejected trios."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == "Not specified"


# ---------------------------------------------------------------------------------------------
# M3 -- text-artifact scrub

def test_m3_strips_control_word_suffix():
    # acadreason: stray ')Skip.' glued mid-prose -> repaired to a clean sentence boundary.
    t = ("All questions are sourced from top-tier publications in recent years)Skip. The dataset "
         "is constructed by selecting 430 papers from leading journals.")
    out = C._scrub_generation_artifacts(t)
    assert ")Skip." not in out and "Skip." not in out
    assert "in recent years. The dataset is constructed" in out


def test_m3_strips_bare_control_word_suffix():
    t = "Each task includes several fieldsContinue. The labels are human-verified."
    out = C._scrub_generation_artifacts(t)
    assert "Continue." not in out
    assert "Each task includes several fields. The labels are human-verified." == out


def test_m3_flags_glued_word_fragment_not_repaired():
    # acemath: glued 'problemhol' is flagged (leave-and-flag), never auto-repaired.
    calc = ("The rm@8 result is computed by randomly sampling 8 responses from the 64 candidates "
            "per problemhol, and the final accuracy is averaged over 100 random seeds.")
    assert C._scrub_generation_artifacts(calc) == calc  # not repaired
    assert C._glued_word_fragments(calc) == ["problemhol"]
    card = {"methodology": {"calculation": calc}}
    C._apply_generation_artifact_scrub(card)
    assert card["methodology"]["calculation"] == calc  # still not repaired
    assert "[Probable generation artifact]" in card["flagged_fields"]["methodology.calculation"]


def test_m3_glued_fragment_precision_real_derived_words_not_flagged():
    # Legitimate derived words (stem + real English suffix) must never be flagged as artifacts.
    for w in ("problematic", "questionable", "prompted", "sampled", "answered", "problems",
              "questioning", "benchmarked", "instances", "responsive"):
        assert C._glued_word_fragments(f"It is {w} in practice.") == [], w
    # but a stem fused to a non-word fragment still flags.
    assert C._glued_word_fragments("8 candidates per problemhol, averaged") == ["problemhol"]


def test_m3_trims_midsentence_truncation():
    # arc-agi out_of_scope: field-final clause ends on dangling 'if' -> trim to last full sentence.
    t = ("Task-specific performance is a perfectly appropriate measure of success if and only if "
         "handling the task as initially specified is the end goal of the system. However, it is "
         "deficient if")
    out = C._scrub_generation_artifacts(t)
    assert out.endswith("is the end goal of the system.")
    assert "However, it is deficient if" not in out


def test_m3_legit_if_clause_not_truncated():
    # A complete '...if X.' clause (terminal punctuation) must be kept verbatim.
    t = "However, it is deficient if the task specification changes after deployment."
    assert C._scrub_generation_artifacts(t) == t
    t2 = "Models with high recall feel responsive."  # no dangling tail, no marker
    assert C._scrub_generation_artifacts(t2) == t2


def test_m3_idempotent_and_clean_untouched():
    src = ("All questions are sourced from top-tier publications in recent years)Skip. The dataset "
           "is constructed by selecting 430 papers.")
    once = C._scrub_generation_artifacts(src)
    assert C._scrub_generation_artifacts(once) == once
    clean = "Baselines reach 65% accuracy on the held-out split."
    assert C._scrub_generation_artifacts(clean) == clean
    assert C._glued_word_fragments(clean) == []


def test_m3_apply_scrub_covers_data_source_and_out_of_scope():
    card = {
        "data": {"source": "Sourced from publications in recent years)Skip. The dataset has 430 papers."},
        "purpose_and_intended_users": {"out_of_scope_uses": [
            "Task-specific performance is appropriate if the goal is the task. However, it is "
            "deficient if"]},
    }
    C._apply_generation_artifact_scrub(card)
    assert ")Skip." not in card["data"]["source"]
    assert card["data"]["source"].startswith("Sourced from publications in recent years.")
    assert card["purpose_and_intended_users"]["out_of_scope_uses"][0].endswith(
        "if the goal is the task.")
