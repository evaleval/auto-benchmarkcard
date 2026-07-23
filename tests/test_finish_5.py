"""FINISH-5 composer fixes (synthetic fixtures only; no network, no real cards).

C1 overview subject-coherence, C2 numeric internal-consistency, C3 coverage reconciliation,
C4 metric prose->structured promotion, C5 source-label scrub. Named benchmarks below are encoded
as minimal synthetic inputs and serve as regression fixtures, never as name-keyed logic.
"""

from auto_benchmarkcard.tools.composer import composer_tool as C


# ---------------------------------------------------------------------------------------------
# C1 -- overview subject-coherence

_ARC_AGI_DESC = (
    "The Abstraction and Reasoning Corpus (ARC) is a benchmark designed to measure human-like "
    "general fluid intelligence through grid-based reasoning tasks. It consists of 800 tasks "
    "(400 training, 400 evaluation) where each task presents input-output grids."
)


def test_c1_acronym_collision_is_conflict():
    # EEE slug "arc" describes ARC-AGI, but the resolved identity is the AI2 Reasoning Challenge.
    assert C._overview_subject_conflicts(_ARC_AGI_DESC, "AI2 Reasoning Challenge (ARC)") is True
    assert C._overview_subject_conflicts(_ARC_AGI_DESC, "Think you have Solved Question Answering? "
                                         "Try ARC, the AI2 Reasoning Challenge arc") is True


def test_c1_matching_expansion_no_conflict():
    desc = "The AI2 Reasoning Challenge (ARC) is a set of grade-school science questions."
    assert C._overview_subject_conflicts(desc, "AI2 Reasoning Challenge ARC") is False


def test_c1_no_acronym_head_no_conflict():
    # aider-polyglot husk: no `Full Name (ACRONYM)` head -> never suppressed.
    desc = ("A coding benchmark that evaluates LLMs on 225 challenging Exercism programming "
            "exercises across C++, Go, Java, JavaScript, Python, and Rust.")
    assert C._overview_subject_conflicts(desc, "Aider-Polyglot") is False


def test_c1_no_identity_no_conflict():
    assert C._overview_subject_conflicts(_ARC_AGI_DESC, "") is False


def test_c1_extract_facts_suppresses_conflicting_overview():
    eee = {"benchmark_name": "arc",
           "metrics": {"llm_stats.arc.score": {"metric_kind": "benchmark_score",
                                               "evaluation_description": _ARC_AGI_DESC}}}
    facts = C.extract_deterministic_facts(eee, None, {"paper_title": "Think you have Solved "
                                                      "Question Answering? Try ARC, the AI2 "
                                                      "Reasoning Challenge"})
    assert "benchmark_details.overview" not in facts


def test_c1_extract_facts_keeps_self_consistent_overview():
    desc = ("A coding benchmark that evaluates LLMs on 225 challenging Exercism programming "
            "exercises across six languages, with two attempts and test feedback after the first.")
    eee = {"benchmark_name": "aider-polyglot",
           "metrics": {"llm_stats.aider-polyglot.score": {"metric_kind": "benchmark_score",
                                                          "evaluation_description": desc}}}
    facts = C.extract_deterministic_facts(eee, None, None)
    assert facts.get("benchmark_details.overview")


# ---------------------------------------------------------------------------------------------
# C2 -- numeric internal-consistency

def test_c2_prose_example_count():
    assert C._prose_example_count("consisting of 2,294 software engineering problems drawn") == 2294
    assert C._prose_example_count("100 hard text-based questions that require reasoning") == 100
    assert C._prose_example_count("3,807 multiple-choice questions paired with images") == 3807
    # corpus / measure / identifier / per-rate are NOT example counts
    assert C._prose_example_count("a corpus of approximately 100,000 web documents") is None
    assert C._prose_example_count("a total of 849 video hours across the dataset") is None
    assert C._prose_example_count("sourced from GPT-4-0613, with seed examples from ODEX") is None
    assert C._prose_example_count("137 untrimmed videos per class") is None


def test_c2_size_bucket_range():
    assert C._size_bucket_range("Less than 1K examples") == (0, 1000)
    assert C._size_bucket_range("1K to 10K examples") == (1000, 10000)
    assert C._size_bucket_range("23,679 English text generation prompts") is None


def test_c2_rule_a_bucket_contradicted_by_prose():
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": "consisting of 2,294 software engineering problems"}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"


def test_c2_rule_a_in_bucket_count_is_kept():
    card = {"data": {"size": "1K to 10K examples", "source": ""},
            "benchmark_details": {"overview": "reformats tasks into 3,807 multiple-choice questions"}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "1K to 10K examples"


def test_c2_rule_a_margin_protects_intermediate_count():
    # "1,005 ... from the previous stage, 830 passed" must not unseat a correct n<1K tag.
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": "of the 1,005 question-answer pairs from the previous "
                                  "stage, 830 passed human verification"}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Less than 1K examples"


def test_c2_rule_b_shard_host_abstains_when_unconfirmed():
    # media/archive-hosted bucket with no prose count to confirm -> abstain.
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": "A large-scale video benchmark with 203 activity "
                                  "classes and 137 untrimmed videos per class."}}
    C._enforce_numeric_consistency(card, {"_size_is_shard_host": True}, {})
    assert card["data"]["size"] == "Not specified"


def test_c2_rule_b_shard_host_kept_when_prose_confirms():
    # aa-lcr: archive-hosted but the prose confirms a small set -> keep the bucket.
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": "AA-LCR includes 100 hard text-based questions."}}
    C._enforce_numeric_consistency(card, {"_size_is_shard_host": True}, {})
    assert card["data"]["size"] == "Less than 1K examples"


# FINISH-8 Fix 3 (Rule A2b) -- a prose MULTIPLICATIVE magnitude unseats a too-small bucket even when
# _size_is_shard_host is unset (file_list unavailable / blocked source) and no structured breakdown
# exists. Ports the gate's _prose_magnitude into the composer (the asymmetry that let the gate catch
# activitynet but the composer pass it). The REAL activitynet overview text is used (with the literal
# "average of N ... per <unit>"), not the earlier paraphrase that relied on the shard-host flag.

_ACTIVITYNET_OVERVIEW = ("Provides samples from 203 activity classes with an average of 137 untrimmed "
                         "videos per class and 1.41 activity instances per video, for a total of 849 "
                         "video hours.")


def test_c2_prose_magnitude_unit():
    # 203 classes x 137 untrimmed VIDEOS (content noun) per class = 27811. Needs both the rate and a
    # matching "<M> <unit>" count; the no-"average of" paraphrase and a lone integer do not trigger it.
    assert C._prose_magnitude(_ACTIVITYNET_OVERVIEW) == 27811
    assert C._prose_magnitude("203 activity classes and 137 untrimmed videos per class") is None
    assert C._prose_magnitude("a dataset of 27,811 examples") is None


def test_c2_prose_magnitude_measure_noun_denylist():
    # FINISH-8 fix-round: the COUNTED noun must be an example-ish CONTENT noun. A measure/sub-item rate
    # (frames/tokens/annotations/hours per <unit>) is NOT an example yield -> None (never false-drops a
    # correct size). The content-noun rate ("videos per class") still computes.
    assert C._prose_magnitude("5,000 videos, with an average of 30 frames per video.") is None
    assert C._prose_magnitude("1,000 documents with an average of 512 tokens per document.") is None
    assert C._prose_magnitude("100 questions, with an average of 3 annotations per question.") is None
    assert C._prose_magnitude("a corpus with an average of 2 hours per recording across 100 recordings"
                              ) is None
    # content counted noun (questions) with a regular-plural per-unit (class->classes) -> computes
    assert C._prose_magnitude("50 classes with an average of 100 questions per class.") == 5000
    # a counted noun that is not generic-content but EQUALS the size bucket's own unit is accepted
    assert C._prose_magnitude("10 shards with an average of 500 widgets per shard.",
                              bucket_unit="widgets") == 5000
    # conservative gap (fails SAFE -> keep, never false-drop): an irregular-plural per-unit whose
    # base count cannot be re-matched (category->categories) abstains rather than asserting a number.
    assert C._prose_magnitude("50 categories with an average of 100 questions per category.") is None


def test_c2_prose_magnitude_anchors_M_not_rate_number():
    # The nit: the M-count must come from the magnitude unit's own count, not the per-rate's number.
    # "3,807 items ... average of 5 problems per item" -> 3807 x 5 = 19035 (not 5 x 5).
    assert C._prose_magnitude("3,807 items, with an average of 5 problems per item.") == 19035


def test_c2_rule_a2b_prose_magnitude_abstains():
    # activitynet shape: file_list unavailable so _size_is_shard_host is unset (the live gap), no
    # structured size_breakdown -- only the prose magnitude can fire -> abstain to NS.
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": _ACTIVITYNET_OVERVIEW}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"


def test_c2_rule_a2b_small_perrate_kept():
    # A legitimate small dataset with an unrelated "per" rate (product 300 << 10000*1.5) plus a real
    # in-bucket count is NOT dropped -- the prose magnitude is over-count direction only.
    card = {"data": {"size": "1K to 10K examples", "source": ""},
            "benchmark_details": {"overview": "Contains 3,807 multiple-choice questions, with an "
                                  "average of 3 annotations per question."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "1K to 10K examples"


def test_c2_rule_a2b_no_double_signal_when_breakdown_present():
    # When a per-structured breakdown already disproves the bucket (Rule A2), the prose path is not
    # needed; the result is still NS. Guards that Rule A2b does not regress Rule A2.
    card = {"data": {"size": "Less than 1K examples", "source": "",
                     "size_breakdown": {"classes": 203, "avg videos per class": 137}},
            "benchmark_details": {"overview": _ACTIVITYNET_OVERVIEW}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"


def test_c2_rule_a2b_measure_rate_does_not_false_drop():
    # FINISH-8 fix-round (multimodal false-drop): "average of 30 frames per video" over M videos must
    # NOT unseat a correct video-count bucket -- frames is a measure noun, not an example yield.
    card = {"data": {"size": "1K to 10K examples", "source": ""},
            "benchmark_details": {"overview": "A video benchmark containing 5,000 videos, with an "
                                  "average of 30 frames per video sampled for evaluation."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "1K to 10K examples"


def test_c2_rule_a2b_content_rate_still_drops():
    # The activitynet content-noun case still abstains (regression guard for the measure-noun fix).
    card = {"data": {"size": "Less than 1K examples", "source": ""},
            "benchmark_details": {"overview": _ACTIVITYNET_OVERVIEW}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size"] == "Not specified"


def test_c2_hf_size_is_shard_count():
    parquet = {"file_list": ["data/train-00000.parquet", "README.md"], "dataset_info": None}
    archive = {"file_list": ["v1-2_train.tar.gz", "v1-2_train.tar.gz.00", "classes.txt"],
               "dataset_info": None}
    with_info = {"file_list": ["v1.tar.gz"], "dataset_info": {"splits": [{"name": "test",
                                                                          "num_examples": 5}]}}
    assert C._hf_size_is_shard_count(parquet) is False     # parquet -> trust
    assert C._hf_size_is_shard_count(archive) is True       # archive shards -> abstain
    assert C._hf_size_is_shard_count(with_info) is False    # authoritative row count -> trust


def test_c2_rule_c_percentage_breakdown_dropped():
    card = {"data": {"size_breakdown": {"validation": 10, "public test": 60, "private test": 30}}}
    prov = {"data": {"size_breakdown": {"evidence": "validation (10%), public (60%), private (30%)"}}}
    C._enforce_numeric_consistency(card, {}, prov)
    assert card["data"]["size_breakdown"] == "Not specified"


def test_c2_rule_c_percentage_converted_with_known_total():
    card = {"data": {"size_breakdown": {"validation": 10, "public test": 60, "private test": 30}}}
    prov = {"data": {"size_breakdown": {"evidence": "split into (10%), (60%), (30%)"}}}
    C._enforce_numeric_consistency(card, {"data.total_examples": 440}, prov)
    assert card["data"]["size_breakdown"] == {"validation": 44, "public test": 264, "private test": 132}


def test_c2_rule_d_partition_overshoot_dropped():
    # token-length column read as counts: sum overshoots the stated trio total.
    card = {"data": {"size_breakdown": {"llmbar-natural": 907, "math-prm": 1608, "donotanswer": 745}},
            "benchmark_details": {"overview": "It consists of 2,985 prompt-chosen-rejected trios."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == "Not specified"


def test_c2_rule_d_heterogeneous_breakdown_kept():
    card = {"data": {"size_breakdown": {"# Task": 1140, "# Domain": 7, "# Library Combo": 577}},
            "benchmark_details": {"overview": "1,140 programming tasks across 7 domains."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert isinstance(card["data"]["size_breakdown"], dict)


def test_c2_rule_d_subgroup_breakdown_kept():
    # bold: subgroup counts (not example counts); sum is far below the total -> no overshoot, kept.
    card = {"data": {"size": "23,679 English text generation prompts",
                     "size_breakdown": {"Profession": 18, "Gender": 2, "Race": 4, "Total": 43}},
            "benchmark_details": {"overview": "23,679 English text generation prompts."}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == {"Profession": 18, "Gender": 2, "Race": 4, "Total": 43}


# ---------------------------------------------------------------------------------------------
# FINISH-7 G2-2 -- size_breakdown stated-Total vs sum-of-parts self-consistency (Rule E)

def _mmlu_pro_breakdown(total):
    # 14 disciplines summing to 12032, plus a stated Total.
    return {"Mathematics": 1351, "Physics": 1299, "Chemistry": 1132, "Law": 1101, "Engineering": 969,
            "Other": 924, "Economics": 844, "Health": 818, "Psychology": 798, "Business": 789,
            "Biology": 717, "Philosophy": 499, "Computer Science": 410, "History": 381, "Total": total}


def test_g2_2_total_overshot_by_parts_dropped_when_uncorroborated():
    # mmlu-pro: stated Total=6810 but the per-discipline parts sum to 12032 (overshoot). With no
    # authoritative total to corroborate a complete partition, ABSTAIN: drop the contradictory Total,
    # keep the clean per-discipline parts. Never assert a recomputed (possibly wrong-incomplete) total.
    card = {"data": {"size_breakdown": _mmlu_pro_breakdown(6810)}}
    C._enforce_numeric_consistency(card, {}, {})
    sb = card["data"]["size_breakdown"]
    assert isinstance(sb, dict) and "Total" not in sb
    assert sb["Mathematics"] == 1351 and len(sb) == 14


def test_g2_2_total_overshot_recomputed_when_det_total_corroborates():
    # When the HF authoritative example total equals the parts sum, the parts ARE a complete partition,
    # so the stated Total is the only wrong field -> recompute it to the corroborated sum (12032).
    card = {"data": {"size_breakdown": _mmlu_pro_breakdown(6810)}}
    C._enforce_numeric_consistency(card, {"data.total_examples": 12032}, {})
    assert card["data"]["size_breakdown"]["Total"] == 12032


def test_g2_2_consistent_total_untouched():
    card = {"data": {"size_breakdown": {"train": 60, "test": 40, "Total": 100}}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == {"train": 60, "test": 40, "Total": 100}


def test_g2_2_undershoot_total_kept():
    # Parts summing BELOW the stated Total is legitimate (incomplete enumeration / dimension counts);
    # only overshoot is a contradiction. (Same shape as the BOLD subgroup control.)
    card = {"data": {"size_breakdown": {"Profession": 18, "Gender": 2, "Race": 4, "Total": 43}}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == {"Profession": 18, "Gender": 2, "Race": 4, "Total": 43}


def test_g2_2_heterogeneous_total_untouched():
    # A '# Tasks'/'# Domains' statistic object is not a summable partition -> never corrected even on
    # an apparent overshoot.
    card = {"data": {"size_breakdown": {"# Tasks": 8000, "# Domains": 5000, "Total": 100}}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"]["Total"] == 100


def test_g2_2_no_total_key_untouched():
    card = {"data": {"size_breakdown": {"train": 60, "test": 40}}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == {"train": 60, "test": 40}


def test_g2_2_total_substring_stat_key_not_treated_as_partition_total():
    # A "Total Tokens"/"Subtotal"-style key must NOT be mistaken for the partition total (matched by
    # normalized equality to "total", not substring). With a genuine overshoot Total present, only the
    # real Total is dropped; the stat key is preserved untouched.
    card = {"data": {"size_breakdown": {"Math": 5000, "Physics": 8000, "Total Tokens": 999,
                                        "Total": 6810}}}
    C._enforce_numeric_consistency(card, {}, {})
    sb = card["data"]["size_breakdown"]
    assert sb["Total Tokens"] == 999                      # stat key preserved (not the partition total)
    assert "Total" not in sb                              # real overshoot Total (13000>6810) dropped
    assert sb["Math"] == 5000 and sb["Physics"] == 8000


def test_g2_2_subtotal_only_not_touched():
    # "Subtotal" alone is not the partition total -> Rule E does not fire, breakdown unchanged.
    card = {"data": {"size_breakdown": {"A": 5000, "B": 8000, "Subtotal": 100}}}
    C._enforce_numeric_consistency(card, {}, {})
    assert card["data"]["size_breakdown"] == {"A": 5000, "B": 8000, "Subtotal": 100}


# ---------------------------------------------------------------------------------------------
# C3 -- coverage reconciliation backstops

def test_c3_concise_license():
    q = ("All charts are subjected to their respective copyrights by the authors from their arXiv "
         "preprints. All QAs are licensed under CC BY-SA 4.0. Our code is licensed under Apache 2.0.")
    assert C._concise_license(q) == "CC BY-SA 4.0; Apache 2.0"
    q2 = "The benchmark is licensed under the MIT License with an additional clause in `LICENSE`."
    assert "MIT" in C._concise_license(q2)


def test_c3_marker_evidence():
    assert C._is_marker_evidence("No out-of-scope uses are stated in the evidence.") is True
    assert C._is_marker_evidence("Not specified") is True
    assert C._is_marker_evidence("All QAs are licensed under CC BY-SA 4.0.") is False


def test_c3_tier1_lifts_license_from_provenance():
    card = {"ethical_and_legal_considerations": {"data_licensing": "Not specified"}}
    prov = {"ethical_and_legal_considerations": {"data_licensing": {
        "source": "paper", "evidence": "All QAs are licensed under CC BY-SA 4.0."}}}
    C.apply_coverage_backstops(card, prov, {})
    assert "CC BY-SA 4.0" in card["ethical_and_legal_considerations"]["data_licensing"]


def test_c3_tier1_no_provenance_stays_ns():
    card = {"ethical_and_legal_considerations": {"data_licensing": "Not specified"}}
    C.apply_coverage_backstops(card, {}, {})
    assert card["ethical_and_legal_considerations"]["data_licensing"] == "Not specified"


def test_c3_tier1_marker_provenance_not_lifted():
    card = {"ethical_and_legal_considerations": {"data_licensing": "Not specified"}}
    prov = {"ethical_and_legal_considerations": {"data_licensing": {
        "source": "paper", "evidence": "No license is stated in the paper."}}}
    C.apply_coverage_backstops(card, prov, {})
    assert card["ethical_and_legal_considerations"]["data_licensing"] == "Not specified"


def test_c3_tier2_code_languages_not_applicable():
    card = {"benchmark_details": {"languages": ["Not specified"],
                                  "overview": "1,140 fine-grained programming tasks in Python."},
            "methodology": {"metrics": ["other:x.score"], "methods": []}}
    C.apply_coverage_backstops(card, {}, {})
    assert card["benchmark_details"]["languages"] == ["not-applicable"]


def test_c3_tier2_passk_metric_is_code_signal():
    card = {"benchmark_details": {"languages": ["Not specified"], "overview": "tasks"},
            "methodology": {"metrics": ["pass@1"], "methods": []}}
    C.apply_coverage_backstops(card, {}, {})
    assert card["benchmark_details"]["languages"] == ["not-applicable"]


def test_c3_tier2_non_code_unchanged():
    # incidental "programmatic" must not trip the strong code signal.
    card = {"benchmark_details": {"languages": ["Not specified"],
                                  "overview": "116 programmatic tasks in a GUI environment."},
            "methodology": {"metrics": ["other:x.score"], "methods": ["Success rates are reported."]}}
    C.apply_coverage_backstops(card, {}, {})
    assert card["benchmark_details"]["languages"] == ["Not specified"]


def test_c3_tier2_existing_languages_untouched():
    card = {"benchmark_details": {"languages": ["English"], "overview": "programming tasks"},
            "methodology": {"metrics": ["pass@1"]}}
    C.apply_coverage_backstops(card, {}, {})
    assert card["benchmark_details"]["languages"] == ["English"]


# ---------------------------------------------------------------------------------------------
# C4 -- metric prose -> structured promotion

def test_c4_promote_targets():
    assert C._metric_from_prose("Success rates (SR) are presented for agents.") == ["success-rate"]
    assert C._metric_from_prose("Accuracy is calculated as #correct/#total. ROUGE-L is used.") == \
        ["Accuracy", "ROUGE"]
    assert C._metric_from_prose("evaluated using the area under the receiver operating "
                                "characteristic curve (AUC) metric") == ["AUC"]
    assert C._metric_from_prose("using evaluation methods (e.g., word-level F1 for strings)") == ["F1"]
    assert C._metric_from_prose("Accuracy follows BrowseComp: an LLM-as-judge compares answers.") == \
        ["Accuracy"]
    assert C._metric_from_prose("Pass@1 is calculated as the primary metric.") == ["pass@1"]


def test_c4_composites_not_promoted():
    assert C._metric_from_prose("We use an LLM-based equality checker to evaluate responses.") == []
    assert C._metric_from_prose("The Adjusted Rand Index and Silhouette Score yield values in "
                                "[-1, 1]; 1 signifies perfect, -1 incorrect assignments.") == []
    assert C._metric_from_prose("The primary metric is the Expansion Rate, with Shrinkage and "
                                "Preservation as diagnostics.") == []
    assert C._metric_from_prose("Not specified") == []


def test_c4_definition_context_gate():
    # incidental mention without a definition cue -> not promoted
    assert C._metric_from_prose("Models with high recall feel responsive to users.") == []
    # with a cue -> promoted
    assert C._metric_from_prose("Recall is computed over the retrieved set.") == ["recall"]


def test_c4_map_is_case_sensitive():
    assert C._metric_from_prose("The map of the maze is measured at each step.") == []
    assert C._metric_from_prose("mean average precision is reported.") == ["mAP"]


# ---------------------------------------------------------------------------------------------
# C5 -- prose source-label scrub

def test_c5_strips_leading_source_label():
    t = "From Every Eval Ever, Granite 3.3 8B Base scored 0.8850 on the AttaQ score metric."
    out = C._scrub_pipeline_jargon(t)
    assert "Every Eval Ever" not in out
    assert "Granite 3.3 8B Base scored 0.8850" in out
    assert out.startswith("Granite")


def test_c5_strips_inline_source_label():
    t = "Based on 3 model evaluations from Every Eval Ever: mean AA-Index score = 0.6450."
    out = C._scrub_pipeline_jargon(t)
    assert "Every Eval Ever" not in out
    assert "Based on 3 model evaluations: mean AA-Index score = 0.6450." == out


def test_c5_idempotent_and_clean_untouched():
    t = "Evaluation results from Every Eval Ever include: nvidia/Llama: 0.9411."
    once = C._scrub_pipeline_jargon(t)
    assert C._scrub_pipeline_jargon(once) == once
    clean = "Baselines reach 65% accuracy on the held-out split."
    assert C._scrub_pipeline_jargon(clean) == clean


def test_c5_ieee_preserved():
    t = "Published at IEEE CVPR 2024 with strong results."
    assert C._scrub_pipeline_jargon(t) == t


def test_c5_existing_jargon_still_dropped():
    # the pre-existing ergon/rollout sentence-drop behaviour must remain.
    t = "This uses an ergon-native rollout-card. The metric is accuracy."
    out = C._scrub_pipeline_jargon(t)
    assert "ergon-native" not in out and "accuracy" in out


# ---------------------------------------------------------------------------------------------
# FINISH-7 G2-1 -- baseline_results grounding guard (drop fabricated leaderboards)

def _bres(text):
    return {"methodology": {"baseline_results": text}}


# A REALISTIC production det_facts: extract_deterministic_facts stores evaluation_summary
# unconditionally, and run_composer passes the real det_facts. The top_performer models carry
# version digits in their NAMES (claude-3-5-sonnet, gpt-4o) -- the bug-hiding case: an earlier impl
# parsed model:score TEXT and let those name digits (3,5,4) serve as the grounding pool, so the
# fabrication shipped. These tests run the REAL path (non-empty det_facts) to pin that closed.
_REAL_DET = {"evaluation_summary": {
    "primary_metric": "accuracy",
    "top_performers": [{"model": "claude-3-5-sonnet", "developer": "anthropic", "score": 0.887},
                       {"model": "gpt-4o", "developer": "openai", "score": 0.864}],
    "score_statistics": {"mean": 0.66, "std_dev": 0.05, "min": 0.60, "max": 0.887}}}


def test_g2_1_fabricated_leaderboard_dropped_on_real_det_facts_path():
    # mmlu-pro fixture ON THE REAL PATH: a fabricated leaderboard whose percents (91.50/90.99/89.20)
    # match no top_performer score at any scale. The model-name version digits in _REAL_DET (3,5,4)
    # must NOT serve as grounding -> every sentence ungrounded -> whole field NS + flag.
    card = _bres("Claude Fable 5 leads with 91.50% accuracy, followed by Gemini 3.1 Pro Preview at "
                 "90.99%. GPT-5 reaches 89.20% on the benchmark.")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    assert card["methodology"]["baseline_results"] == "Not specified"
    assert "methodology.baseline_results" in card.get("flagged_fields", {})


def test_g2_1_pool_has_no_model_name_version_digits():
    # The grounded pool is built from STRUCTURED score values, never re-parsed from model:score text,
    # so a model-name version digit (3 / 5 / 4 from claude-3-5-sonnet / gpt-4o) can never enter it.
    pool = C._grounded_score_pool([], _REAL_DET)
    assert 3.0 not in pool and 5.0 not in pool and 4.0 not in pool
    assert 0.887 in pool and 0.864 in pool                         # the real structured scores


def test_g2_1_real_top_performer_score_reformatted_survives():
    # A REAL top_performer score (0.887) restated as a percent ("88.70%") must NOT be false-dropped:
    # the cross-scale match recognizes 0.887 == 88.70%/100. Kept, no flag.
    card = _bres("claude-3-5-sonnet scored 88.70% overall accuracy on the held-out set.")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    assert card["methodology"]["baseline_results"].startswith("claude-3-5-sonnet scored 88.70%")
    assert "methodology.baseline_results" not in card.get("flagged_fields", {})


def test_g2_1_name_version_digit_is_not_a_score():
    # A name-glued version digit ("Gemini 3.1", "GPT-5", "Fable 5") carries no score context (no '%',
    # not a [0,1] decimal) -> not a stated score, so it neither grounds nor triggers a drop on its own.
    # The only stated score in the first string is the 90.99% percent (-> ~0.9099), NOT the 3.1.
    scores = C._stated_score_numbers("Gemini 3.1 Pro at 90.99%")
    assert len(scores) == 1 and abs(scores[0] - 0.9099) < 1e-9
    assert C._stated_score_numbers("GPT-5 and Claude Fable 5 lead the pack") == []


def test_g2_1_tight_cross_scale_does_not_false_ground():
    # The tolerance is tight: a fabricated 91.50% (0.9150) must NOT match a real 0.887 (a loose 5pp
    # band would have). A genuine reformat / rounding still matches.
    assert C._score_is_grounded(0.9150, [0.887]) is False
    assert C._score_is_grounded(0.887, [0.887]) is True
    assert C._score_is_grounded(0.887, [0.8874]) is True          # rounding noise tolerated
    assert C._score_is_grounded(0.7184, [71.84]) is True          # cross-scale pool value


def test_g2_1_grounded_from_evidence_quote_survives():
    # A baseline whose percent is explicit in a verified evidence quote is grounded -> kept.
    card = _bres("A strong model achieves 70.20% accuracy on the math split.")
    ev = [{"quote": "The strongest system achieves 70.20% accuracy in the Mathematics category."}]
    C._drop_ungrounded_baselines(card, ev, {})
    assert card["methodology"]["baseline_results"].startswith("A strong model achieves 70.20%")
    assert "methodology.baseline_results" not in card.get("flagged_fields", {})


def test_g2_1_qualitative_no_number_untouched():
    # A purely-qualitative baseline sentence (no stated score) is not adjudicated -> kept, no flag.
    card = _bres("Larger instruction-tuned models generally outperform smaller base models here.")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    assert card["methodology"]["baseline_results"].startswith("Larger instruction-tuned")
    assert "methodology.baseline_results" not in card.get("flagged_fields", {})


def test_g2_1_grounded_sentence_kept_when_fabricated_majority():
    # finding [3]: with a fabricated MAJORITY, the grounded sentence must still survive (no over-drop
    # to NS) -- only the ungrounded sentences are dropped, whole-field NS only if NO grounded survives.
    card = _bres("claude-3-5-sonnet scored 88.70%. Claude Fable 5 hit 91.50%. "
                 "Made-up Model X hit 99.90%. Another fake reached 77.70%.")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    out = card["methodology"]["baseline_results"]
    assert "88.70%" in out                                        # grounded survives
    assert "91.50" not in out and "99.90" not in out and "77.70" not in out  # fabricated dropped
    assert out != "Not specified"
    assert "methodology.baseline_results" in card.get("flagged_fields", {})


def test_g2_1_mixed_numbers_in_one_sentence_kept_and_flagged():
    # A single sentence with one grounded (0.887) AND one ungrounded (99.90%) score is ambiguous ->
    # KEEP (a real baseline must not be lost to a co-located stray number) but FLAG for audit.
    card = _bres("claude-3-5-sonnet scored 88.70% while a mystery model hit 99.90%.")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    assert "88.70%" in card["methodology"]["baseline_results"]
    assert "99.90" in card["methodology"]["baseline_results"]
    assert "methodology.baseline_results" in card.get("flagged_fields", {})


def test_g2_1_ns_baseline_untouched():
    card = _bres("Not specified")
    C._drop_ungrounded_baselines(card, [], _REAL_DET)
    assert card["methodology"]["baseline_results"] == "Not specified"
    assert "flagged_fields" not in card or "methodology.baseline_results" not in card["flagged_fields"]
