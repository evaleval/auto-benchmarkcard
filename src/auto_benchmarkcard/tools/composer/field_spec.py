"""Single source of truth for Stage-B field semantics.

Both the Stage-B prompt builder (composer_tool.py) and the validator
(validator.py) read from this module so the enum vocabularies, the set of
enum-controlled fields, the group split, and the per-field classification can
never drift apart. Nothing else in the codebase declares a Stage-B enum.

Enum scope is exactly the paths in ENUM_REGISTRY (Trap #5: if a prose field
ever entered this set the flaggable prose surface behind the Flag-F1 metric
would silently shrink). The deterministic enums (data_type / languages /
data_licensing) take their vocabulary live from the card_utils maps that
extract_hf_tags uses, so [0]'s _ALWAYS_OVERRIDE pass stays a byte-identical
no-op against what the validator considers valid.
"""

from typing import Optional, Set

from auto_benchmarkcard.card_utils import _LANG_CODE_MAP, _LICENSE_MAP

# Enum vocabularies. The deterministic ones are built live from card_utils.
DATA_TYPE_VOCAB = {
    "text", "image", "video", "audio", "tabular", "code", "interactive-environment",
}
BENCHMARK_TYPE_VOCAB = {"single", "composite_suite", "leaderboard_aggregate"}
AUDIENCE_VOCAB = {
    "AI researchers", "model developers", "safety evaluators",
    "industry practitioners", "policymakers", "educators",
}


def _languages_vocab() -> Set[str]:
    return set(_LANG_CODE_MAP.values()) | {"Multilingual", "not-applicable"}


def _license_vocab() -> Set[str]:
    return set(_LICENSE_MAP.values()) | {"Not specified"}


# The ONLY declaration of which fields are enum-controlled. path -> () -> vocab.
ENUM_REGISTRY = {
    "benchmark_details.data_type": lambda: DATA_TYPE_VOCAB,
    "benchmark_details.languages": _languages_vocab,
    "benchmark_details.benchmark_type": lambda: BENCHMARK_TYPE_VOCAB,
    "ethical_and_legal_considerations.data_licensing": _license_vocab,
    "purpose_and_intended_users.audience": lambda: AUDIENCE_VOCAB,
}

# Trap #5 guard: the enum scope must be exactly these five paths.
EXPECTED_ENUM_PATHS = frozenset({
    "benchmark_details.data_type",
    "benchmark_details.languages",
    "benchmark_details.benchmark_type",
    "ethical_and_legal_considerations.data_licensing",
    "purpose_and_intended_users.audience",
})
assert set(ENUM_REGISTRY) == EXPECTED_ENUM_PATHS, "ENUM_REGISTRY drifted from the agreed enum scope"

# Shape of each enum value.
ENUM_MULTI = {  # list-valued enums (each element checked)
    "benchmark_details.languages",
    "purpose_and_intended_users.audience",
}
ENUM_COMMA_JOINED = {"benchmark_details.data_type"}  # one lowercased comma-joined string
# (benchmark_type, data_licensing are plain scalars)

# Enums B genuinely authors -> hard inline check (a miss becomes schema_invalid + triggers repair).
B_AUTHORED_ENUMS = {
    "purpose_and_intended_users.audience",
}
# Deterministic enums [0] sets via _ALWAYS_OVERRIDE; B only echoes them. Validated
# softly (telemetry / other-rate) since [0]'s override is the source of truth and
# extract_hf_tags may legitimately emit an unmapped raw fallback.
ESTABLISHED_ENUMS = {
    "benchmark_details.data_type",
    "benchmark_details.languages",
    "ethical_and_legal_considerations.data_licensing",
}
# benchmark_type is EEE-injected by [0] post-composition and never appears in
# Stage-B output, so V never validates it inline; [0] enforces it at the inject site.

# Display-only (populated by [0] post-composition; B never authors; no evidence).
DISPLAY_FIELDS = {
    "benchmark_details.authors",
    "benchmark_details.logo",
    "benchmark_details.org_url",
}

# Deterministically established fields shown to B as "already filled"; B echoes them.
ESTABLISHED_FIELDS = {
    "benchmark_details.data_type",
    "benchmark_details.languages",
    "ethical_and_legal_considerations.data_licensing",
    "data.format",
    "data.size",
    "methodology.metrics",
}

# judge_setup is ONE evidence bucket in the digest but FOUR flat output fields.
JUDGE_SETUP_BUCKET = "methodology.judge_setup"
JUDGE_FLAT_FIELDS = [
    "judge_uses_llm", "judge_num", "judge_models", "judge_score_consolidation",
]

# Union[structured, str] fields: an empty/placeholder structured value collapses to NS.
COLLAPSE_FIELDS = {
    "data.size_breakdown",
    "methodology.judge_uses_llm",
    "methodology.judge_num",
}

# Fields whose value carries no extractable citation -> exempt from evidence-linkage
# (NS fields are also exempt, handled at runtime). Established/display fields, the new
# structured/scalar fields, plus identity URL/name fields.
EVIDENCE_EXEMPT = (
    ESTABLISHED_FIELDS
    | DISPLAY_FIELDS
    | {
        "benchmark_details.name",
        "benchmark_details.resources",
        "data.size_breakdown",
        "methodology.judge_uses_llm",
        "methodology.judge_num",
        "methodology.judge_models",
        "methodology.judge_score_consolidation",
    }
)

# List-valued fields (short field name) for list-hygiene + list-form NS sentinel.
LIST_FIELDS = {
    "domains", "languages", "similar_benchmarks", "resources",
    "audience", "tasks", "out_of_scope_uses",
    "methods", "metrics", "judge_models", "authors",
}
# Numeric caps for list-hygiene truncation (dotted path).
LIST_CAPS = {
    "benchmark_details.similar_benchmarks": 5,
    "purpose_and_intended_users.audience": 6,
    "purpose_and_intended_users.tasks": 12,
    "methodology.methods": 12,
}

# Human-readable caps / guidance for the prompt field-spec block (dotted path).
FIELD_CAPS = {
    "benchmark_details.overview": (
        "2-3 sentences; state the direct measurement criterion -- the literal measured "
        "object (primary metric + item/sample type + count or domain breadth), e.g. "
        "'accuracy on 817 expert-written multiple-choice items across 3 domains'"
    ),
    "benchmark_details.similar_benchmarks": "up to ~5 related benchmark names",
    "purpose_and_intended_users.goal": "1-3 sentences",
    "purpose_and_intended_users.limitations": (
        "1-3 sentences on benchmark shortcomings/biases/constraints; do NOT restate "
        "baseline or human scores"
    ),
    "data.collection_date": (
        "when the data was collected or its cutoff, as a year or year range "
        "(e.g. '2023', '2022-2024', '2023-06'); not the paper/card date; \"Not specified\" if none"
    ),
    "data.contamination_controls": (
        "canary string / held-out or private split / encryption, as stated in the source"
    ),
    "methodology.human_baseline": (
        "human or reference baseline result + its conditions; NOT model scores (those go in baseline_results)"
    ),
    "methodology.baseline_results": "model/baseline scores with numbers; include model conditions where reported",
    "methodology.validity_justification": (
        "only what the paper says about construct validity / metric-choice justification (extract-only)"
    ),
    # List content guidance (form is added generically for any LIST_FIELDS field).
    "benchmark_details.domains": "the subject areas / application domains the items cover, as stated",
    "purpose_and_intended_users.tasks": (
        "discrete evaluation tasks only; no scope-framing or what-it-is-not"
    ),
    "purpose_and_intended_users.out_of_scope_uses": (
        "concrete inappropriate/unsupported uses (a misuse case), NOT a restatement of "
        "intended scope; \"Not specified\" if none stated"
    ),
    "methodology.judge_models": (
        "bare model identifier(s) only (e.g. GPT-4o, self); put judging-policy prose in "
        "judge_score_consolidation"
    ),
    "methodology.calculation": (
        "how the score is computed/aggregated (formula/normalization/averaging), distinct from methods"
    ),
    "methodology.methods": (
        "the scoring/evaluation mechanism (e.g. zero-shot, exact-match grading), not the "
        "run/invocation command"
    ),
    "ethical_and_legal_considerations.compliance_with_regulations": (
        "regulatory compliance (GDPR/IRB/ethical review) only; do NOT restate the license "
        "(that is data_licensing); \"Not specified\" if none"
    ),
}

# Group definitions for Stage-B (each group is one LLM call). Default 3; pilot-switchable.
GROUP_DEFS_3 = [
    ("identity_purpose", ["benchmark_details", "purpose_and_intended_users"]),
    ("data_method", ["data", "methodology"]),
    ("ethics", ["ethical_and_legal_considerations"]),
]
GROUP_DEFS_2 = [
    ("core", ["benchmark_details", "purpose_and_intended_users", "data"]),
    ("method_ethics", ["methodology", "ethical_and_legal_considerations"]),
]
GROUP_DEFS_4 = [
    ("identity", ["benchmark_details"]),
    ("purpose", ["purpose_and_intended_users"]),
    ("data_method", ["data", "methodology"]),
    ("ethics", ["ethical_and_legal_considerations"]),
]
ACTIVE_GROUPS = GROUP_DEFS_3

ALL_SECTIONS = [
    "benchmark_details", "purpose_and_intended_users",
    "data", "methodology", "ethical_and_legal_considerations",
]


def assert_groups_cover(groups=ACTIVE_GROUPS) -> None:
    """Every section appears in exactly one group, and the union is the five sections."""
    seen = []
    for _, secs in groups:
        seen.extend(secs)
    assert sorted(seen) == sorted(ALL_SECTIONS), (
        f"group definitions must cover the 5 sections exactly once, got {seen}"
    )


def enum_vocab(path: str) -> Optional[Set[str]]:
    """The allowed vocabulary for an enum path, or None if the path is not enum-controlled."""
    provider = ENUM_REGISTRY.get(path)
    return provider() if provider else None


def field_class(path: str) -> str:
    """Coarse class used by the prompt field-spec block."""
    if path in DISPLAY_FIELDS:
        return "display"
    if path in ENUM_REGISTRY:
        return "enum"
    if path in ESTABLISHED_FIELDS:
        return "established"
    if path in COLLAPSE_FIELDS or path == "data.size_breakdown":
        return "structured"
    return "prose"
