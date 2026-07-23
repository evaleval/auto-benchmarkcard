"""Post-Stage-B aboutness guard: a structural, model-independent backstop.

The Stage-A verifier checks a quote is a verbatim substring of the source, not
whether the quote is ABOUT the field it was routed to. A faithful quote routed to
the wrong field therefore passes verification. This guard inspects the evidence
TEXT behind two failure-prone clusters and corrects the field when the evidence is
off-topic:

  judge_* cluster -- only an LLM that SCORES model outputs makes judge_uses_llm true.
    Evidence about how the dataset was built or annotated (e.g. "13 authors as human
    annotators", "Code Interpreter session in the web-based GPT-4") is not a judge.
  data.source -- the dataset's origin, not how a subset/variant (e.g. a "Hard" split)
    was constructed.

Everything here is keyword/regex over the verified quotes; it is independent of the
composer model, so its effect does not depend on a model re-pilot.
"""

from __future__ import annotations

import re
from typing import Dict, List

from auto_benchmarkcard.card_utils import is_not_specified
from auto_benchmarkcard.tools.composer import field_spec as fs

# Unambiguous LLM-judge / output-scoring terms. Wide on purpose: a missed marker
# demotes a CORRECT LLM-judge card (e.g. aa-lcr's "equality checker model"). The
# construction veto below protects precision against the opposite error. A "reward
# model" / "critic model" denotes a discriminative scalar scorer (a classifier), not
# a generative LLM-as-judge, so it is vetoed by _DISCRIMINATIVE_SCORER_RE below rather
# than counted here.
_JUDGE_TERM_RE = re.compile(
    r"\bjudges?\b|\bjudging\b|as an? (?:judge|evaluator)|as the (?:judge|evaluator)|"
    r"llm[- ]as[- ]?a?[- ]?judge|\bevaluator\b|automatic evaluator|\bgrader\b|"
    r"model[- ]graded|model[- ]based evaluation|\bverdict\b|"
    r"\brubric\b|\bverifier\b|\bchecker\b|equality checker|correctness checker|"
    r"answer match(?:ing)?|match(?:es)? the reference",
    re.IGNORECASE,
)

# A scoring verb sitting near an output noun ("evaluate the responses", "score each
# answer"). The output-noun requirement is what separates "rate the RESPONSE" (scoring)
# from "rate the task DIFFICULTY" (annotation).
_SCORE_VERB_RE = re.compile(
    r"\b(?:scor\w*|grad\w*|rat\w*|assess\w*|evaluat\w*|rank\w*|check\w*|match\w*)\b",
    re.IGNORECASE,
)
_OUTPUT_NOUN_RE = re.compile(
    r"\b(?:response|answer|output|completion|generation|prediction|candidate)s?\b|"
    r"model output",
    re.IGNORECASE,
)
_PROXIMITY = 50  # chars on either side of a scoring verb to look for an output noun

# Data construction / annotation markers. When present, the quote is about building
# the dataset, not scoring outputs, so it never counts as judge evidence.
_CONSTRUCTION_VETO_RE = re.compile(
    r"annotat|human annotator|crowd ?worker|during construction|to (?:build|construct) the|"
    r"task difficulty|seed examples?|test case generation|refactor|label the data",
    re.IGNORECASE,
)

# Programmatic / execution-based scoring: positive evidence that scoring is NOT an LLM
# judge, which is what licenses judge_uses_llm=False (a derived claim).
_EXEC_SCORING_RE = re.compile(
    r"pass@|pass @|unit ?test|test cases?|exact match|execution|executed|"
    r"functional correctness|sandbox|compiles?|run the code",
    re.IGNORECASE,
)

# Discriminative scalar scorer: a reward/ranking model or classifier that emits a
# numeric score (a regression/classification head), NOT a generative LLM-as-judge that
# produces a verdict. attaq's DeBERTa "reward model" is this. It is also non-LLM-judge
# scoring evidence, so it licenses judge_uses_llm=False (a derived claim).
_DISCRIMINATIVE_SCORER_RE = re.compile(
    r"\branking model\b|\breward model\b|\bclassifier\b|\bdeberta\b|\broberta\b|"
    r"scores?\s+indicating\s+the\s+likelihood|min-max\s+normaliz|scalar\s+(?:reward|score)",
    re.IGNORECASE,
)
# A genuine generative LLM-judge marker. Its presence spares a quote from the
# discriminative veto (e.g. "Nemotron-4 340B Reward model is used as the judge").
_GENERATIVE_JUDGE_RE = re.compile(
    r"\bjudges?\b|\bjudging\b|equality checker|llm[- ]as|\brubric\b|graded by|"
    r"as an? (?:judge|evaluator)|as the (?:judge|evaluator)",
    re.IGNORECASE,
)

# Subset / variant construction: how a slice (e.g. a "Hard" split) was selected,
# distinct from where the original data came from.
_SUBSET_CONSTRUCTION_RE = re.compile(
    r"\bsubset\b|-hard\b|hard subset|\bvariant\b|after deduplicat|deduplicat|"
    r"\bthreshold\b|bridge the query|sentence embedding|all-mpnet|"
    r"resulting in [\d,]+|filtered (?:down )?to [\d,]+|sampl\w* [\d,]+ (?:from|of)",
    re.IGNORECASE,
)

# Genuine provenance: where the original data actually came from.
_PROVENANCE_RE = re.compile(
    r"sourced from|collected from|crawl\w*|scrap\w*|drawn from|gathered from|compiled from|"
    r"sampled from|seed examples?|derived from the|based on the .*?(?:dataset|corpus)|"
    r"\bextends\b|built on top of|written by|authored by|generated by|synthesi[sz]ed|"
    r"github|stack ?overflow|wikipedia|common crawl",
    re.IGNORECASE,
)

# Stage-A evidence-routing markers (used by evidence.route_evidence, not the post-Stage-B
# guards below). They re-file or drop verified quotes by topic before the digest is built,
# so a faithful-but-mis-tagged quote reaches the right field. Regex-only, model-independent.

# Construction / item-modification not covered by _CONSTRUCTION_VETO_RE: a quote describing
# how the dataset was built or its items perturbed belongs in data.annotation, not tasks.
_CONSTRUCTION_INTRO_RE = re.compile(
    r"\bintroduc(?:e|ed|ing)\b[^.]{0,60}\b(?:mistakes?|errors?|perturbations?|modifications?|bugs?)\b|"
    r"present(?:ed)? (?:these |the )?modified|"
    r"we (?:created|constructed|generated|curated|built|designed|assembled) "
    r"(?:the |a |an |our )?(?:dataset|benchmark|protocols?|test cases?|corpus|questions?|examples?)",
    re.IGNORECASE,
)

# Scope-framing: what the benchmark is explicitly NOT for -> out_of_scope_uses, not tasks.
# Kept tight (a bare "rather than" is a frequent false positive in genuine task text).
_SCOPE_FRAMING_RE = re.compile(
    r"\bnot (?:suitable|designed|intended|meant|appropriate) for\b|"
    r"\bnot an? (?:agent|agentic)\b|"
    r"benchmark for [^.]{0,40}\brather than\b|"
    r"\bfor (?:llms?|models?|language models?) rather than\b|"
    r"\binstead of\b[^.]{0,40}\b(?:agent|benchmark)",
    re.IGNORECASE,
)

# Metric / methodology description: licenses dropping a data_type tag carrying such text.
_METRIC_METHOD_RE = re.compile(
    r"\baccuracy\b|\bf1\b|\bpass@\s?\d|\bbleu\b|\brouge\b|\bprecision\b|\brecall\b|"
    r"\bmeasur(?:e|ed|es|ing) the (?:accuracy|performance|score)\b|"
    r"\bmetric\b|\bevaluat(?:e|ed|es|ing|ion)\b|\btest cases?\b|\bscored?\b",
    re.IGNORECASE,
)

# Any benchmark_details.data_type modality term (fs.DATA_TYPE_VOCAB). Its presence vetoes
# the metric-only drop, so a genuine modality quote that also mentions a metric survives.
_MODALITY_TERM_RE = re.compile(
    "|".join(r"\b" + re.escape(t) + r"\b" for t in sorted(fs.DATA_TYPE_VOCAB)),
    re.IGNORECASE,
)

_JUDGE_SECTION = fs.JUDGE_SETUP_BUCKET.split(".", 1)[0]   # "methodology"
_JUDGE_FIELD = fs.JUDGE_SETUP_BUCKET.split(".", 1)[1]     # "judge_setup"
# Where to look for execution-scoring evidence (the methodology scoring fields).
_EXEC_EVIDENCE_FIELDS = frozenset({
    "methodology.methods", "methodology.metrics", "methodology.calculation",
})


def _ns(field: str):
    """Canonical Not-specified sentinel for a short field name (mirrors validator._canonical_ns)."""
    return ["Not specified"] if field in fs.LIST_FIELDS else "Not specified"


def _verb_near_noun(text: str) -> bool:
    for m in _SCORE_VERB_RE.finditer(text):
        window = text[max(0, m.start() - _PROXIMITY): m.end() + _PROXIMITY]
        if _OUTPUT_NOUN_RE.search(window):
            return True
    return False


def describes_output_scoring(quote: str) -> bool:
    """True iff the quote describes an LLM scoring/grading MODEL OUTPUTS.

    A construction/annotation marker vetoes the whole quote (it is about building the
    dataset, not scoring outputs); a discriminative scalar scorer (reward/ranking model,
    classifier) is not a generative LLM judge and is vetoed unless the quote also names a
    generative judge; otherwise an unambiguous judge term or a scoring verb next to an
    output noun qualifies."""
    if not quote:
        return False
    if _CONSTRUCTION_VETO_RE.search(quote):
        return False
    if _DISCRIMINATIVE_SCORER_RE.search(quote) and not _GENERATIVE_JUDGE_RE.search(quote):
        return False
    if _JUDGE_TERM_RE.search(quote):
        return True
    return _verb_near_noun(quote)


def describes_exec_scoring(quote: str) -> bool:
    """True iff the quote describes execution-based / programmatic scoring (Pass@k, unit tests, ...)."""
    return bool(quote) and bool(_EXEC_SCORING_RE.search(quote))


def is_subset_construction(quote: str) -> bool:
    """True iff the quote describes how a subset/variant was constructed (not the data origin)."""
    return bool(quote) and bool(_SUBSET_CONSTRUCTION_RE.search(quote))


def is_genuine_provenance(quote: str) -> bool:
    """True iff the quote names where the original data came from / how it was collected."""
    return bool(quote) and bool(_PROVENANCE_RE.search(quote))


def is_construction_quote(quote: str) -> bool:
    """True iff the quote describes how the dataset/benchmark was built or its items modified.

    Such a quote belongs in data.annotation, not purpose.tasks. Reuses the judge-guard's
    construction veto plus an introduce/modify marker the veto does not cover (e.g. biolp's
    'we introduced ... a single mistake that would cause it to fail')."""
    if not quote:
        return False
    return bool(_CONSTRUCTION_VETO_RE.search(quote) or _CONSTRUCTION_INTRO_RE.search(quote))


def is_scope_framing(quote: str) -> bool:
    """True iff the quote frames what the benchmark is NOT for (out_of_scope_uses, not tasks)."""
    return bool(quote) and bool(_SCOPE_FRAMING_RE.search(quote))


def is_metric_only_data_type(quote: str) -> bool:
    """True iff a data_type-tagged quote is a metric/methodology description with no modality term.

    benchmark_details.data_type must be a modality (fs.DATA_TYPE_VOCAB); a metric/accuracy
    sentence mis-tagged here is dropped so Stage-B abstains instead of emitting a junk modality.
    A quote that does name a modality term survives even if it also mentions a metric."""
    if not quote:
        return False
    if _MODALITY_TERM_RE.search(quote):
        return False
    return bool(_METRIC_METHOD_RE.search(quote))


# Widened LLM-judge / equality-checker signal for the EEE+HF judge rescue. Shared by the EEE
# injection (eee_workflow), the J4 escape in _guard_judge_cluster below, and the gate's denial
# mirror -- one detector, no second reimpl. Additive: the in-pipeline guard above is unchanged.
# Matched per sentence with citation / answer-extractor / execution / human-evaluation /
# dataset-construction vetoes, so it names an LLM that SCORES MODEL OUTPUTS -- never firing on
# exact-match scoring, human evaluation, or a model used to build/filter the dataset.
_LLM_JUDGE_SIGNAL_RE = re.compile(
    r"model[\s-]+graded"
    r"|llm[\s-]*as[\s-]*(?:a[\s-]*)?judge"
    r"|\blm[\s-]*(?:as[\s-]*(?:an?[\s-]*)?)?judge\b|language[\s-]*model[\s-]*judge"
    r"|(?:graded|judged|scored)\s+by\s+(?:an?\s+)?(?:llm|model|language\s+model)"
    r"|equality[\s-]*checker"
    r"|(?:llm|model)[\s-]*based\s+(?:\w+\s+){0,2}(?:checker|grader|evaluator|judge)"
    r"|uses?\s+(?:an?\s+)?[A-Za-z0-9][\w.+-]*(?:[ -][\w.+-]+){0,5}\s+(?:model\s+)?"
    r"(?:to\s+(?:grade|assess|evaluate|judge|score)"
    r"|as\s+(?:the\s+|an?\s+)?(?:judge|evaluator|grader|equality[\s-]*checker))"
    r"|(?:used|uses|acted?|acts|acting|serv(?:e|es|ed|ing))\s+as\s+(?:the\s+|an?\s+)?"
    r"(?:\w+\s+){0,3}(?:judge|evaluator|grader|equality[\s-]*checker)"
    r"|(?:judge|evaluator|grader|equality[\s-]*checker)\s+model\b"
    # win rate is only a judge signal alongside a comparison-judge context: a bare game
    # win rate (e.g. wordle) is deterministic, not an LLM judge.
    r"|(?:pairwise|preference|head[\s-]?to[\s-]?head|judged|adjudicat|preferred|\belo\b)"
    r"[^.]{0,40}win[\s-]*rate"
    r"|win[\s-]*rate[^.]{0,40}"
    r"(?:pairwise|preference|head[\s-]?to[\s-]?head|judg|adjudicat|preferred|\belo\b)",
    re.IGNORECASE,
)

# A judge sentence is disqualified when it is a bibliographic citation, an answer-EXTRACTOR
# mention (parsing the model's own answer, not scoring it), execution-based scoring, HUMAN
# evaluation, or a model used to BUILD/filter the dataset (construction, not output scoring).
_JUDGE_CITATION_RE = re.compile(r"\bet\s+al\b|\(\d{4}\)|\[\d+\]|arxiv", re.IGNORECASE)
_ANSWER_EXTRACTOR_RE = re.compile(
    r"(?:extract|pars\w+)[^.]*\b(?:answer|choice|option|label|response)s?\b", re.IGNORECASE)
# Human evaluation -- a PERSON doing the scoring (not an LLM judge). Requires 'human' to be the
# evaluating agent so it does not fire on a model applying a HUMAN-WRITTEN rubric.
_HUMAN_EVAL_VETO_RE = re.compile(
    r"humans?[\s-]*(?:evaluat|annotat|judg|rat|review|grad|assess|scor|label|coder|expert|"
    r"panel|jury|subject)"
    r"|(?:evaluat\w*|annotat\w*|rated|reviewed|judged|graded|assessed|scored)\s+by\s+"
    r"(?:\w+\s+){0,3}humans?\b"
    r"|annotators?\b|crowd[\s-]?work|mechanical\s+turk|\bturkers?\b|\bpanel\b|\braters?\b|"
    r"\bparticipants?\b|\bvolunteers?\b",
    re.IGNORECASE)
# Dataset construction / curation markers. Vetoed ONLY when the sentence names no model OUTPUT:
# an LLM scoring RESPONSES is output scoring even if it mentions difficulty/filtering/inclusion.
_JUDGE_CONSTRUCTION_VETO_RE = re.compile(
    r"difficult|for\s+inclusion|candidate\s+(?:item|example|question|answer)|\bfilter|\bcurat|"
    r"dataset\s+construction|during\s+(?:the\s+)?(?:\w+\s+)?construction|answerab|quality\s+control|"
    r"seed\s+examples?|test\s+case|\bgenerat\w*\s+(?:the\s+|each\s+)?(?:question|item|example|dataset|task)",
    re.IGNORECASE)
_JUDGE_OUTPUT_NOUN_RE = re.compile(
    r"\b(?:response|answer|output|completion|generation|prediction)s?\b|model\s+output", re.IGNORECASE)
# Split on sentence/clause punctuation (incl. ';') followed by whitespace, so a vetoed sibling
# clause cannot suppress a judge clause; a version like "Llama-3.3-70B" is not torn at its dots.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|[\r\n]+")
_JUDGE_SENTENCE_MAX = 400  # bound per-sentence template backtracking (naming sentences are short)

# A model name: starts model-shaped (capital letter or digit), then 0-5 more dotted/hyphen tokens.
_MODEL_NAME = r"([A-Z0-9][\w.+-]*(?:[ -][A-Za-z0-9][\w.+-]*){0,5})"
_JUDGE_MODEL_TEMPLATES = (
    re.compile(_MODEL_NAME + r"\s+is\s+used\s+as\s+(?:the\s+|an?\s+)?(?:\w+\s+){0,3}"
               r"(?:equality\s+)?(?:checker|judge|evaluator|grader)", re.IGNORECASE),
    re.compile(r"\buses?\s+(?:an?\s+)?" + _MODEL_NAME +
               r"\s+(?:model\s+)?(?:as\s+(?:the\s+|an?\s+)?(?:judge|evaluator|checker|grader)"
               r"|to\s+(?:grade|assess|evaluate|judge|score))", re.IGNORECASE),
    re.compile(_MODEL_NAME + r"\s+(?:acts?|acted|serv(?:e|es|ed|ing))\s+as\s+(?:the\s+|an?\s+)?"
               r"(?:judge|evaluator)\b", re.IGNORECASE),
    re.compile(r"(?:judge|evaluator|equality\s+checker|grader)\s+(?:model\s+)?(?:is|:|=)\s+"
               + _MODEL_NAME, re.IGNORECASE),
)
# Cross-sentence judge-model naming: a model the authors say they "use" followed by a config
# boundary (with/at <config>, "model", punctuation). Used only as a fallback once the passage
# has established LLM-judge grading context, so it captures e.g. ace's
# "We use Gemini 2.5 Pro with Thinking = High ..." whose own sentence names no judge role.
_USE_MODEL_RE = re.compile(
    r"\bwe\s+use[sd]?\s+" + _MODEL_NAME + r"\s+(?:with|at|model|\(|,|\.|$)", re.IGNORECASE
)
# Capitalized sentence-initial common words that are not model names.
_MODEL_STOPWORDS = frozenset({
    "the", "we", "this", "a", "an", "our", "it", "they", "scoring", "responses", "answers",
    "each", "all", "both", "model", "models", "for", "here", "these", "those", "their",
})
# Descriptor words ("an LLM-based equality checker") and generic role nouns ("a panel",
# "crowdworkers"): their presence disqualifies a captured span as a model identifier. ('reward' /
# 'critic' are NOT here -- they occur inside real model names; they are trimmed as trailing roles.)
_MODEL_DESCRIPTOR_WORDS = frozenset({
    "based", "checker", "judge", "evaluator", "grader", "equality", "correctness", "automatic",
    "reference", "scoring", "llm", "language", "verifier", "rubric",
})
_NON_MODEL_NOUNS = frozenset({
    "panel", "crowdworkers", "crowdworker", "workers", "worker", "annotators", "annotator",
    "evaluators", "judges", "raters", "rater", "participants", "participant", "volunteers",
    "humans", "human", "experts", "expert", "students", "student", "someone", "anyone",
    "unclear", "unspecified", "undisclosed", "various", "none",
})
# LLM-family vendor/model tokens. A digitless capture is only accepted as a model identifier when
# it carries one of these (so 'Claude Opus' / 'Mistral Large' pass but 'Gold Standard' / 'Majority
# Voting' do not). General model families, NOT benchmark names; a digit-bearing name needs no entry.
_MODEL_VENDORS = frozenset({
    "gpt", "chatgpt", "openai", "o1", "o3", "claude", "anthropic", "gemini", "palm", "bard",
    "llama", "mistral", "mixtral", "qwen", "deepseek", "yi", "command", "cohere", "phi", "gemma",
    "nemotron", "olmo", "falcon", "vicuna", "wizardlm", "tulu", "starling", "prometheus", "skywork",
    "internlm", "baichuan", "glm", "grok", "jamba", "dbrx", "solar", "nous", "hermes", "zephyr",
    "openchat", "aya", "reka", "sonnet", "opus", "haiku",
})
# Leading words (incl. capitalized articles like sentence-initial "The") a greedy capture sweeps
# in before the model id, and trailing role-descriptor tokens after it ("... Reward model").
_MODEL_LEADING_DROP = frozenset({"the", "a", "an", "our", "this", "that", "these", "those",
                                 "model", "models", "using", "used"})
_MODEL_TRIM_TRAILING = frozenset({"model", "models", "checker", "judge", "evaluator", "grader",
                                  "system", "reward", "critic"})


def _judge_sentences(text: str):
    """Yield sentences describing an LLM that SCORES MODEL OUTPUTS: a judge signal that is not a
    citation, an answer-extractor, execution scoring, human evaluation, or a discriminative
    scalar scorer (reward/ranking model, classifier) -- and not dataset construction, which is
    vetoed only when the sentence names no model output."""
    if not isinstance(text, str) or not text:
        return
    for raw in _SENTENCE_SPLIT_RE.split(text):
        s = raw.strip()
        if not s or not _LLM_JUDGE_SIGNAL_RE.search(s):
            continue
        if (_JUDGE_CITATION_RE.search(s) or _ANSWER_EXTRACTOR_RE.search(s)
                or _EXEC_SCORING_RE.search(s) or _HUMAN_EVAL_VETO_RE.search(s)):
            continue
        if _DISCRIMINATIVE_SCORER_RE.search(s) and not _GENERATIVE_JUDGE_RE.search(s):
            continue  # scalar reward/ranking/classifier scorer, not a generative LLM judge
        if _JUDGE_CONSTRUCTION_VETO_RE.search(s) and not _JUDGE_OUTPUT_NOUN_RE.search(s):
            continue  # construction/curation of dataset items, not scoring of model outputs
        yield s


def _model_shaped(tok: str) -> bool:
    """A token that can be part of a model id: capitalized (Claude / GPT-4) or digit-bearing (340B)."""
    return bool(tok) and (tok[:1].isupper() or any(c.isdigit() for c in tok))


def _clean_model_span(span: str) -> str:
    """Trim a captured span to the model identifier: drop leading non-model words (a greedy capture
    sweeps in 'where' / 'fine-tuned' / a capitalized article), trailing role descriptors, and any
    trailing connector words a greedy capture swept in ('GPT-4 as a judge' -> 'GPT-4')."""
    toks = [t for t in span.split() if t]
    while toks and (toks[0].lower().strip(".,;:") in _MODEL_LEADING_DROP or not _model_shaped(toks[0])):
        toks.pop(0)
    while toks and (toks[-1].lower().strip(".,;:") in _MODEL_TRIM_TRAILING or not _model_shaped(toks[-1])):
        toks.pop()
    return " ".join(toks).strip(" .,;:")


def _looks_like_model(name: str) -> bool:
    """A cleaned span is a model identifier, not a descriptor or a generic role noun. It must lead
    with a capitalized / digit-bearing token, carry no descriptor or role word, and look
    model-shaped: contain a digit (GPT-4 / Qwen3) or be an all-capitalized vendor/tier name
    (Claude / Mistral Large / Claude Opus)."""
    n = (name or "").strip()
    if not n or len(n) > 60:
        return False
    toks = n.split()
    first = toks[0].lower().strip(".,;:")
    if first in _MODEL_STOPWORDS or first in _NON_MODEL_NOUNS:
        return False
    words = [w for w in re.split(r"[\s.+-]+", n.lower()) if w]
    if any(w in _MODEL_DESCRIPTOR_WORDS or w in _NON_MODEL_NOUNS for w in words):
        return False
    if not (toks[0][:1].isupper() or any(c.isdigit() for c in n)):
        return False
    # A digit-bearing name (GPT-4 / Qwen3) or a known model-vendor token (Claude Opus / Mistral
    # Large); a bare capitalized phrase with neither ('Gold Standard') is not a model.
    return any(c.isdigit() for c in n) or any(w in _MODEL_VENDORS for w in words)


def describes_llm_judge(text: str) -> bool:
    """True iff the text has >=1 sentence describing an LLM scoring model outputs (LLM-as-judge /
    equality-checker), after the citation / extractor / exec / human-eval / construction vetoes."""
    for _ in _judge_sentences(text):
        return True
    return False


def _has_grading_context(sentences) -> bool:
    """True iff some sentence establishes LLM-judge grading context (a judge signal, an
    unambiguous judge term, or a rubric being scored). Gates the cross-sentence model fallback."""
    for s in sentences:
        if _LLM_JUDGE_SIGNAL_RE.search(s) or _JUDGE_TERM_RE.search(s):
            return True
        if "rubric" in s.lower() and _SCORE_VERB_RE.search(s):
            return True
    return False


def extract_judge_model(text: str):
    """The named LLM-judge / equality-checker model in the text, or None.

    Scans only non-vetoed judge sentences with a few high-precision templates, cleans the captured
    span and validates it as a model identifier; returns None when no model is confidently
    capturable, so the caller degrades to boolean-only. Fallback: when no template matches but the
    passage HAS grading context, capture a model the authors say they "use" in a sibling sentence
    (e.g. the judge role and the model name split across two sentences, as in ace)."""
    for s in _judge_sentences(text):
        if len(s) > _JUDGE_SENTENCE_MAX:
            continue  # bound template backtracking on a pathological run-on line
        for tmpl in _JUDGE_MODEL_TEMPLATES:
            m = tmpl.search(s)
            if not m:
                continue
            cand = _clean_model_span(m.group(1))
            if _looks_like_model(cand):
                return cand

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()] if text else []
    if _has_grading_context(sentences):
        for s in sentences:
            if len(s) > _JUDGE_SENTENCE_MAX or re.search(r"under\s+test|tested\s+model", s, re.IGNORECASE):
                continue  # a model 'under test' is the subject, never the judge
            m = _USE_MODEL_RE.search(s)
            if m:
                cand = _clean_model_span(m.group(1))
                if _looks_like_model(cand):
                    return cand
    return None


def collect_judge_scan_text(eee_metadata=None, hf_json=None) -> str:
    """Assemble the judge-scan text from EEE metric descriptions + the HF readme so the EEE
    injection and the gate mirror scan byte-identical input. Newline-joined to preserve the
    sentence boundaries the detectors split on."""
    texts: List[str] = []
    metrics = (eee_metadata or {}).get("metrics") if isinstance(eee_metadata, dict) else None
    if isinstance(metrics, dict):
        for cfg in metrics.values():
            if isinstance(cfg, dict):
                desc = cfg.get("evaluation_description")
                if isinstance(desc, str) and desc.strip():
                    texts.append(desc)
    readme = hf_json.get("readme_markdown") if isinstance(hf_json, dict) else None
    if isinstance(readme, str) and readme.strip():
        texts.append(readme)
    return "\n".join(texts)


def _is_real_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real_list(v) -> bool:
    return isinstance(v, list) and bool(v) and not is_not_specified(v)


def _cited_quotes(prov_section: Dict, field: str, id2quote: Dict[str, str]) -> List[str]:
    entry = prov_section.get(field) if isinstance(prov_section, dict) else None
    ids = entry.get("evidence_ids") if isinstance(entry, dict) else None
    if not isinstance(ids, list):
        return []
    return [id2quote[i] for i in ids if i in id2quote]


def _drop_prov(prov_section: Dict, field: str) -> None:
    if isinstance(prov_section, dict):
        prov_section.pop(field, None)


def apply_aboutness_guard(generated_sections: Dict, all_provenance: Dict,
                          evidence_items: List[Dict], all_flagged: Dict) -> Dict:
    """Correct off-topic judge_* and data.source fields in place. Returns telemetry.

    generated_sections: section -> field dict (provenance already split out).
    all_provenance: section -> {short_field -> {evidence_ids, source, evidence, ...}}.
    evidence_items: the flat verified-evidence list (each has evidence_id, field, quote).
    all_flagged: dotted path -> reason; demotions are recorded here (never with the
    'schema_invalid' marker, which feeds the gate's parse-failure metric).
    """
    id2quote = {it["evidence_id"]: it.get("quote", "") for it in evidence_items or []}
    digest_by_field: Dict[str, List[str]] = {}
    for it in evidence_items or []:
        digest_by_field.setdefault(it["field"], []).append(it.get("quote", ""))

    telem = {"judge_demoted": False, "judge_branch": None,
             "data_source_demoted": False, "demoted_fields": []}

    _guard_judge_cluster(generated_sections, all_provenance, evidence_items,
                         id2quote, digest_by_field, all_flagged, telem)
    _guard_data_source(generated_sections, all_provenance, id2quote,
                       digest_by_field, all_flagged, telem)
    return telem


def _guard_judge_cluster(generated_sections, all_provenance, evidence_items,
                         id2quote, digest_by_field, all_flagged, telem) -> None:
    meth = generated_sections.get(_JUDGE_SECTION)
    if not isinstance(meth, dict):
        return

    asserts = (
        meth.get("judge_uses_llm") is True
        or _is_real_list(meth.get("judge_models"))
        or _is_real_int(meth.get("judge_num"))
    )
    if not asserts:
        return

    meth_prov = all_provenance.get(_JUDGE_SECTION, {})
    candidates: List[str] = list(digest_by_field.get(fs.JUDGE_SETUP_BUCKET, []))
    for f in fs.JUDGE_FLAT_FIELDS:
        candidates.extend(_cited_quotes(meth_prov, f, id2quote))

    if any(describes_output_scoring(q) for q in candidates):
        return  # genuine output-scoring evidence -> leave the cluster intact

    # No output-scoring evidence backs the asserted cluster: demote.
    original = meth.get("judge_uses_llm")
    exec_eid, exec_quote = _find_exec_evidence(evidence_items)

    # J4: a named LLM-judge / equality-checker anywhere in the verified evidence keeps the cluster
    # intact even when that quote was routed off the judge bucket (it may be cited on another
    # field) -- BUT only when there is no execution-scoring evidence. An exec-scored cluster (Pass@1)
    # must still demote even if a separate quote (e.g. a related-work mention) names a judge model.
    if exec_eid is None and any(extract_judge_model(q) for q in id2quote.values()):
        return

    if exec_eid is not None:
        meth["judge_uses_llm"] = False  # derived: scoring is programmatic, not an LLM judge
        all_provenance.setdefault(_JUDGE_SECTION, {})["judge_uses_llm"] = {
            "source": "derived",
            "evidence": exec_quote,
            "evidence_ids": [exec_eid],
            "status": "derived",
            "verified": True,
        }
        telem["judge_branch"] = "a"
        reason = (f"[aboutness] judge cluster demoted to False: no output-scoring evidence; "
                  f"scoring is programmatic or a discriminative scorer (was {original!r})")
    else:
        meth["judge_uses_llm"] = _ns("judge_uses_llm")
        _drop_prov(all_provenance.get(_JUDGE_SECTION, {}), "judge_uses_llm")
        telem["judge_branch"] = "b"
        reason = (f"[aboutness] judge cluster demoted to Not specified: no output-scoring "
                  f"evidence (was {original!r})")

    meth["judge_num"] = _ns("judge_num")
    meth["judge_models"] = _ns("judge_models")
    meth["judge_score_consolidation"] = _ns("judge_score_consolidation")
    for f in ("judge_num", "judge_models", "judge_score_consolidation"):
        _drop_prov(all_provenance.get(_JUDGE_SECTION, {}), f)

    all_flagged[f"{_JUDGE_SECTION}.judge_uses_llm"] = reason
    for f in ("judge_num", "judge_models", "judge_score_consolidation"):
        all_flagged[f"{_JUDGE_SECTION}.{f}"] = "[aboutness] judge cluster demoted (see judge_uses_llm)"
    telem["judge_demoted"] = True
    telem["demoted_fields"].extend(
        f"{_JUDGE_SECTION}.{f}" for f in fs.JUDGE_FLAT_FIELDS)


def _describes_nonllm_scoring(quote: str) -> bool:
    """True iff the quote is positive evidence that scoring is NOT an LLM judge: execution-based
    (Pass@k, unit tests) or a discriminative scalar scorer (reward/ranking model, classifier).
    Both license the derived judge_uses_llm=False claim."""
    if not quote:
        return False
    if describes_exec_scoring(quote):
        return True
    return bool(_DISCRIMINATIVE_SCORER_RE.search(quote)) and not bool(_GENERATIVE_JUDGE_RE.search(quote))


def _find_exec_evidence(evidence_items):
    for it in evidence_items or []:
        if it.get("field") in _EXEC_EVIDENCE_FIELDS and _describes_nonllm_scoring(it.get("quote", "")):
            return it["evidence_id"], it.get("quote", "")
    return None, None


def _guard_data_source(generated_sections, all_provenance, id2quote,
                       digest_by_field, all_flagged, telem) -> None:
    data = generated_sections.get("data")
    if not isinstance(data, dict):
        return
    src = data.get("source")
    if is_not_specified(src) or not isinstance(src, str):
        return

    data_prov = all_provenance.get("data", {})
    cited = _cited_quotes(data_prov, "source", id2quote)
    if not cited:
        cited = list(digest_by_field.get("data.source", []))
    if not cited:
        return

    if all(is_subset_construction(q) for q in cited) and not any(is_genuine_provenance(q) for q in cited):
        data["source"] = "Not specified"
        # Mandatory: drop provenance so backfill_from_provenance (prose_only) cannot
        # resurrect the subset quote as the field value.
        _drop_prov(all_provenance.get("data", {}), "source")
        all_flagged["data.source"] = "[aboutness] data.source demoted (subset/variant construction, not provenance)"
        telem["data_source_demoted"] = True
        telem["demoted_fields"].append("data.source")
