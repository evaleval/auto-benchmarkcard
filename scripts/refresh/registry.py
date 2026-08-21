"""The registration interface Phase A/B/C plug their targeted fields into.

A phase never edits the merge or the harness; it appends FieldSpec objects via
register_phase_a/b/c(). The harness then runs reconstruct -> invoke -> merge and,
for each registered field, calls that field's recompute/grounding/decide hooks.

The improve-only FLOOR is enforced by the harness in merge.py, independent of any
decide(): a `new` that is blank / Not-specified / ungrounded can never replace an
existing value. A decide() may be stricter, never looser.

Contract for the four hooks on a FieldSpec:

  path        Dotted path into the INNER benchmark_card, e.g. "methodology.interpretation"
              or "data.size". Top-level card keys are never added or reordered.
  phase       "A" | "B" | "C" (for the changelog and roll-up grouping).
  recompute   Optional (refreshed_card, kwargs, ctx) -> new_value. When None, the
              harness reads `new` straight from the refreshed card at `path`. Use a
              recompute when the value is derived deterministically (Phase A) rather
              than taken from the re-composed prose.
  grounding   (new, refreshed_card, ctx) -> evidence-or-None. Return a truthy evidence
              handle (a span, a source id, a short string) when `new` traces to cached
              evidence, else None. None forces the floor to keep `old` (ungrounded).
  decide      (old, new, evidence, ctx) -> bool. True means replace. This is where a
              phase encodes "strictly better" for its field. Only consulted after the
              floor allows the replacement.
  eligible    (ctx) -> bool. False makes the phase abstain for this card (e.g. Phase B
              abstains when ctx["b_eligible"] is False). Defaults to always-eligible.
  is_fake     Optional (old, ctx) -> bool. DEMOTE affordance. When set, this is a
              "demote spec": if is_fake(old, ctx) is True the harness ALLOWS replacing
              even when the new value is blank / Not-specified (it bypasses ONLY the
              is_blank(new) clause of the floor, only for this spec, only when is_fake
              fires). Use it to lower provably-fake content (e.g. a synthesized
              other:<slug> metric, or the non-canonical "No evidence provided") to an
              honest "Not specified" via recompute. is_fake MUST be deterministic (no
              LLM); evidence is still required, so grounding must return the basis. When
              is_fake is None (or returns False) the field is never demoted: genuine
              content is still protected by the full floor. Default None.

ctx is a dict the harness passes through: {slug, run_dir, b_eligible, eee_metadata,
kwargs, live_card, refreshed_card}.

Illustrative example (DO NOT register; A/B/C own the real predicates):

    FieldSpec(
        path="methodology.interpretation",
        phase="B",
        recompute=None,                      # take the re-composed value
        grounding=lambda new, refreshed, ctx: "eee" if new in _ENUM else None,
        decide=lambda old, new, ev, ctx: new in _ENUM and old not in _ENUM,
        eligible=lambda ctx: ctx["b_eligible"],
    )

Illustrative demote spec (DO NOT register; Phase A owns the real predicate):

    FieldSpec(
        path="methodology.metrics",
        phase="A",
        recompute=lambda refreshed, kwargs, ctx: ["Not specified"],   # honest abstention
        grounding=lambda new, refreshed, ctx: "deterministic-demote",  # the audited basis
        decide=lambda old, new, ev, ctx: True,
        is_fake=lambda old, ctx: _is_synthesized_other_metric(old),    # deterministic
    )
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# card_utils is already imported by merge.py (the null-refresh path needs it). A's license/format
# helpers are imported here; the eee_workflow / composer symbols are imported LAZILY inside the
# Phase-B predicates so an empty (null-refresh) registry never pulls the LLM stack.
from auto_benchmarkcard.card_utils import (
    _AMBIGUOUS_LICENSE_TAGS,
    _LICENSE_MAP,
    _label_hosting_format,
    STORAGE_FORMAT_TOKENS,
    extract_hf_tags,
    is_not_specified,
)

_LICENSE_VALUES = frozenset(_LICENSE_MAP.values())


@dataclass(frozen=True)
class FieldSpec:
    path: str
    phase: str
    grounding: Callable[[Any, Any, Dict[str, Any]], Any]
    decide: Callable[[Any, Any, Any, Dict[str, Any]], bool]
    recompute: Optional[Callable[[Any, Dict[str, Any], Dict[str, Any]], Any]] = None
    eligible: Callable[[Dict[str, Any]], bool] = field(default=lambda ctx: True)
    is_fake: Optional[Callable[[Any, Dict[str, Any]], bool]] = None


class FieldRegistry:
    """Ordered collection of FieldSpecs. Duplicate paths are rejected so two phases
    cannot silently both own the same field (the ownership matrix forbids it)."""

    def __init__(self) -> None:
        self._specs: List[FieldSpec] = []

    def register(self, spec: FieldSpec) -> None:
        if any(s.path == spec.path for s in self._specs):
            raise ValueError(f"field already registered: {spec.path}")
        self._specs.append(spec)

    def specs(self) -> List[FieldSpec]:
        return list(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


# Phase A: deterministic demotions (size / format / metrics / license). Every recompute reads
# the LIVE published card (ctx["live_card"]) and the reconstructed kwargs, never an LLM, so the
# refresh is fully reproducible offline. Each predicate is improve-only: it returns None (a no-op,
# the floor keeps old) unless the value is provably a bare storage token, an other:<slug> proxy,
# a junk license tag, or a coarse bucket the breakdown refines. The composer's _size_bucket_range /
# split-key helpers are imported lazily so a null refresh (no phases) never pulls the LLM stack.


def _inner_card(card: Any) -> Dict[str, Any]:
    return card.get("benchmark_card", card) if isinstance(card, dict) else {}


def _live(ctx: Dict[str, Any], section: str, field_name: str) -> Any:
    sec = _inner_card(ctx.get("live_card")).get(section)
    return sec.get(field_name) if isinstance(sec, dict) else None


# data.format: a bare storage token describes hosting, not the instance structure.
def _format_recompute(refreshed: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    old = _live(ctx, "data", "format")
    if isinstance(old, str) and old.strip().lower() in STORAGE_FORMAT_TOKENS:
        return _label_hosting_format(old.strip())
    return None


def _format_grounding(new: Any, refreshed: Any, ctx: Dict[str, Any]) -> Any:
    old = _live(ctx, "data", "format")
    if isinstance(old, str) and old.strip().lower() in STORAGE_FORMAT_TOKENS:
        return old.strip().lower()
    return None


def _format_decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    return (isinstance(old, str) and isinstance(new, str)
            and old != new and old.strip().lower() in STORAGE_FORMAT_TOKENS)


# methodology.metrics: clean a descriptive other:<slug>; demote a pure <benchmark>.score proxy.
_SCORE_PROXY_RE = re.compile(r"^other:.+\.score$", re.I)


def _is_score_proxy(metric: Any) -> bool:
    return isinstance(metric, str) and bool(_SCORE_PROXY_RE.match(metric.strip()))


def _metrics_all_proxy(metrics: Any) -> bool:
    return (isinstance(metrics, list) and len(metrics) > 0
            and all(_is_score_proxy(m) for m in metrics))


def _readable_other_metric(metric: str) -> str:
    body = metric[len("other:"):] if metric.startswith("other:") else metric
    body = re.sub(r"_on_.+_task$", "", body)
    return body.replace("_", " ").strip()


def _clean_metrics(metrics: List[Any]) -> List[Any]:
    out: List[Any] = []
    for m in metrics:
        if isinstance(m, str) and m.startswith("other:") and not _is_score_proxy(m):
            out.append(_readable_other_metric(m))
        else:
            out.append(m)  # canonical names and .score proxies are left untouched
    return out


def _metrics_recompute(refreshed: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    old = _live(ctx, "methodology", "metrics")
    if not isinstance(old, list) or not old:
        return None
    if _metrics_all_proxy(old):
        return ["Not specified"]  # demote: the is_fake bypass authorizes this blank value
    cleaned = _clean_metrics(old)
    return cleaned if cleaned != old else None


def _metrics_grounding(new: Any, refreshed: Any, ctx: Dict[str, Any]) -> Any:
    if _metrics_all_proxy(_live(ctx, "methodology", "metrics")):
        return "deterministic-demote:other-score-proxy"
    return "readable-metric-clean" if isinstance(new, list) and new else None


def _metrics_is_fake(old: Any, ctx: Dict[str, Any]) -> bool:
    return _metrics_all_proxy(old)


def _metrics_decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    if _metrics_all_proxy(old):
        return new == ["Not specified"]
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != len(new) or new == old:
        return False
    gained = False
    for o, n in zip(old, new):
        if isinstance(o, str) and o.startswith("other:") and not _is_score_proxy(o):
            if n != o:
                gained = True
        elif n != o:
            return False  # a canonical name or .score proxy was altered -> reject
    return gained


# ethical...data_licensing: map a junk tag or strip an other: escape; never touch a prose license.
def _is_junk_license(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    s = value.strip()
    if not s or is_not_specified(s):
        return True
    low = s.lower()
    if low.startswith("other:") or low in _AMBIGUOUS_LICENSE_TAGS:
        return True
    # a bare single-token tag (no spaces) that is not already a clean mapped license name
    return " " not in s and s not in _LICENSE_VALUES


def _license_recompute(refreshed: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    old = _live(ctx, "ethical_and_legal_considerations", "data_licensing")
    if not isinstance(old, str):
        return None
    s = old.strip()
    if s.lower().startswith("other:"):
        return s[len("other:"):].strip() or None
    hf = (kwargs or {}).get("hf_metadata")
    mapped = (extract_hf_tags(hf).get("ethical_and_legal_considerations.data_licensing")
              if hf else None)
    if isinstance(mapped, str) and mapped in _LICENSE_VALUES:
        return mapped
    if " " not in s and s.lower() in _LICENSE_MAP:
        return _LICENSE_MAP[s.lower()]
    return None


def _license_grounding(new: Any, refreshed: Any, ctx: Dict[str, Any]) -> Any:
    if not isinstance(new, str) or not new.strip():
        return None
    old = _live(ctx, "ethical_and_legal_considerations", "data_licensing")
    if isinstance(old, str) and old.strip().lower().startswith("other:"):
        return "prefix-strip"
    return "hf-license"


def _license_decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    return isinstance(new, str) and bool(new.strip()) and new != old and _is_junk_license(old)


# data.size: a coarse bucket yields to a precise example count summed from a consistent breakdown.
def _size_from_breakdown(size: Any, breakdown: Any) -> Optional[str]:
    from auto_benchmarkcard.tools.composer.composer_tool import (
        _is_heterogeneous_stat_key,
        _is_split_name,
        _size_bucket_range,
    )
    rng = _size_bucket_range(size)
    if rng is None or not isinstance(breakdown, dict) or not breakdown:
        return None
    vals = [v for v in breakdown.values() if isinstance(v, int) and not isinstance(v, bool)]
    if len(vals) != len(breakdown) or not vals:
        return None
    if any(_is_heterogeneous_stat_key(k) for k in breakdown):
        return None  # '# Tasks' / 'domains' / 'libraries' are statistics, not an example partition
    if all(v == 1 for v in vals):
        return None  # column-key placeholders (listings:1, qa1:1), not real counts
    splits = [_is_split_name(k) for k in breakdown]
    if any(splits) and not all(splits):
        return None  # a split/non-split mix is not a clean partition
    total = sum(vals)
    low, high = rng
    return f"{total} examples" if low < total <= high else None


def _size_is_bucket(value: Any) -> bool:
    from auto_benchmarkcard.tools.composer.composer_tool import _size_bucket_range
    return isinstance(value, str) and _size_bucket_range(value) is not None


def _size_recompute(refreshed: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    data = _inner_card(ctx.get("live_card")).get("data", {})
    if not isinstance(data, dict):
        return None
    return _size_from_breakdown(data.get("size"), data.get("size_breakdown"))


def _size_grounding(new: Any, refreshed: Any, ctx: Dict[str, Any]) -> Any:
    return "breakdown-sum" if isinstance(new, str) and new else None


def _size_decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    if not isinstance(new, str) or not any(ch.isdigit() for ch in new):
        return False
    return is_not_specified(old) or _size_is_bucket(old)


def register_phase_a(registry: FieldRegistry) -> None:
    """Phase A registers its deterministic-demotion fields here."""
    registry.register(FieldSpec(
        path="data.format", phase="A",
        recompute=_format_recompute, grounding=_format_grounding, decide=_format_decide,
    ))
    registry.register(FieldSpec(
        path="methodology.metrics", phase="A",
        recompute=_metrics_recompute, grounding=_metrics_grounding,
        decide=_metrics_decide, is_fake=_metrics_is_fake,
    ))
    registry.register(FieldSpec(
        path="ethical_and_legal_considerations.data_licensing", phase="A",
        recompute=_license_recompute, grounding=_license_grounding, decide=_license_decide,
    ))
    registry.register(FieldSpec(
        path="data.size", phase="A",
        recompute=_size_recompute, grounding=_size_grounding, decide=_size_decide,
    ))


# Phase B helpers. All grounding reads the re-composed card's IN-MEMORY provenance
# (refreshed_card["provenance"][sec][field]); the eee-injected interpretation provenance and the
# deterministic human_baseline relocation are sidecar-only and invisible here (the grounding
# gotcha), so those two ground on ctx.eee_metadata + the card's own fields instead.

_NO_EVIDENCE_RE = re.compile(r"^\s*no evidence provided\b", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_DIGIT_RE = re.compile(r"\d")
_ENUM_INTERP = {"higher_is_better", "lower_is_better"}


def _b_eligible(ctx: Dict[str, Any]) -> bool:
    return bool(ctx.get("b_eligible"))


def _inner(card: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(card, dict):
        return None
    return card.get("benchmark_card", card)


def _get(card: Any, path: str) -> Any:
    node = _inner(card)
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _is_blank(value: Any) -> bool:
    """Matches merge.is_blank (kept local to avoid a registry<->merge import cycle)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or is_not_specified(value)
    if isinstance(value, (list, dict)):
        return len(value) == 0 or is_not_specified(value)
    return False


def _is_no_evidence(value: Any) -> bool:
    """True when the value is the non-canonical 'No evidence provided...' abstention (any
    suffixed variant), for either a string field or a single-item list field."""
    vals = value if isinstance(value, list) else [value]
    return any(isinstance(v, str) and _NO_EVIDENCE_RE.match(v) for v in vals)


def _prov_entry(refreshed_card: Any, path: str) -> Optional[Dict[str, Any]]:
    if not isinstance(refreshed_card, dict):
        return None
    prov = refreshed_card.get("provenance")
    if not isinstance(prov, dict):
        return None
    sec, _, field_name = path.partition(".")
    section = prov.get(sec)
    entry = section.get(field_name) if isinstance(section, dict) else None
    return entry if isinstance(entry, dict) else None


def _evidence_ids(refreshed_card: Any, path: str) -> Optional[Dict[str, Any]]:
    """The provenance entry when the field traces to a cached source span (non-empty
    evidence_ids), else None. This is the anti-fabrication gate for paper-extraction fills."""
    entry = _prov_entry(refreshed_card, path)
    if entry and entry.get("evidence_ids"):
        return entry
    return None


def _ground_evidence(path: str) -> Callable[[Any, Any, Dict[str, Any]], Any]:
    def grounding(new: Any, refreshed_card: Any, ctx: Dict[str, Any]) -> Any:
        return _evidence_ids(refreshed_card, path)
    return grounding


def _decide_fill_if_ns(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    """Floor already guaranteed `new` is non-blank and grounded; adopt only to fill a gap."""
    return _is_blank(old)


# interpretation: two-sided. The SAME compose_interp_prose the eee_workflow injection uses is
# re-run here over the card's own grounded baseline/human + the eee direction. It prefers the
# re-composed card's fields, and falls back to the LIVE card's own baseline/human when compose is
# skipped, so the interpretation prose is computable OFFLINE for every enum card: it only restates
# the card's own grounded fields (its eee direction + reported baselines), never a new extraction.

def _recompute_interpretation(refreshed_card: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    from auto_benchmarkcard.eee_workflow import (
        _derive_interpretation, _primary_metric_label, compose_interp_prose)
    meth = _get(refreshed_card, "methodology") or _get(ctx.get("live_card"), "methodology") or {}
    eee = ctx.get("eee_metadata") or {}
    return compose_interp_prose(
        _derive_interpretation(eee),
        meth.get("baseline_results"),
        meth.get("human_baseline"),
        _primary_metric_label(eee),
    )


def _ground_interpretation(new: Any, refreshed_card: Any, ctx: Dict[str, Any]) -> Any:
    # compose_interp_prose returns prose ONLY when the eee direction is real AND at least one of
    # the card's own baseline_results/human_baseline is non-NS (both grounded: Stage-B paper or
    # eee-derived). So a non-empty recompute IS the grounding basis.
    return "interp:eee-direction+own-baseline" if new else None


def _decide_interpretation(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    old_ok = _is_blank(old) or (isinstance(old, str) and old.strip() in _ENUM_INTERP)
    return bool(old_ok and new and _DIGIT_RE.search(new))


# human_baseline: paper-grounded (evidence_ids) OR a verbatim cross-field lift from the card's own
# baseline_results/overview (the relocation provenance is sidecar-only). The verbatim lift is the
# exempt cross-field backfill -- no evidence_ids required.

def _ground_human_baseline(new: Any, refreshed_card: Any, ctx: Dict[str, Any]) -> Any:
    ev = _evidence_ids(refreshed_card, "methodology.human_baseline")
    if ev:
        return ev
    if isinstance(new, str) and new.strip():
        needle = new.strip()
        for src_path in ("methodology.baseline_results", "benchmark_details.overview"):
            src = _get(refreshed_card, src_path)
            if isinstance(src, str) and needle in src:
                return f"verbatim:{src_path}"
    return None


def _decide_collection_date(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
    return _is_blank(old) and bool(_YEAR_RE.search(str(new)))


# Ethics conditional fill. The signal must be SPECIFIC to the benchmark's own domain/source, not a
# stray generic mention; it gates ADOPTION only (retrieval runs for every card). gap_query still
# requires a grounded span, so a math benchmark with no ethics text stays NS regardless.

# "diagnos*" is deliberately excluded: "diagnose model failures" / "diagnostic benchmark" are
# common ML phrasings, not medical (they over-fire, e.g. GSM8K's "diagnose the failures of current
# models"). Real medical cards fire via patient / clinical / radiology / disease / hospital etc.
_MED_RE = re.compile(
    r"\b(?:medical|clinical|patient|healthcare|health\s+care|biomedical|radiolog\w*|radiograph\w*|"
    r"disease|hospital|physician|clinician|nurse|psychiatr\w*|mental\s+health|genomic\w*|"
    r"electronic\s+health\s+record|electronic\s+medical\s+record|clinical\s+note|chest\s+x-?ray|"
    r"\bmri\b|\bct\s+scan|patholog\w*|endoscop\w*|\bicu\b)\b", re.I)
_HUMANSUBJ_RE = re.compile(
    r"\b(?:human\s+subject\w*|study\s+participant\w*|volunteer\w*|biometric\w*|fingerprint\w*|"
    r"facial\s+(?:image|recognition)|voice\s+recording\w*|speech\s+recording\w*|"
    r"images\s+of\s+(?:people|persons|faces)|demographic\w*)\b", re.I)
_SCRAPE_RE = re.compile(
    r"\b(?:scrap(?:ed|ing)|web[\s-]?crawl\w*|crawled|reddit|twitter|social\s+media|online\s+forum\w*|"
    r"user[-\s]generated|comments?\s+from|posts?\s+from|web\s+pages?)\b", re.I)
_USERDATA_RE = re.compile(
    r"\b(?:personal\s+data|personally\s+identifiable|\bpii\b|user\s+data|conversation\s+logs?|"
    r"chat\s+logs?|emails?|search\s+quer\w*|browsing\s+histor\w*|location\s+data|private\s+messages?)\b",
    re.I)
_NOPII_RE = re.compile(
    r"\b(?:synthetic|no\s+personal\s+(?:data|information)|no\s+personally\s+identifiable|no\s+pii|"
    r"contains?\s+no\s+personal|de-?identif\w+|anonymiz\w+)\b", re.I)


def _ethics_signal(refreshed_card: Any) -> set:
    """Ethics fields the benchmark's OWN domain/source/overview signal as relevant."""
    inner = _inner(refreshed_card) or {}
    bd = inner.get("benchmark_details") or {}
    pu = inner.get("purpose_and_intended_users") or {}
    data = inner.get("data") or {}

    def txt(v: Any) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v) if v else ""

    blob = " ".join([
        txt(bd.get("domains")), txt(bd.get("overview")), txt(pu.get("goal")),
        txt(data.get("source")), txt(data.get("annotation")),
    ])
    allow: set = set()
    if _MED_RE.search(blob) or _HUMANSUBJ_RE.search(blob):
        allow |= {"privacy", "consent", "compliance"}
    if _SCRAPE_RE.search(blob):
        allow |= {"privacy", "compliance"}
    if _USERDATA_RE.search(blob):
        allow |= {"privacy", "consent"}
    if _NOPII_RE.search(blob):
        allow.add("privacy")  # a grounded synthetic / no-PII privacy statement is adoptable
    return allow


def _decide_ethics(field_key: str) -> Callable[..., bool]:
    def decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
        return _is_blank(old) and field_key in _ethics_signal(ctx.get("refreshed_card"))
    return decide


def _make_demotable(
    path: str,
    list_field: bool,
    extra_fill_ok: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> FieldSpec:
    """A spec that fills `path` from a grounded paper span when the old value is blank (or the
    fake 'No evidence provided'), and otherwise demotes that fake phrase to canonical Not specified
    via is_fake. extra_fill_ok(ctx)->bool gates only the FILL path (e.g. the ethics signal)."""
    ns_value: Any = ["Not specified"] if list_field else "Not specified"

    def recompute(refreshed_card: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
        new = _get(refreshed_card, path)
        fill_ok = extra_fill_ok is None or extra_fill_ok(ctx)
        if not _is_blank(new) and fill_ok:
            return new          # an adoptable grounded fill (signal gate already applied)
        return ns_value         # nothing adoptable -> demote target (if is_fake fires below)

    def grounding(new: Any, refreshed_card: Any, ctx: Dict[str, Any]) -> Any:
        if not _is_blank(new):
            return _evidence_ids(refreshed_card, path)   # fill: require a cached span
        return "demote:no-evidence-phrase"               # demote: deterministic basis

    def decide(old: Any, new: Any, evidence: Any, ctx: Dict[str, Any]) -> bool:
        if _is_blank(new):
            return _is_no_evidence(old)                  # demote only the fake phrase
        return _is_blank(old) or _is_no_evidence(old)    # fill a gap or overwrite the fake

    def is_fake(old: Any, ctx: Dict[str, Any]) -> bool:
        return _is_no_evidence(old)

    return FieldSpec(
        path=path, phase="B", recompute=recompute, grounding=grounding,
        decide=decide, eligible=_b_eligible, is_fake=is_fake,
    )


def _ethics_fill_ok(field_key: str) -> Callable[[Dict[str, Any]], bool]:
    def ok(ctx: Dict[str, Any]) -> bool:
        return field_key in _ethics_signal(ctx.get("refreshed_card"))
    return ok


def register_phase_b(registry: FieldRegistry) -> None:
    """Phase B paper-gated recall fields. Every spec abstains on no-docling cards (eligible =
    ctx['b_eligible']) so shells stay byte-identical. Paper-extraction fills require a grounded
    span (non-empty evidence_ids); interpretation recompute and the human_baseline cross-field
    lift are the two exempt cases (they restate the card's own grounded fields)."""

    # interpretation: enum/NS -> grounded prose (two-sided; recompute mirrors the eee injection).
    # NOT gated on b_eligible: it only restates the card's OWN baseline/human + eee direction, so it
    # is a safe offline upgrade for every enum card (including no-paper shells). The decide gate still
    # limits it to cards whose old value is the enum/NS and whose restatement carries a number.
    registry.register(FieldSpec(
        path="methodology.interpretation", phase="B",
        recompute=_recompute_interpretation,
        grounding=_ground_interpretation,
        decide=_decide_interpretation,
        eligible=lambda ctx: True,
    ))

    # Straight paper-extraction fills: adopt a grounded, non-blank value only over an NS gap.
    for path in ("methodology.calculation", "methodology.validation",
                 "data.contamination_controls", "benchmark_details.similar_benchmarks"):
        registry.register(FieldSpec(
            path=path, phase="B",
            grounding=_ground_evidence(path),
            decide=_decide_fill_if_ns,
            eligible=_b_eligible,
        ))

    # human_baseline: paper span OR a verbatim cross-field lift from the card's own fields.
    registry.register(FieldSpec(
        path="methodology.human_baseline", phase="B",
        grounding=_ground_human_baseline,
        decide=_decide_fill_if_ns,
        eligible=_b_eligible,
    ))

    # collection_date is intentionally NOT recalled: its gap_query grabs the nearest date in the
    # paper, which misattributes when no collection date is stated (e.g. a pricing date read as the
    # collection date). It is low value and the field keeps its v2 value instead. A strict
    # deterministic pass could revisit it later. _decide_collection_date is kept for that option.

    # Ethics privacy/consent: conditional fill gated by the benchmark's own domain signal.
    registry.register(FieldSpec(
        path="ethical_and_legal_considerations.privacy_and_anonymity", phase="B",
        grounding=_ground_evidence("ethical_and_legal_considerations.privacy_and_anonymity"),
        decide=_decide_ethics("privacy"),
        eligible=_b_eligible,
    ))
    registry.register(FieldSpec(
        path="ethical_and_legal_considerations.consent_procedures", phase="B",
        grounding=_ground_evidence("ethical_and_legal_considerations.consent_procedures"),
        decide=_decide_ethics("consent"),
        eligible=_b_eligible,
    ))

    # compliance: conditional fill (signal-gated) + demote the fake 'No evidence provided'.
    registry.register(_make_demotable(
        "ethical_and_legal_considerations.compliance_with_regulations",
        list_field=False, extra_fill_ok=_ethics_fill_ok("compliance"),
    ))

    # out_of_scope_uses (list) + limitations (str): grounded fill + fake-phrase demote (item 7).
    registry.register(_make_demotable(
        "purpose_and_intended_users.out_of_scope_uses", list_field=True))
    registry.register(_make_demotable(
        "purpose_and_intended_users.limitations", list_field=False))


def register_phase_c(registry: FieldRegistry) -> None:
    """Phase C registers its possible_risks fields here. Empty until C builds."""


def build_registry(phases: str = "") -> FieldRegistry:
    """Build a registry with the requested phases' fields.

    phases is a subset of "abc"; "" yields the empty registry the null-refresh gate
    requires. The harness calls this; A/B/C only edit register_phase_*.
    """
    registry = FieldRegistry()
    if "a" in phases:
        register_phase_a(registry)
    if "b" in phases:
        register_phase_b(registry)
    if "c" in phases:
        register_phase_c(registry)
    return registry
