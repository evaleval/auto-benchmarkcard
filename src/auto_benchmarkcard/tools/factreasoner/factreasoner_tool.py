import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

from fact_reasoner.atom_extractor import AtomExtractor
from fact_reasoner.atom_reviser import AtomReviser
from fact_reasoner.context_retriever import ContextRetriever
from fact_reasoner.factreasoner import FactReasoner
from fact_reasoner.nli_extractor import NLIExtractor

logger = logging.getLogger(__name__)


def _create_atom_marginal_mappings(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Map atom IDs to atom data and variable names to marginal data."""
    atoms_by_id = {atom["id"]: atom for atom in formatted_rag_results.get("atoms", [])}
    marginals_by_var = {m.get("variable", ""): m for m in marginals}
    return atoms_by_id, marginals_by_var


def _find_marginal_for_atom(
    atom_id: str, atom_text: str, marginals_by_var: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Find the marginal probability data corresponding to an atom."""
    # FactReasoner variable names may contain the atom ID or a text prefix
    for var_name, marginal in marginals_by_var.items():
        if atom_id in var_name or atom_text[:50] in var_name:
            return marginal
    return marginals_by_var.get(atom_id)


def _determine_flag_reason(field_stats: Dict[str, Any], threshold: float) -> tuple[bool, str, str]:
    """Determine if a field should be flagged and why, based on probability stats."""
    avg_probability = field_stats.get("avg_probability", 1.0)
    all_neutral = field_stats.get("all_neutral", False)
    neutral_count = field_stats.get("neutral_count", 0)

    should_flag = avg_probability < threshold or all_neutral

    if not should_flag:
        return False, "", ""

    if all_neutral:
        reason = "all_atoms_neutral"
        reason_desc = f"All {neutral_count} atoms have neutral scores (no evidence found)"
    else:
        reason = "low_factuality_score"
        reason_desc = f"Average factuality score {avg_probability:.3f} below threshold {threshold}"

    return True, reason, reason_desc


def analyze_factuality_by_field(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate factuality scores per benchmark card field and compute accuracy metrics."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    field_stats = {}
    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            if field not in field_stats:
                field_stats[field] = {
                    "total_atoms": 0,
                    "high_confidence_correct": 0,  # p_true > 0.8
                    "likely_correct": 0,  # p_true > 0.6
                    "uncertain": 0,  # 0.4 <= p_true <= 0.6
                    "likely_incorrect": 0,  # p_true < 0.4
                    "high_confidence_incorrect": 0,  # p_true < 0.2
                    "avg_probability": 0,
                    "probabilities": [],
                    "atoms": [],
                }

            field_stats[field]["total_atoms"] += 1
            field_stats[field]["probabilities"].append(p_true)
            field_stats[field]["atoms"].append(
                {
                    "id": atom_id,
                    "text": atom_text,
                    "p_true": p_true,
                    "variable": marginal_data.get("variable", ""),
                }
            )

            if p_true > 0.8:
                field_stats[field]["high_confidence_correct"] += 1
            elif p_true > 0.6:
                field_stats[field]["likely_correct"] += 1
            elif p_true >= 0.4:
                field_stats[field]["uncertain"] += 1
            elif p_true >= 0.2:
                field_stats[field]["likely_incorrect"] += 1
            else:
                field_stats[field]["high_confidence_incorrect"] += 1

    summary = {
        "total_fields": len(field_stats),
        "fields_with_errors": 0,
        "most_problematic_field": None,
        "most_accurate_field": None,
        "overall_field_accuracy": {},
    }

    for field, stats in field_stats.items():
        if stats["total_atoms"] > 0:
            # Exclude neutral scores (0.5) from averaging -- they indicate no evidence
            non_neutral_probabilities = [p for p in stats["probabilities"] if p != 0.5]

            if len(non_neutral_probabilities) == 0:
                stats["avg_probability"] = 0.5
                stats["all_neutral"] = True
            else:
                stats["avg_probability"] = sum(non_neutral_probabilities) / len(
                    non_neutral_probabilities
                )
                stats["all_neutral"] = False

            stats["non_neutral_count"] = len(non_neutral_probabilities)
            stats["neutral_count"] = len(stats["probabilities"]) - len(non_neutral_probabilities)

            accurate_atoms = stats["high_confidence_correct"] + stats["likely_correct"]
            stats["accuracy_percentage"] = (accurate_atoms / stats["total_atoms"]) * 100

            error_atoms = stats["likely_incorrect"] + stats["high_confidence_incorrect"]
            stats["error_percentage"] = (error_atoms / stats["total_atoms"]) * 100

            if error_atoms > 0:
                summary["fields_with_errors"] += 1

            summary["overall_field_accuracy"][field] = stats["accuracy_percentage"]

    if summary["overall_field_accuracy"]:
        most_accurate = max(summary["overall_field_accuracy"].items(), key=lambda x: x[1])
        least_accurate = min(summary["overall_field_accuracy"].items(), key=lambda x: x[1])

        summary["most_accurate_field"] = {
            "field": most_accurate[0],
            "accuracy": most_accurate[1],
        }
        summary["most_problematic_field"] = {
            "field": least_accurate[0],
            "accuracy": least_accurate[1],
        }

    return {"summary": summary, "field_details": field_stats}


def print_clean_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> None:
    """Print each atom with its factuality probability score."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_counter = 1
    for atom_id, atom in atoms_by_id.items():
        atom_text = atom.get("text", "")
        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])
            logger.debug("Atom %d: %s, Probability=%.3f", atom_counter, atom_text, p_true)
        else:
            logger.debug("Atom %d: %s, Probability=N/A", atom_counter, atom_text)
        atom_counter += 1


def create_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build a per-atom summary with field labels and scores, sorted lowest first."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_summary = []

    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            if p_true > 0.8:
                confidence = "HIGH_CONFIDENCE_CORRECT"
            elif p_true > 0.6:
                confidence = "LIKELY_CORRECT"
            elif p_true >= 0.4:
                confidence = "UNCERTAIN"
            elif p_true >= 0.2:
                confidence = "LIKELY_INCORRECT"
            else:
                confidence = "HIGH_CONFIDENCE_INCORRECT"

            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": round(p_true, 4),
                    "confidence_level": confidence,
                    "variable_name": marginal_data.get("variable", ""),
                }
            )
        else:
            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": None,
                    "confidence_level": "NO_SCORE",
                    "variable_name": None,
                }
            )

    # Sort by factuality score (lowest first to highlight problems)
    atom_summary.sort(
        key=lambda x: x["factuality_score"] if x["factuality_score"] is not None else -1
    )

    return atom_summary


def _create_factreasoner_llm_handler():
    """Create a FactReasoner LLMHandler using the workflow's configured engine credentials."""
    try:
        from auto_benchmarkcard.config import Config
        from fact_reasoner.llm_handler import LLMHandler as FRLLMHandler

        engine_type = Config.LLM_ENGINE_TYPE.lower()
        model_id = Config.FACTREASONER_MODEL

        if engine_type == "rits":
            api_key = os.environ.get("RITS_API_KEY")
            if api_key:
                return FRLLMHandler(
                    model_id=model_id,
                    backend="rits",
                    api_key=api_key,
                )
            return None

        if engine_type == "hf":
            api_key = os.environ.get("HF_TOKEN")
            api_base = os.environ.get(
                "HF_API_URL", "https://router.huggingface.co/v1"
            )
            if not api_key:
                logger.warning("HF_TOKEN not set; FactReasoner will use defaults")
                return None
            extra_headers = {}
            hf_org = os.environ.get("HF_ORG")
            if hf_org:
                extra_headers["X-HF-Bill-To"] = hf_org
            return FRLLMHandler(
                model_id=model_id,
                backend="openai",
                api_base=api_base,
                api_key=api_key,
                extra_headers=extra_headers or None,
            )

        # Generic OpenAI-compatible fallback
        api_base = os.environ.get("FACTREASONER_API_BASE") or os.environ.get("LLM_API_BASE")
        api_key = (
            os.environ.get("FACTREASONER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if api_base and api_key:
            return FRLLMHandler(
                model_id=model_id,
                backend="openai",
                api_base=api_base,
                api_key=api_key,
            )
    except Exception:
        logger.debug("Could not create FactReasoner LLM handler", exc_info=True)
    return None


def evaluate_factuality(
    formatted_rag_results: Dict[str, Any],
    model: str = "llama-3.3-70b-instruct",
    nli_prompt_version: str = "v1",
    cache_dir: str = "factreasoner_cache",
    merlin_path: str = "merlin/bin/merlin",
    debug_mode: bool = False,
    use_priors: bool = False,
) -> Dict[str, Any]:
    """Run NLI-based factuality evaluation on benchmark card claims.

    Uses FactReasoner to check whether source evidence supports, contradicts,
    or is neutral about each extracted claim, then runs probabilistic reasoning
    to produce per-claim confidence scores.
    """
    os.makedirs(cache_dir, exist_ok=True)
    llm_handler = _create_factreasoner_llm_handler()

    from auto_benchmarkcard.config import Config
    context_retriever = ContextRetriever(service_type="langchain", top_k=Config.DEFAULT_TOP_K, cache_dir=cache_dir)
    atom_extractor = AtomExtractor(model, llm_handler=llm_handler)
    atom_reviser = AtomReviser(model, llm_handler=llm_handler)
    nli_extractor = NLIExtractor(model, prompt_version=nli_prompt_version, llm_handler=llm_handler)

    pipeline = FactReasoner(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=merlin_path,
        debug_mode=False,
        use_priors=use_priors,
    )

    pipeline.from_dict_with_contexts(data=formatted_rag_results)

    # Suppress FactReasoner's verbose stdout/stderr during processing
    from contextlib import redirect_stderr, redirect_stdout

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            pipeline.build(
                has_atoms=True,
                has_contexts=True,
                revise_atoms=False,
                rel_atom_context=True,
                rel_context_context=False,
                contexts_per_atom_only=True,
                remove_duplicates=False,
            )
            results, marginals = pipeline.score()

    entropy = 0.0
    valid_marginals = []

    eps = 1e-7
    for info in marginals:
        var = info.get("variable")
        probs = info.get("probabilities")
        if probs and len(probs) > 1:
            p_true = max(probs[1], eps)
            p_false = max(1.0 - p_true, eps)
            entropy += -p_true * math.log10(p_true) - p_false * math.log10(p_false)
            valid_marginals.append({"variable": var, "probabilities": probs, "p_true": probs[1]})

    n = len(valid_marginals)
    normalized_entropy = entropy / n if n > 0 else 0.0
    scaled_entropy = entropy / math.log10(n) if n > 1 else 0.0

    field_analysis = analyze_factuality_by_field(formatted_rag_results, valid_marginals)
    atom_summary = create_atom_summary(formatted_rag_results, valid_marginals)

    flagged_count = len([m for m in valid_marginals if m.get("p_true", 1) < 0.3])
    logger.debug(
        "FactReasoner: %d claims evaluated, %d/%d flagged",
        len(valid_marginals), flagged_count, len(valid_marginals),
    )

    return {
        "results": results,
        "marginals": valid_marginals,
        "entropy_metrics": {
            "total_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "scaled_entropy": scaled_entropy,
            "num_variables": n,
        },
        "fact_graph_info": {
            "num_atoms": (
                len(pipeline.fact_graph.atoms) if hasattr(pipeline.fact_graph, "atoms") else 0
            ),
            "num_contexts": (
                len(pipeline.fact_graph.contexts) if hasattr(pipeline.fact_graph, "contexts") else 0
            ),
        },
        "field_analysis": field_analysis,
        "atom_summary": atom_summary,
    }


def flag_benchmark_card_fields(
    benchmark_card: Dict[str, Any],
    field_analysis: Dict[str, Any],
    threshold: float = 0.8,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a flagged_fields section to the benchmark card for low-confidence fields.

    Fields with provenance evidence are not flagged even if the NLI score is neutral,
    since the NLI model may simply fail to match wording.
    """

    flagged_card = copy.deepcopy(benchmark_card)
    field_details = field_analysis.get("field_details", {})

    flagged_fields = {}

    for field_name, field_stats in field_details.items():
        if field_name == "benchmark_details.name" or field_name.endswith(".name"):
            continue

        should_flag, reason, reason_desc = _determine_flag_reason(field_stats, threshold)

        if should_flag and provenance:
            parts = field_name.split(".")
            section_key = parts[0] if len(parts) > 0 else ""
            field_key = parts[1] if len(parts) > 1 else ""

            section_prov = provenance.get(section_key, {})
            field_prov = section_prov.get(field_key, {})

            if field_prov.get("source") and field_prov.get("evidence"):
                # Provenance has clear source and evidence -- the NLI model
                # just couldn't match the wording. Don't flag as hallucination.
                should_flag = False
                logger.debug(
                    "Skipping flag for %s: provenance confirms source=%s",
                    field_name,
                    field_prov["source"],
                )

        if should_flag:
            if reason == "all_atoms_neutral":
                flag_reason = (
                    "[Possible Hallucination], no supporting evidence found in source material"
                )
            else:
                avg_prob = field_stats.get("avg_probability", 1.0)
                flag_reason = f"[Factuality Score: {avg_prob:.2f}], low factual alignment with source material"

            flagged_fields[field_name] = flag_reason

    if flagged_fields:
        flagged_card["flagged_fields"] = flagged_fields

    return flagged_card


def save_factuality_results(results: Dict[str, Any], output_path: str) -> None:
    """Save factuality evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_formatted_rag_results(file_path: str) -> Dict[str, Any]:
    """Load formatted RAG results from a JSON or JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            line = f.readline().strip()
            if line:
                return json.loads(line)
            else:
                raise ValueError("Empty JSONL file")
    else:
        with open(file_path, "r") as f:
            return json.load(f)


def main():
    """CLI interface for factuality evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate factuality of RAG results and flag benchmark cards"
    )
    parser.add_argument("input_file", help="Path to formatted RAG results file (.json or .jsonl)")
    parser.add_argument("--benchmark-card", help="Path to original benchmark card (for flagging)")
    parser.add_argument(
        "--output-dir",
        default="factuality_results",
        help="Output directory for results",
    )
    parser.add_argument("--model", default="llama-3.3-70b-instruct", help="LLM model name")
    parser.add_argument("--cache-dir", default="factreasoner_cache", help="Cache directory")
    parser.add_argument("--merlin-path", default="merlin/bin/merlin", help="Path to merlin binary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Factuality threshold for flagging fields",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    parser.add_argument("--use-priors", action="store_true", help="Use atom/context priors")

    args = parser.parse_args()

    benchmark_name = os.path.basename(args.input_file)
    benchmark_name = (
        benchmark_name.replace("formatted_rag_results_", "")
        .replace(".jsonl", "")
        .replace(".json", "")
    )

    logger.debug("Starting FactReasoner evaluation...")

    try:
        logger.debug("Loading evidence and claims...")
        rag_results = load_formatted_rag_results(args.input_file)

        logger.debug(
            "Found %d atoms and %d contexts",
            len(rag_results.get("atoms", [])),
            len(rag_results.get("contexts", [])),
        )

        logger.debug("Running probabilistic reasoning...")
        factuality_results = evaluate_factuality(
            formatted_rag_results=rag_results,
            model=args.model,
            cache_dir=args.cache_dir,
            merlin_path=args.merlin_path,
            debug_mode=False,
            use_priors=args.use_priors,
        )

        output_path = os.path.join(args.output_dir, f"factuality_results_{benchmark_name}.json")
        save_factuality_results(factuality_results, output_path)

        if args.benchmark_card:
            logger.debug("Flagging low-confidence fields (threshold: %.2f)", args.threshold)

            with open(args.benchmark_card, "r") as f:
                benchmark_card_data = json.load(f)

            if isinstance(benchmark_card_data, dict) and "benchmark_card" in benchmark_card_data:
                benchmark_card = benchmark_card_data["benchmark_card"]
            else:
                benchmark_card = benchmark_card_data

            field_analysis = factuality_results.get("field_analysis", {})
            flagged_card = flag_benchmark_card_fields(
                benchmark_card=benchmark_card,
                field_analysis=field_analysis,
                threshold=args.threshold,
            )

            flagged_output_path = os.path.join(
                args.output_dir, f"benchmark_card_{benchmark_name}_flagged.json"
            )
            with open(flagged_output_path, "w") as f:
                json.dump({"benchmark_card": flagged_card}, f, indent=2)

            field_analysis_data = factuality_results.get("field_analysis", {})
            field_details = field_analysis_data.get("field_details", {})
            total_fields = len(field_details)

            flagged_count = 0
            for field_name, field_stats in field_details.items():
                avg_probability = field_stats.get("avg_probability", 1.0)
                all_neutral = field_stats.get("all_neutral", False)
                if avg_probability < args.threshold or all_neutral:
                    flagged_count += 1

            logger.debug("Flagged benchmark card created")
            logger.debug("Fields flagged for review: %d/%d", flagged_count, total_fields)

            if flagged_count > 0:
                logger.debug("Fields requiring review:")
                for field_name, field_stats in field_details.items():
                    avg_probability = field_stats.get("avg_probability", 1.0)
                    all_neutral = field_stats.get("all_neutral", False)
                    if avg_probability < args.threshold or all_neutral:
                        atoms_to_review = len(
                            [
                                atom
                                for atom in field_stats.get("atoms", [])
                                if atom.get("p_true", 1.0) < args.threshold
                                or atom.get("p_true", 1.0) == 0.5
                            ]
                        )
                        logger.debug(
                            "  • %s (score: %.2f, %d atoms)",
                            field_name.replace("_", " ").title(),
                            avg_probability,
                            atoms_to_review,
                        )
            else:
                logger.debug("All fields passed factuality review")

        logger.debug("FactReasoner evaluation completed")
        logger.debug("Full results: %s", json.dumps(factuality_results, indent=2))

    except Exception as e:
        logger.error("Error during factuality evaluation: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
