"""LangGraph worker nodes for the benchmark processing pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict

from auto_benchmarkcard.card_utils import (
    apply_deterministic_overrides,
    backfill_from_provenance,
    extract_card,
    extract_hf_tags,
    extract_missing_fields,
    normalize_not_specified,
)
from auto_benchmarkcard.config import Config
from auto_benchmarkcard.output import sanitize_benchmark_name
from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import identify_and_integrate_risks
from auto_benchmarkcard.tools.composer.composer_tool import compose_benchmark_card
from auto_benchmarkcard.tools.docling.docling_tool import extract_paper_with_docling
from auto_benchmarkcard.tools.extractor.extractor_tool import extract_ids
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (
    evaluate_factuality,
    flag_benchmark_card_fields,
)
from auto_benchmarkcard.tools.hf.hf_tool import hf_dataset_metadata
from auto_benchmarkcard.tools.rag.atomizer import atomize_benchmark_card
from auto_benchmarkcard.tools.rag.format_converter import (
    convert_rag_to_required_format,
    save_formatted_results,
)
from auto_benchmarkcard.tools.rag.indexer import MetadataIndexer
from auto_benchmarkcard.tools.rag.rag_retriever import RAGRetriever
from auto_benchmarkcard.tools.unitxt import unitxt_tool

logger = logging.getLogger(__name__)

# Identity fields are descriptive (relaxed threshold); analytical fields
# are reasoned from context (skip NLI-based flagging entirely).
_IDENTITY_FIELDS = {
    "benchmark_details.overview", "benchmark_details.domains",
    "benchmark_details.data_type", "benchmark_details.similar_benchmarks",
    "purpose_and_intended_users.goal", "purpose_and_intended_users.audience",
    "purpose_and_intended_users.tasks",
}
_ANALYTICAL_FIELDS = {
    "purpose_and_intended_users.limitations",
    "purpose_and_intended_users.out_of_scope_uses",
    "methodology.interpretation",
}


def handle_error(error: Exception, operation: str, state) -> Dict[str, Any]:
    """Log an exception and return updated state with the failure recorded."""
    error_msg = f"{operation} failed: {error}"
    logger.error(error_msg, exc_info=True)

    if hasattr(error, "original_error") and error.original_error:
        logger.error("Original error: %s", error.original_error)

    errors = state.get("errors", [])
    errors.append(error_msg)

    return {"errors": errors, "completed": [f"{operation.lower()} failed"]}


def record_skip(message: str, operation: str, state) -> Dict[str, Any]:
    """Record a skipped step when preconditions are not met (no traceback)."""
    logger.warning("%s: %s", operation, message)
    errors = state.get("errors", [])
    errors.append(f"{operation}: {message}")
    return {"errors": errors, "completed": [f"{operation.lower()} failed"]}


def run_unitxt(state) -> Dict[str, Any]:
    """Retrieve UnitXT metadata for the benchmark."""
    try:
        result = unitxt_tool.unitxt_benchmark_lookup(
            state["query"], catalog_path=state.get("catalog_path")
        )
        unitxt_data = result.model_dump(mode="json")

        logger.info("UnitXT metadata retrieved")
        name = unitxt_data.get("name", "N/A")
        description = unitxt_data.get("description", "")
        if description:
            desc_preview = description[:60] + "..." if len(description) > 60 else description
            logger.info(f"Found: {name} - {desc_preview}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(unitxt_data, "unitxt", filename)
        logger.info("UnitXT output saved to: %s", output_file)

        return {
            "unitxt_json": unitxt_data,
            "completed": ["unitxt done"],
        }
    except Exception as e:
        return handle_error(e, "UnitXT lookup", state)


def run_extractor(state) -> Dict[str, Any]:
    """Extract HF repo ID, paper URL, and risk tags from UnitXT data."""
    logger.info("Starting ID and URL extraction")
    try:
        extracted = extract_ids.func(
            source=state["unitxt_json"], want=["hf_repo", "paper_url", "risk_tags"]
        )

        hf_repo = extracted.get("hf_repo")
        paper_url = extracted.get("paper_url")

        logger.info("ID extraction completed")
        hf_status = hf_repo if hf_repo else "None"
        paper_status = "Found" if paper_url else "None"
        logger.info(f"Extracted: HF={hf_status}, Paper={paper_status}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(extracted, "extractor", filename)
        logger.info("Extractor output saved to: %s", output_file)

        return {
            "extracted_ids": extracted,
            "hf_repo": hf_repo,
            "completed": [f"extract hf_repo={hf_repo}, paper_url={paper_url}"],
        }
    except Exception as e:
        return handle_error(e, "ID extraction", state)


def run_hf_extractor(state):
    """Extract paper URL from HF metadata when UnitXT didn't provide one."""
    logger.info("Starting HuggingFace extraction")
    try:
        current_extracted = state.get("extracted_ids", {})
        hf_data = state["hf_json"]
        paper_url = None

        hf_extracted = extract_ids.func(source=hf_data, want=["paper_url"])
        paper_url = hf_extracted.get("paper_url")
        if paper_url:
            logger.info("Found paper_url in HF top-level metadata: %s", paper_url)

        if not paper_url:
            for dataset_id, dataset_metadata in hf_data.items():
                if isinstance(dataset_metadata, dict):
                    hf_extracted = extract_ids.func(source=dataset_metadata, want=["paper_url"])
                    extracted_paper_url = hf_extracted.get("paper_url")
                    if extracted_paper_url:
                        paper_url = extracted_paper_url
                        logger.info("Found paper_url in HF dataset %s: %s", dataset_id, paper_url)
                        break

        if paper_url:
            updated_extracted = current_extracted.copy()
            updated_extracted["paper_url"] = paper_url
            updated_extracted["paper_url_from_hf"] = paper_url

            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                updated_extracted, "extractor", filename
            )
            logger.info("HF extractor output saved to: %s", output_file)

            return {
                "extracted_ids": updated_extracted,
                "hf_extraction_attempted": True,
                "completed": [f"hf_extract paper_url={paper_url}"],
            }
        else:
            logger.info("No paper_url found in HF metadata")
            return {
                "hf_extraction_attempted": True,
                "completed": ["hf_extract no paper_url found"],
            }

    except Exception as e:
        result = handle_error(e, "HF extraction", state)
        result["hf_extraction_attempted"] = True
        return result


def run_docling(state):
    """Extract paper content using Docling."""
    paper_url = state.get("extracted_ids", {}).get("paper_url")
    if paper_url:
        logger.info("Starting paper extraction")

    if not paper_url:
        return {
            "docling_output": None,
            "completed": ["docling skipped (no paper_url)"],
        }

    try:
        logger.info("Extracting paper from: %s", paper_url)

        docling_result = extract_paper_with_docling.func(paper_url=paper_url)

        if docling_result.get("success"):
            logger.info("Docling extraction completed successfully")
            metadata = docling_result.get("metadata", {})
            title = metadata.get("title", "Unknown Paper")
            char_count = len(docling_result.get("filtered_text", ""))
            logger.info(f"Paper: {title} ({char_count:,} chars)")

            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                docling_result, "docling", filename
            )
            logger.info("Docling output saved to: %s", output_file)

            return {
                "docling_output": docling_result,
                "completed": ["docling done"],
            }
        else:
            warning_msg = docling_result.get("warning")
            if warning_msg:
                logger.warning("Docling warning: %s", warning_msg)
                return {
                    "docling_output": None,
                    "completed": ["docling warning - continuing without paper"],
                }
            else:
                error_msg = (
                    f"Docling extraction failed: {docling_result.get('error', 'Unknown error')}"
                )
                result = record_skip(error_msg, "Docling extraction", state)
                result["docling_output"] = None
                return result

    except Exception as e:
        result = handle_error(e, "Docling extraction", state)
        result["docling_output"] = None
        return result


def run_hf(state):
    """Fetch HuggingFace dataset metadata."""
    if not state["hf_repo"]:
        return record_skip("No hf_repo available", "HuggingFace lookup", state)

    try:
        hf_data = hf_dataset_metadata.func(repo_id=state["hf_repo"])

        logger.info("HuggingFace metadata retrieved successfully")
        card_data = hf_data.get("card_data") or {}
        dataset_info = hf_data.get("dataset_info") or {}
        name = dataset_info.get("dataset_name", card_data.get("pretty_name", "Unknown"))
        task_cats = card_data.get("task_categories", [])
        if task_cats:
            task_preview = ", ".join(task_cats[:2])
            logger.info(f"Dataset: {name} ({task_preview})")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(hf_data, "hf", filename)
        logger.info("HuggingFace output saved to: %s", output_file)

        return {
            "hf_json": hf_data,
            "completed": ["hf done"],
        }
    except Exception as e:
        return handle_error(e, "HuggingFace lookup", state)


def run_composer(state):
    """Compose a benchmark card from all collected metadata using an LLM."""
    logger.info("Starting benchmark card composition")

    try:
        # Extract just the benchmark name from catalog queries (after the last dot)
        query_for_composer = state["query"]
        if state.get("catalog_path") and "." in state["query"]:
            query_for_composer = state["query"].split(".")[-1]

        unitxt_for_composer = state.get("unitxt_json") if not state.get("eee_metadata") else None

        result = compose_benchmark_card.func(
            unitxt_metadata=unitxt_for_composer,
            hf_metadata=state.get("hf_json"),
            extracted_ids=state.get("extracted_ids", {}),
            docling_output=state.get("docling_output"),
            query=query_for_composer,
            eee_metadata=state.get("eee_metadata"),
        )

        logger.info("Successfully composed benchmark card")

        if result.get("provenance"):
            provenance_filename = f"provenance_{sanitize_benchmark_name(state['query'])}.json"
            provenance_output = state["output_manager"].save_tool_output(
                result["provenance"], "composer", provenance_filename
            )
            logger.info(f"Provenance tracking saved to: {provenance_output}")

        benchmark_card = result.get("benchmark_card", {})
        details = benchmark_card.get("benchmark_details", {})
        name = details.get("name", "N/A")
        domains = details.get("domains", [])
        languages = details.get("languages", [])

        domain_str = ", ".join(domains[:2]) if domains else "General"
        lang_str = ", ".join(languages[:2]) if languages else "Unknown"
        logger.info(f"Card: {name} | {domain_str} | {lang_str}")

        return {
            "composed_card": result,
            "completed": ["composer done"],
        }
    except Exception as e:
        return handle_error(e, "Composer", state)


def run_risk_identification(state):
    """Identify risks using AI Atlas Nexus and integrate them into the card."""
    logger.info("Starting risk identification")

    if not state.get("composed_card"):
        return record_skip("No composed card available", "Risk identification", state)

    try:
        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        risk_enhanced_card = identify_and_integrate_risks(benchmark_card)

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(
            {"benchmark_card": risk_enhanced_card}, "risk_enhanced", filename
        )
        logger.info("Risk-enhanced card saved to: %s", output_file)

        possible_risks = risk_enhanced_card.get("possible_risks", [])
        if possible_risks:
            risk_output = {
                "benchmark": state["query"],
                "risks_identified": len(possible_risks),
                "taxonomy": "ibm-risk-atlas",
                "risks": possible_risks,
            }
            risk_filename = f"risks_{sanitize_benchmark_name(state['query'])}.json"
            risk_output_file = state["output_manager"].save_tool_output(
                risk_output, "ai_atlas_nexus", risk_filename
            )
            logger.info("Risk identification results saved to: %s", risk_output_file)

        logger.info("Risk identification completed")
        possible_risks = risk_enhanced_card.get("possible_risks", [])

        if possible_risks:
            risk_count = len(possible_risks)
            risk_names = [
                risk.get("category", risk.get("name", "Unknown")) for risk in possible_risks[:2]
            ]
            risk_preview = ", ".join(risk_names)
            if risk_count > 2:
                risk_preview += f" (+{risk_count-2} more)"
            logger.info(f"Risks: {risk_preview}")
        else:
            logger.info("Risks: None detected")

        return {
            "risk_enhanced_card": {"benchmark_card": risk_enhanced_card},
            "completed": ["risk identification done"],
        }

    except Exception as e:
        return handle_error(e, "Risk identification", state)


def run_rag(state) -> Dict[str, Any]:
    """Atomize the benchmark card and retrieve supporting evidence via RAG."""
    logger.info("Starting RAG processing")

    if not state.get("composed_card"):
        return record_skip("No composed card available", "RAG processing", state)

    try:
        benchmark_name = sanitize_benchmark_name(state["query"])

        unitxt_data = state.get("unitxt_json") or {}
        hf_data = state.get("hf_json") or {}
        docling_data = state.get("docling_output")

        indexer = MetadataIndexer()
        documents = indexer.create_documents(unitxt_data, hf_data, state["query"], docling_data)

        from auto_benchmarkcard.config import get_llm_handler

        try:
            llm = get_llm_handler(Config.COMPOSER_MODEL)
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=Config.ENABLE_LLM_RERANKING,
                enable_hybrid_search=Config.ENABLE_HYBRID_SEARCH,
                enable_query_expansion=Config.ENABLE_QUERY_EXPANSION,
                llm_handler=llm,
            )
        except Exception as e:
            logger.warning("Enhanced retriever failed, using basic fallback: %s", e)
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=False,
                enable_hybrid_search=False,
                enable_query_expansion=False,
            )

        retriever.index_documents(documents)

        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        statements = atomize_benchmark_card(
            benchmark_card, "all",
            engine_type=Config.LLM_ENGINE_TYPE,
            model_name=Config.COMPOSER_MODEL,
        )

        statement_texts = []
        for statement_obj in statements:
            if isinstance(statement_obj, str):
                statement_texts.append(statement_obj)
            else:
                statement_texts.append(statement_obj.get("text", ""))

        logger.debug("Processing %d statements for evidence retrieval", len(statement_texts))

        if retriever.enable_llm_reranking and retriever.llm_handler:
            import asyncio

            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                logger.warning("nest_asyncio not installed, async reranking may fail in nested event loops")

            try:
                batch_chunks = asyncio.run(
                    retriever.retrieve_for_statements_batch_parallel(statement_texts)
                )
            except RuntimeError:
                logger.warning("Async reranking failed (event loop conflict), falling back to sync")
                batch_chunks = retriever.retrieve_for_statements_batch(statement_texts)
        else:
            batch_chunks = retriever.retrieve_for_statements_batch(statement_texts)

        results = []
        for statement_obj, chunks in zip(statements, batch_chunks):
            results.append({"statement": statement_obj, "retrieved_chunks": chunks})

        raw_results = {
            "benchmark": state["query"],
            "num_statements": len(statements),
            "num_documents_indexed": len(documents),
            "results": results,
        }

        formatted_results = convert_rag_to_required_format(raw_results, "all", benchmark_card)

        rag_filename = f"formatted_rag_results_{benchmark_name}.jsonl"
        rag_tool_dir = state["output_manager"].get_tool_output_path("rag")
        output_path = os.path.join(rag_tool_dir, rag_filename)
        save_formatted_results(formatted_results, output_path)

        logger.info("RAG processing completed")
        atom_count = len(formatted_results.get("atoms", []))
        context_count = len(formatted_results.get("contexts", []))
        logger.info(f"RAG: {atom_count} claims, {context_count} evidence sources")
        logger.info("RAG results saved to: %s", output_path)

        return {
            "rag_results": formatted_results,
            "completed": ["rag done"],
        }

    except Exception as e:
        return handle_error(e, "RAG processing", state)


def run_factreasoner(state):
    """Evaluate factuality of card claims and produce the final flagged card."""
    logger.info("Starting factuality evaluation")

    if not state.get("rag_results"):
        return record_skip("No RAG results available", "FactReasoner evaluation", state)

    try:
        benchmark_name = sanitize_benchmark_name(state["query"])
        rag_results = state["rag_results"]

        factuality_results = evaluate_factuality(
            formatted_rag_results=rag_results,
            model=Config.FACTREASONER_MODEL,
            cache_dir=Config.FACTREASONER_CACHE_DIR,
            merlin_path=str(Config.MERLIN_BIN),
            debug_mode=False,
            use_priors=False,
        )

        factuality_filename = f"factuality_results_{benchmark_name}.json"
        factuality_output = state["output_manager"].save_tool_output(
            factuality_results, "factreasoner", factuality_filename
        )

        risk_card_src = state.get("risk_enhanced_card") or state.get("composed_card") or {}
        clean_card = extract_card(risk_card_src)

        composed_card_data = state.get("composed_card", {})
        provenance_data = composed_card_data.get("provenance") if isinstance(composed_card_data, dict) else None

        field_analysis = factuality_results.get("field_analysis", {})

        flagged_card = flag_benchmark_card_fields(
            benchmark_card=clean_card,
            field_analysis=field_analysis,
            threshold=Config.DEFAULT_FACTUALITY_THRESHOLD,
            provenance=provenance_data,
        )

        if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
            fields_to_unflag = []
            for field_name, flag_reason in flagged_card["flagged_fields"].items():
                if field_name in _ANALYTICAL_FIELDS:
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging analytical field %s (reasoned, not extracted)", field_name)
                elif field_name in _IDENTITY_FIELDS and "Possible Hallucination" in str(flag_reason):
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging identity field %s (neutral NLI expected for descriptive content)", field_name)
            for field_name in fields_to_unflag:
                del flagged_card["flagged_fields"][field_name]
            if fields_to_unflag:
                logger.info("Field-type-aware scoring: unflagged %d identity/analytical fields", len(fields_to_unflag))

        if provenance_data:
            flagged_card = backfill_from_provenance(flagged_card, provenance_data)

        flagged_card = normalize_not_specified(flagged_card)

        hf_overrides = extract_hf_tags(state.get("hf_json"))
        if hf_overrides:
            logger.info("Applying %d deterministic HF overrides: %s",
                        len(hf_overrides), list(hf_overrides.keys()))
            flagged_card = apply_deterministic_overrides(flagged_card, hf_overrides)

            if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
                for dotted_key in hf_overrides:
                    flag_key = dotted_key.split(".")[-1]
                    for k in (flag_key, dotted_key):
                        if k in flagged_card["flagged_fields"]:
                            del flagged_card["flagged_fields"][k]
                            logger.debug("Cleared stale flag for %s (deterministic override)", k)

        flagged_card["missing_fields"] = extract_missing_fields(flagged_card)

        flagged_card["card_info"] = {
            "created_at": datetime.now().isoformat(),
            "llm": Config.COMPOSER_MODEL,
        }

        card_filename = f"benchmark_card_{benchmark_name}.json"
        output_path = state["output_manager"].save_benchmark_card(
            {"benchmark_card": flagged_card},
            card_filename,
        )

        logger.info("FactReasoner evaluation complete")
        marginals = factuality_results.get("marginals", [])
        claims_evaluated = len(marginals)
        flagged_fields = len(
            [m for m in marginals if m.get("p_true", 1.0) < 0.3 or m.get("p_true", 1.0) == 0.5]
        )
        logger.info(
            f"Factuality: {claims_evaluated} claims evaluated, {flagged_fields}/{claims_evaluated} fields flagged"
        )

        logger.info("Factuality results saved to: %s", factuality_output)
        logger.info("Final benchmark card saved to: %s", output_path)

        return {
            "factuality_results": factuality_results,
            "final_card": {"benchmark_card": flagged_card},
            "completed": ["factreasoner done"],
        }

    except Exception as e:
        return handle_error(e, "FactReasoner evaluation", state)
