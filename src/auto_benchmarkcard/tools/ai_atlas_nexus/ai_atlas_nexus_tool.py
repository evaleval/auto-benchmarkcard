"""AI Atlas Nexus integration for AI risk identification.

Maps benchmark metadata to known AI risks using the BenchmarkRiskDetector.
"""

import logging
from typing import Any, Dict, List, Optional

from ai_atlas_nexus.blocks.risk_detector import BenchmarkRiskDetector
from ai_atlas_nexus.library import AIAtlasNexus

logger = logging.getLogger(__name__)


def identify_risks_with_benchmark_detector(
    ai_atlas_nexus: AIAtlasNexus,
    usecases: List[str],
    inference_engine,
    taxonomy: str = "ibm-ai-risk-atlas",
    max_risk: Optional[int] = None,
) -> List[List]:
    """Match benchmark use cases to known AI risks using the BenchmarkRiskDetector."""
    try:
        all_risks = ai_atlas_nexus.get_all_risks(taxonomy)

        benchmark_detector = BenchmarkRiskDetector(
            risks=all_risks,
            inference_engine=inference_engine,
            max_risk=max_risk,
        )

        return benchmark_detector.detect(usecases)

    except Exception as e:
        logger.error("Error in benchmark risk detection: %s", e)
        return []


def create_inference_engine():
    """Create an inference engine for risk classification using the configured LLM backend."""
    try:
        from auto_benchmarkcard.config import get_llm_handler, Config

        handler = get_llm_handler(Config.COMPOSER_MODEL)
        return handler.engine
    except Exception as e:
        logger.warning("Failed to create inference engine for risk identification: %s", e)
        logger.warning("Risk identification will be skipped.")
        return None


def identify_risks_from_benchmark_metadata(
    benchmark_card: Dict[str, Any], taxonomy: str = "ibm-risk-atlas", max_risk: int = 5
) -> Optional[List[Dict[str, Any]]]:
    """Identify AI risks from benchmark metadata using AI Atlas Nexus."""
    try:
        inference_engine = create_inference_engine()
        if not inference_engine:
            logger.warning("No inference engine available - skipping risk identification")
            return None

        ai_atlas_nexus = AIAtlasNexus()
        usecase = create_usecase_from_benchmark_card(benchmark_card)
        if not usecase:
            logger.warning("Could not create usecase description from benchmark card")
            return None

        logger.debug("Identifying potential AI risks...")

        risks = identify_risks_with_benchmark_detector(
            ai_atlas_nexus=ai_atlas_nexus,
            usecases=[usecase],
            inference_engine=inference_engine,
            taxonomy=taxonomy,
            max_risk=max_risk,
        )

        if risks and len(risks) > 0 and len(risks[0]) > 0:
            risk_objects = risks[0][:max_risk]

            formatted_risks = []
            for risk_obj in risk_objects:
                formatted_risk = {
                    "id": risk_obj.id,
                    "category": risk_obj.name,
                    "description": [risk_obj.description],
                    "tag": risk_obj.tag,
                    "type": risk_obj.type,
                    "concern": risk_obj.concern,
                    "url": risk_obj.url if risk_obj.url else None,
                    "taxonomy": risk_obj.isDefinedByTaxonomy,
                }
                formatted_risks.append(formatted_risk)

            logger.debug("Identified %d potential risks", len(formatted_risks))
            return formatted_risks
        else:
            logger.debug("No specific risks identified")
            return []

    except Exception as e:
        logger.error("Error identifying risks: %s", e)
        return None


def create_usecase_from_benchmark_card(benchmark_card: Dict[str, Any]) -> Optional[str]:
    """Build a textual use-case description from benchmark card metadata for risk detection."""
    _EMPTY = {"not specified", "not specified.", "no information found", ""}

    def _is_specified(val) -> bool:
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip().lower() not in _EMPTY
        if isinstance(val, list):
            return bool(val) and not (
                len(val) == 1 and isinstance(val[0], str) and val[0].strip().lower() in _EMPTY
            )
        return bool(val)

    def _join_list(val) -> str:
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)

    try:
        details = benchmark_card.get("benchmark_details", {})
        purpose = benchmark_card.get("purpose_and_intended_users", {})
        data = benchmark_card.get("data", {})
        methodology = benchmark_card.get("methodology", {})
        ethical = benchmark_card.get("ethical_and_legal_considerations", {})

        name = details.get("name", "")
        overview = details.get("overview", "")
        domains = details.get("domains", [])
        languages = details.get("languages", [])
        tasks = purpose.get("tasks", [])
        goal = purpose.get("goal", "")
        limitations = purpose.get("limitations", "")

        data_source = data.get("source", "")
        data_size = data.get("size", "")
        annotation = data.get("annotation", "")

        methods = methodology.get("methods", [])
        metrics = methodology.get("metrics", [])
        license_info = ethical.get("data_licensing", "")

        parts = []

        if name:
            parts.append(f"{name} is a benchmark")

        if _is_specified(overview):
            parts.append(overview)

        if _is_specified(goal):
            parts.append(f"The goal is {goal}")

        if _is_specified(data_source):
            parts.append(f"The data was sourced from: {data_source}")

        if _is_specified(data_size):
            parts.append(f"Dataset size: {data_size}")

        if _is_specified(annotation):
            parts.append(f"Annotation: {annotation}")

        if _is_specified(methods):
            parts.append(f"Evaluation methods: {_join_list(methods)}")

        if _is_specified(metrics):
            parts.append(f"Metrics: {_join_list(metrics)}")

        if _is_specified(languages):
            parts.append(f"Languages: {_join_list(languages)}")

        if _is_specified(domains):
            parts.append(f"Domains: {_join_list(domains)}")

        if _is_specified(tasks):
            parts.append(f"Tasks: {_join_list(tasks)}")

        if _is_specified(limitations):
            parts.append(f"Known limitations: {limitations}")

        if _is_specified(license_info):
            parts.append(f"Data license: {license_info}")

        if parts:
            usecase = ". ".join(parts).strip()
            if not usecase.endswith("."):
                usecase += "."
            return usecase
        else:
            logger.warning("Insufficient information to create usecase description")
            return None

    except Exception as e:
        logger.error("Error creating usecase from benchmark card: %s", e)
        return None


def integrate_risks_into_benchmark_card(
    benchmark_card: Dict[str, Any], risks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Add identified risks to the benchmark card under 'possible_risks'."""
    try:
        updated_card = benchmark_card.copy()

        cleaned_risks = []
        for risk in risks:
            cleaned_risk = {
                "category": risk["category"],
                "description": risk["description"],
                "url": risk.get("url") or None,
            }
            cleaned_risks.append(cleaned_risk)

        updated_card["possible_risks"] = cleaned_risks

        logger.debug("Successfully integrated %d risks into benchmark card", len(cleaned_risks))
        return updated_card

    except Exception as e:
        logger.error("Error integrating risks into benchmark card: %s", e)
        return benchmark_card


def identify_and_integrate_risks(benchmark_card: Dict[str, Any]) -> Dict[str, Any]:
    """Identify AI risks and integrate them into the benchmark card."""
    try:
        logger.debug("Running AI Atlas Nexus analysis...")

        risks = identify_risks_from_benchmark_metadata(benchmark_card)

        if risks is None:
            logger.warning("Risk identification failed - returning original benchmark card")
            return benchmark_card

        if not risks:
            logger.debug("No risks identified - returning original benchmark card")
            return benchmark_card

        updated_card = integrate_risks_into_benchmark_card(benchmark_card, risks)

        logger.debug("Risk analysis completed")
        return updated_card

    except Exception as e:
        logger.error("Error in risk identification and integration: %s", e)
        return benchmark_card
