"""Build the audited S150 abstract-recovery manifest from Firecrawl artifacts.

Nineteen records come from the already resolved OpenAlex works. Two OpenAlex
records have no abstract and use their official Springer chapter pages. This
script performs no network access and refuses to overwrite an existing
manifest.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

EXPECTED_TITLES = {
    "360vot": "360VOT: A New Benchmark Dataset for Omnidirectional Visual Object Tracking",
    "activitynet": "ActivityNet: A large-scale video benchmark for human activity understanding",
    "assistgui": "AssistGUI: Task-Oriented PC Graphical User Interface Automation",
    "cbnsl": "Curriculum Learning of Bayesian Network Structures",
    "dtto": "Tracking Transforming Objects: A Benchmark",
    "giantsteps-tempo": "Two Data Sets For Tempo Estimation And Key Detection In Electronic Dance Music Annotated From User Corrections.",
    "hypersim": "Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding",
    "lvos": "LVOS: A Benchmark for Long-term Video Object Segmentation",
    "mega-xcopa": "MEGA: Multilingual Evaluation of Generative AI",
    "metaclue": "MetaCLUE: Towards Comprehensive Visual Metaphors Research",
    "multipl-e": "MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation",
    "music-avqa": "Learning to Answer Questions in Dynamic Audio-Visual Scenarios",
    "nt-vot211": "NT-VOT211: A Large-Scale Benchmark for Night-Time Visual Object Tracking",
    "sifo": "The SIFo Benchmark: Investigating the Sequential Instruction Following Ability of Large Language Models",
    "sunrgbd": "SUN RGB-D: A RGB-D scene understanding benchmark suite",
    "tldr9-test": "TLDR9+: A Large Scale Resource for Extreme Summarization of Social Media Posts",
    "uniform-bar-exam": "GPT-4 passes the bar exam",
    "union14m": "Revisiting Scene Text Recognition: A Data Perspective",
    "video-mme": "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis",
    "vqa-rad": "A dataset of clinically generated visual questions and answers about radiology images",
    "xsum": "Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization",
}

SPRINGER_URLS = {
    "dtto": "https://link.springer.com/chapter/10.1007/978-981-97-8493-6_16",
    "nt-vot211": "https://link.springer.com/chapter/10.1007/978-981-96-0901-7_19",
}


class ManifestError(RuntimeError):
    pass


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read valid JSON from {path}: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("\u00a0", " ").split())


def _read_firecrawl_json(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if not match:
        raise ManifestError(f"unexpected Firecrawl JSON wrapper: {path}")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid Firecrawl JSON payload in {path}: {exc}") from exc


def _openalex_abstract(record: dict, *, card: str) -> str:
    index = record.get("abstract_inverted_index")
    if not isinstance(index, dict) or not index:
        raise ManifestError(f"OpenAlex record has no abstract for {card}")
    positioned: dict[int, str] = {}
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            raise ManifestError(f"invalid OpenAlex abstract index for {card}")
        for position in positions:
            if not isinstance(position, int) or position < 0 or position in positioned:
                raise ManifestError(f"invalid/duplicate abstract position for {card}: {position}")
            positioned[position] = token
    expected = list(range(max(positioned) + 1))
    if sorted(positioned) != expected:
        raise ManifestError(f"OpenAlex abstract positions are not contiguous for {card}")
    return " ".join(positioned[position] for position in expected)


def _select_openalex_record(payload, *, card: str, expected_title: str) -> dict:
    candidates = payload.get("results") if isinstance(payload, dict) else None
    if candidates is None:
        candidates = [payload]
    exact = [
        record
        for record in candidates
        if isinstance(record, dict)
        and _normalize(record.get("title", "")) == _normalize(expected_title)
    ]
    if len(exact) != 1:
        raise ManifestError(f"expected one exact OpenAlex title match for {card}, found {len(exact)}")
    return exact[0]


def _springer_abstract(path: Path, *, card: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Abstract\s*\n+(.+?)(?:\n\n|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ManifestError(f"cannot locate Springer abstract for {card}")
    abstract = " ".join(match.group(1).split())
    if not abstract:
        raise ManifestError(f"empty Springer abstract for {card}")
    return abstract


def _abstract_only_cards(sample: dict, repo: Path) -> dict[str, str]:
    result = {}
    for entry in sample.get("cards", []):
        name = entry.get("name")
        source_run_dir = entry.get("source_run_dir")
        if not isinstance(name, str) or not isinstance(source_run_dir, str):
            raise ManifestError("invalid sample card entry")
        run_dir = (repo / source_run_dir).resolve()
        paths = sorted(glob.glob(str(run_dir / "tool_output/composer/docling_telemetry_*.json")))
        if len(paths) != 1:
            raise ManifestError(f"expected one Docling telemetry file for {name}")
        telemetry = _load_json(Path(paths[0]))
        if telemetry.get("degraded_to_abstract_only") is True:
            result[name] = source_run_dir
    return result


def build_manifest(
    *, repo: Path, sample_path: Path, firecrawl_dir: Path, output_path: Path,
    recovered_at: str,
) -> dict:
    if output_path.exists():
        raise ManifestError(f"refusing to overwrite {output_path}")
    sample = _load_json(sample_path)
    expected_cards = _abstract_only_cards(sample, repo)
    if set(expected_cards) != set(EXPECTED_TITLES):
        raise ManifestError(
            "abstract-only card set drifted: "
            f"missing={sorted(set(EXPECTED_TITLES) - set(expected_cards))}, "
            f"extra={sorted(set(expected_cards) - set(EXPECTED_TITLES))}"
        )

    cards = {}
    for card, expected_title in sorted(EXPECTED_TITLES.items()):
        openalex_path = firecrawl_dir / f"openalex-{card}.md"
        openalex_payload = _read_firecrawl_json(openalex_path)
        openalex_record = _select_openalex_record(
            openalex_payload, card=card, expected_title=expected_title
        )
        openalex_id = openalex_record.get("id")
        if not isinstance(openalex_id, str) or not openalex_id.startswith("https://openalex.org/"):
            raise ManifestError(f"invalid OpenAlex id for {card}")

        if card in SPRINGER_URLS:
            source_path = firecrawl_dir / f"springer-{card}.md"
            abstract = _springer_abstract(source_path, card=card)
            recovery_source_url = SPRINGER_URLS[card]
            recovery_method = "official Springer chapter page scraped by Firecrawl"
        else:
            source_path = openalex_path
            abstract = _openalex_abstract(openalex_record, card=card)
            recovery_source_url = (
                "https://api.openalex.org/works/" + openalex_id.rsplit("/", 1)[-1]
            )
            recovery_method = (
                "OpenAlex abstract_inverted_index reconstructed in token order "
                "from Firecrawl-fetched JSON"
            )

        cards[card] = {
            "source_run_dir": expected_cards[card],
            "paper_title": expected_title,
            "paper_abstract": abstract,
            "paper_abstract_sha256": _sha256_bytes(abstract.encode("utf-8")),
            "recovery_source_url": recovery_source_url,
            "recovered_at": recovered_at,
            "recovery_method": recovery_method,
            "recovery_artifact": str(source_path.resolve()),
            "recovery_artifact_sha256": _sha256_file(source_path),
            "openalex_id": openalex_id,
            "original_resolved_doi": openalex_record.get("doi"),
        }

    manifest = {
        "version": "s150-paper-abstracts-v1",
        "status": "post-run evidence recovery; original full abstract strings were not persisted",
        "recovered_at": recovered_at,
        "cards": cards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--sample", default="eval/s150/sample.json")
    parser.add_argument("--firecrawl-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--recovered-at", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sample_path = Path(args.sample)
    if not sample_path.is_absolute():
        sample_path = repo / sample_path
    manifest = build_manifest(
        repo=repo,
        sample_path=sample_path,
        firecrawl_dir=Path(args.firecrawl_dir).resolve(),
        output_path=Path(args.out).resolve(),
        recovered_at=args.recovered_at,
    )
    print(f"wrote {len(manifest['cards'])} recovered abstracts to {args.out}")


if __name__ == "__main__":
    main()
