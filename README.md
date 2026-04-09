# Auto-BenchmarkCard

Automated generation of validated benchmark documentation for AI/NLP benchmarks.

Benchmark documentation is often incomplete, inconsistent, or scattered across different sources. Auto-BenchmarkCard pulls together metadata from Hugging Face, Unitxt, and academic papers, synthesizes it into a structured BenchmarkCard using LLMs, and then fact-checks the result against the original sources using [FactReasoner](https://github.com/evaleval/FactReasoner).

<img width="1050" height="335" alt="Auto-BenchmarkCard architecture diagram" src="https://github.com/user-attachments/assets/c4c1992e-b0c9-4a3c-bc5a-89716b9ff215" />

## How it works

The workflow has three phases:

**Extraction** gathers raw data from multiple sources. The Unitxt tool fetches benchmark definitions from the Unitxt catalog. The HuggingFace tool loads dataset READMEs and metadata. The Docling tool converts referenced academic papers into structured text.

**Composition** takes all extracted data and feeds it to an LLM that produces a structured BenchmarkCard. After that, [AI Atlas Nexus](https://github.com/IBM/risk-atlas-nexus) maps the benchmark to relevant AI risk categories.

**Validation** breaks the generated card into atomic claims, retrieves evidence for each claim using BM25 + vector search + LLM reranking, and sends claim-evidence pairs to FactReasoner. Each claim is classified as supported, contradicted, or neutral with a confidence score. Fields with low factuality or missing evidence get flagged for human review.

The whole pipeline is orchestrated as a LangGraph state machine where each tool runs as a worker node.

## Quick start

```bash
git clone https://github.com/evaleval/auto-benchmarkcard.git
cd auto-benchmarkcard
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Create a `.env` file:

```bash
LLM_ENGINE_TYPE=hf
HF_TOKEN=<your-hf-token>
HF_COMPOSER_MODEL=deepseek-ai/DeepSeek-V3.1
FACTREASONER_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

Generate a benchmark card:

```bash
benchmarkcard generate-unitxt glue -o ./output
```

The default LLM backend is HuggingFace Inference Providers. Other supported backends are Ollama (local), vLLM, WML, and RITS. Set `LLM_ENGINE_TYPE` in your `.env` accordingly.

## Setting up Merlin (for FactReasoner)

FactReasoner uses [Merlin](https://github.com/arishofmann/merlin) for probabilistic inference. This step is required for the validation phase.

```bash
git clone https://github.com/arishofmann/merlin.git external/merlin
cd external/merlin
brew install boost
make clean && make
cd ../..
```

Verify with `./external/merlin/bin/merlin --help`.

## Usage

### CLI

```bash
# From evaluation data (supports multiple benchmarks)
benchmarkcard generate ./external/eee_samples -b "MMLU,TruthfulQA" -o ./output

# From the Unitxt catalog
benchmarkcard generate-unitxt glue -o ./output

# List previous sessions
benchmarkcard list -o ./output

# Check your environment
benchmarkcard validate
```

Add `--debug` to any command for detailed logging.

### Python API

```python
from auto_benchmarkcard.workflow import build_workflow
from auto_benchmarkcard.output import OutputManager

output_manager = OutputManager("glue")
workflow = build_workflow()
state = workflow.invoke({
    "query": "glue",
    "output_manager": output_manager,
    "catalog_path": None,
    "unitxt_json": None,
    "extracted_ids": None,
    "hf_repo": None,
    "hf_json": None,
    "docling_output": None,
    "composed_card": None,
    "risk_enhanced_card": None,
    "completed": [],
    "errors": [],
    "hf_extraction_attempted": False,
    "rag_results": None,
    "factuality_results": None,
})
```

### Batch processing

The batch script processes all benchmarks from the Unitxt catalog:

```bash
python scripts/batch_process.py
python scripts/batch_process.py --limit 10       # first 10 only
python scripts/batch_process.py --no-skip         # reprocess existing
```

It tracks progress, skips already-processed benchmarks, and saves failure logs for review.

## Output

Each run creates a timestamped directory:

```
output/glue_2025-01-08_14-30/
├── tool_output/
│   ├── unitxt/           # Unitxt benchmark definitions
│   ├── hf/               # Hugging Face metadata
│   ├── docling/           # Processed papers
│   ├── extractor/         # Extracted IDs and URLs
│   ├── rag/               # Evidence retrieval results
│   ├── factreasoner/      # Factuality scores
│   └── ai_atlas_nexus/    # Risk assessment
└── auto_benchmarkcard/
    └── benchmark_card_glue.json
```

The final `benchmark_card_*.json` contains the structured card, risk annotations, factuality scores, and a list of flagged fields that need human review.

## Project structure

```
auto_benchmarkcard/
├── src/auto_benchmarkcard/
│   ├── workflow.py        # LangGraph orchestration
│   ├── workers.py         # Worker nodes for each pipeline step
│   ├── state.py           # Graph state definition
│   ├── output.py          # Output directory management
│   ├── card_utils.py      # Card normalization and HF tag overrides
│   ├── config.py          # Environment and model configuration
│   ├── cli.py             # Typer CLI
│   ├── llm_handler.py     # LLM engine abstraction
│   └── tools/
│       ├── unitxt/        # Unitxt catalog lookup
│       ├── extractor/     # ID and URL extraction
│       ├── hf/            # Hugging Face metadata
│       ├── docling/       # Paper conversion
│       ├── composer/      # LLM-based card generation
│       ├── ai_atlas_nexus/ # Risk identification
│       ├── rag/           # Evidence retrieval
│       ├── factreasoner/  # Fact verification
│       └── eee/           # Evaluation data adapter
├── scripts/
│   └── batch_process.py
├── external/
│   └── merlin/
├── pyproject.toml
└── .env
```

## References

A. Sokol et al., "BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks," 2025, arXiv:2410.12974.

R. Marinescu et al., "FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models," 2025, arXiv:2502.18573.

F. Bagehorn et al., "AI Risk Atlas: Taxonomy and Tooling for Navigating AI Risks and Resources," 2025, arXiv:2503.05780.
