# RAG evidence retrieval

This package splits a composed benchmark card into atomic claims and retrieves
source passages for each claim. FactReasoner then scores every claim against
its passages. In the pipeline, `workers.run_rag` drives these steps.

## Components

- `indexer.py`: `MetadataIndexer.create_documents` turns UnitXT, Hugging Face,
  Docling (paper), HTML and GitHub README data into LangChain documents.
- `rag_retriever.py`: `RAGRetriever` indexes the documents with parent and
  child chunks and retrieves evidence per claim. It combines vector search
  (MMR), BM25, keyword filtering, LLM query reformulation and LLM reranking.
- `atomizer.py`: `atomize_benchmark_card` asks the composer model to rewrite
  the card as statements tagged with a field path such as `data.size`. Risk
  sections, display-only fields, structured fields and "Not specified" values
  are removed before the call.
- `format_converter.py`: `convert_rag_to_required_format` builds the atoms
  and contexts structure that FactReasoner reads.
- `source_retrieval.py`: before scoring, `source_scoped_rag` re-retrieves each
  atom's contexts with BM25 from the card's own Docling, HTML and HF README
  text. `assemble_source_text` builds the full source text used to recheck
  claims that stay neutral.

## Usage

```python
from auto_benchmarkcard.config import Config, composer_serving_engine, get_llm_handler
from auto_benchmarkcard.tools.rag import MetadataIndexer, RAGRetriever, atomize_benchmark_card

documents = MetadataIndexer().create_documents(
    unitxt_data=None, hf_data=hf_json, benchmark_name="hellaswag", docling_data=docling_output
)
retriever = RAGRetriever(llm_handler=get_llm_handler(Config.COMPOSER_MODEL))
try:
    retriever.index_documents(documents)
    statements = atomize_benchmark_card(
        card, engine_type=composer_serving_engine(), model_name=Config.COMPOSER_MODEL
    )
    chunks = retriever.retrieve_for_statements_batch([s["text"] for s in statements])
finally:
    retriever.cleanup()
```

## Storage and configuration

Chroma runs in memory (`persist_directory=None`). Each retriever uses its own
collection, and `cleanup()` deletes it, so one benchmark never retrieves
another benchmark's chunks. The embedding model (`bge-large`, `e5-large` or
`minilm`) and the reranking, hybrid-search and query-expansion switches are
set in `auto_benchmarkcard.config.Config`.

## Output

The pipeline writes `formatted_rag_results_<benchmark>.jsonl` to
`<session>/tool_output/rag/`. The file holds one JSON object with the
`atoms` and `contexts` lists.
