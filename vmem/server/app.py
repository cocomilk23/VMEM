import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.logging_config import setup_logging
from vmem.memory.pipeline import MemoryPipeline
from vmem.retrieval.retriever import MemoryRetriever
from vmem.settings import build_config

app = FastAPI(title="ValueMEM")

setup_logging(os.getenv("VMEM_LOG_LEVEL", "INFO"))
config = build_config(os.getenv("VMEM_CONFIG_PATH"))
llm = LLMClient(config.llm)
embedder = OpenAIEmbedder(config.embedding)
graph = GraphStore(config.neo4j)
graph.ensure_schema()

pipeline = MemoryPipeline(
    llm=llm,
    embedder=embedder,
    graph=graph,
    value_threshold=config.retrieval.value_threshold,
    vector_index_name=config.retrieval.vector_index_name,
)
retriever = MemoryRetriever(graph=graph, embedder=embedder, llm=llm, config=config.retrieval)


class IngestRequest(BaseModel):
    text: Any
    source: str = "user"
    add_vector: bool = True
    add_graph: bool = False
    immediate: bool = False


class QueryRequest(BaseModel):
    text: Any


@app.post("/ingest")
def ingest(req: IngestRequest):
    result = pipeline.ingest_text(
        req.text,
        source=req.source,
        add_vector=req.add_vector,
        add_graph=req.add_graph,
        immediate=req.immediate,
    )
    if req.immediate:
        result.extend(
            pipeline.flush_buffer(
                source=req.source,
                add_vector=req.add_vector,
                add_graph=req.add_graph,
            )
        )
    total = len(result.vector_records) + len(result.graph_records)
    return {"count": total}


@app.post("/query")
def query(req: QueryRequest):
    from vmem.utils import extract_query_and_history

    query_text, history_lines = extract_query_and_history(req.text, max_turns=10)
    results = retriever.retrieve(query_text)
    memories = history_lines + [item["memory_text"] for item in results]
    answer = llm.answer_question(query_text, memories) if memories else "insufficient"
    return {"results": results, "answer": answer}
