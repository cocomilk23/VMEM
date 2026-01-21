import os

from fastapi import FastAPI
from pydantic import BaseModel

from vmem.cache import ValueCache
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
cache = ValueCache(config.cache)
graph.ensure_schema()

pipeline = MemoryPipeline(llm=llm, embedder=embedder, graph=graph, cache=cache)
retriever = MemoryRetriever(graph=graph, embedder=embedder, config=config.retrieval)


class IngestRequest(BaseModel):
    text: str
    source: str = "user"


class QueryRequest(BaseModel):
    text: str


@app.post("/ingest")
def ingest(req: IngestRequest):
    records = pipeline.ingest_text(req.text, source=req.source)
    return {"count": len(records)}


@app.post("/query")
def query(req: QueryRequest):
    results = retriever.retrieve(req.text)
    memories = [item["memory_text"] for item in results]
    answer = llm.answer_question(req.text, memories) if memories else "insufficient"
    return {"results": results, "answer": answer}
