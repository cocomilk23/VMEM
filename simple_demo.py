# Minimal, readable end-to-end demo with timing/debug output.
# ssh -L 8280:localhost:8473 -L 8688:localhost:8688 jijingbo@server4
import argparse
import os
import time
import datetime
from dotenv import load_dotenv

from vmem.config import EmbeddingConfig, LLMConfig, Neo4jConfig, RetrievalConfig
from vmem.embeddings.local_embedder import LocalEmbedder
from vmem.embeddings.openai_embedder import OpenAIEmbedder
from vmem.graph.neo4j_store import GraphStore
from vmem.llm.client import LLMClient
from vmem.memory.pipeline import MemoryPipeline
from vmem.retrieval.retriever import MemoryRetriever
from vmem.utils import extract_query_and_history

SAMPLE_JSON = ""
SAMPLE_QUERY = "When did I met Ronaldo?"


class TimedGraph:
    def __init__(self, inner):
        self._inner = inner
        self.timings_ms = {"vector": 0.0, "profile": 0.0, "graph": 0.0}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def write_vector_memory(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.write_vector_memory(*args, **kwargs)
        finally:
            self.timings_ms["vector"] += (time.perf_counter() - t0) * 1000

    def write_profile_memory(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.write_profile_memory(*args, **kwargs)
        finally:
            self.timings_ms["profile"] += (time.perf_counter() - t0) * 1000

    def write_memory(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.write_memory(*args, **kwargs)
        finally:
            self.timings_ms["graph"] += (time.perf_counter() - t0) * 1000

def _build_config():
    load_dotenv()
    llm = LLMConfig(
        model=os.getenv("VMEM_LLM_MODEL", "gpt-5-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout_s=int(os.getenv("VMEM_LLM_TIMEOUT_S", "30")),
        max_retries=int(os.getenv("VMEM_LLM_MAX_RETRIES", "2")),
    )
    embedding = EmbeddingConfig(
        model=os.getenv("VMEM_EMBED_MODEL", "/home/jijingbo/models/all-MiniLM-L6-v2"),
        api_key=llm.api_key,
        base_url=llm.base_url,
    )
    neo4j = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:8688"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    retrieval = RetrievalConfig(
        value_threshold=float(os.getenv("VMEM_VALUE_THRESHOLD", "0.8")),
        top_k=int(os.getenv("VMEM_TOP_K", "8")),
        candidate_k=int(os.getenv("VMEM_CANDIDATE_K", "50")),
        min_high_value_hits=int(os.getenv("VMEM_MIN_HIGH_VALUE_HITS", "3")),
        expand_hops=int(os.getenv("VMEM_EXPAND_HOPS", "1")),
        expand_limit=int(os.getenv("VMEM_EXPAND_LIMIT", "20")),
        score_weight_value=0.6,
        score_weight_similarity=0.4,
        vector_index_name=os.getenv("VMEM_VECTOR_INDEX_NAME", "memory_embedding_index"),
        entity_fulltext_index=os.getenv("VMEM_ENTITY_FT_INDEX", "entity_name_idx"),
        answer_similarity_threshold=float(os.getenv("VMEM_ANSWER_SIM_THRESHOLD", "0.78")),
        retrieval_mode=os.getenv("VMEM_RETRIEVAL_MODE", "entity"),
    )
    return llm, embedding, neo4j, retrieval


def _require_config(llm: LLMConfig, neo4j: Neo4jConfig) -> None:
    missing = []
    if not llm.api_key:
        missing.append("OPENAI_API_KEY")
    if not neo4j.uri:
        missing.append("NEO4J_URI")
    if not neo4j.user:
        missing.append("NEO4J_USER")
    if not neo4j.password:
        missing.append("NEO4J_PASSWORD")
    if missing:
        raise SystemExit(f"Missing required config: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ValueMEM simple demo")
    parser.add_argument("--text", default=SAMPLE_JSON, help="JSON input or raw text")
    parser.add_argument("--query", default=SAMPLE_QUERY, help="Query text")
    parser.add_argument(
        "--mode",
        choices=["entity", "vector", "hybrid"],
        default="vector",
        help="Retrieval mode (default: entity)",
    )
    parser.add_argument(
        "--add-graph",
        action="store_true",
        help="Also update entity-relationship graph (default: vector only)",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Immediately extract memories from current buffer",
    )
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Do not flush remaining buffer at the end of the demo",
    )
    args = parser.parse_args()

    llm_cfg, emb_cfg, neo4j_cfg, retrieval_cfg = _build_config()
    retrieval_cfg = RetrievalConfig(
        **{**retrieval_cfg.__dict__, "retrieval_mode": args.mode}
    )
    _require_config(llm_cfg, neo4j_cfg)

    llm = LLMClient(llm_cfg)
    if os.path.exists(emb_cfg.model):
        embedder = LocalEmbedder(emb_cfg)
    else:
        embedder = OpenAIEmbedder(emb_cfg)
    graph = TimedGraph(GraphStore(neo4j_cfg))
    graph.ensure_schema()

    pipeline = MemoryPipeline(
        llm=llm,
        embedder=embedder,
        graph=graph,
        value_threshold=retrieval_cfg.value_threshold,
        vector_index_name=retrieval_cfg.vector_index_name,
    )
    retriever = MemoryRetriever(
        graph=graph,
        embedder=embedder,
        llm=llm,
        config=retrieval_cfg,
    )

    try:
        print("Ingesting...")
        t0 = time.perf_counter()
        result = pipeline.ingest_text(
            args.text,
            source="demo",
            add_vector=True,
            add_graph=True,
            immediate=True,
        )
        doc_payload = ""
        if doc_payload:
            doc_result = pipeline.ingest_document(
                doc_payload,
                source="document",
                chunk_tokens=2000,
                add_vector=True,
            )
            result.extend(doc_result)
        if not args.no_flush:
            result.extend(
                pipeline.flush_buffer(
                    source="demo",
                    add_vector=True,
                    add_graph=args.add_graph,
                )
            )
        t1 = time.perf_counter()
        ingest_ms = (t1 - t0) * 1000
        total_ingested = (
            len(result.vector_records) + len(result.graph_records) + len(result.profile_records)
        )
        print(f"Ingested {total_ingested} memories in {ingest_ms:.1f} ms")
        for record in result.vector_records:
            hv = "high" if record.high_value else "low"
            print(f"Vector({hv}): {record.memory_text} | value={record.value_score:.2f}")
        for record in result.profile_records:
            hv = "high" if record.high_value else "low"
            print(f"Profile({hv}): {record.memory_text} | value={record.value_score:.2f}")
        print(
            "Insert timings (ms): "
            f"vector={graph.timings_ms['vector']:.1f}, "
            f"profile={graph.timings_ms['profile']:.1f}, "
            f"graph={graph.timings_ms['graph']:.1f}"
        )

        # For graph output, only print triples + entity scores (no fact text / no rel score / no high/low).
        names: list[str] = []
        for record in result.graph_records:
            names.extend([record.subject, record.object])
        score_map = graph.get_entity_scores(list(dict.fromkeys(names)))
        for record in result.graph_records:
            subj_score = score_map.get(record.subject)
            obj_score = score_map.get(record.object)
            subj_str = "NA" if subj_score is None else f"{subj_score:.2f}"
            obj_str = "NA" if obj_score is None else f"{obj_score:.2f}"
            print(
                f"Graph: triple=({record.subject}, {record.predicate}, {record.object}) | "
                f"entity_scores=({record.subject}:{subj_str}, {record.object}:{obj_str})"
            )

        print("Querying ...")
        t4 = time.perf_counter()
        _, history_lines = extract_query_and_history(args.text, max_turns=0)
        query_text = args.query
        t2 = time.perf_counter()
        results = retriever.retrieve(query_text)
        t3 = time.perf_counter()
        retrieve_ms = (t3 - t2) * 1000
        # print(f"Retrieved {len(results)} high-value memories in {(t3 - t2) * 1000:.1f} ms")
        for item in results:
            if item.get("record_type") == "vector":
                print(
                    f'vector | {item["memory_text"]} | '
                    f'value={item["value_score"]:.2f} sim={item.get("similarity", 0.0):.2f}'
                )
            else:
                print(
                    f'Graph | {item["subject"]} - {item["predicate"]} - {item["object"]} | '
                    f'value={item["value_score"]:.2f} sim={item.get("similarity", 0.0):.2f}'
                )

        print(f"Retrieve time: {retrieve_ms:.1f} ms")
        memories = history_lines + [item["memory_text"] for item in results]
        t_ans0 = time.perf_counter()
        answer = llm.answer_question(query_text, memories)
        t_ans1 = time.perf_counter()
        answer_ms = (t_ans1 - t_ans0) * 1000
        if args.mode == "vector" and answer != "insufficient":
            print(f"Debug: total retrieve ms={(t3 - t2) * 1000:.1f}")
        t5 = time.perf_counter()
        print(f"Answer: {answer}(resume {(t5 - t4) * 1000:.1f} ms)")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
