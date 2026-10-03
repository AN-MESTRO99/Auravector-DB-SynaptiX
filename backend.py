import time
import string
import os
import numpy as np
from scipy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantVectorEngine:
    """Vector storage and retrieval engine powered by Qdrant."""
    
    def __init__(self, collection_name: str = "ms_marco_collection"):
        # Uses Qdrant in-memory engine.
        # For Qdrant Cloud, replace with: QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=384,
            sublinear_tf=True
        )
        self.doc_passages = []
        self.doc_categories = []
        self.doc_ids = []
        self.doc_vectors = None
        self.bm25_engine = None

    def _pad_vector(self, vec: np.ndarray) -> np.ndarray:
        """Pads TF-IDF vectors to exactly 384 dimensions."""
        if vec.shape[1] < 384:
            padding = np.zeros((vec.shape[0], 384 - vec.shape[1]))
            return np.hstack([vec, padding])
        return vec[:, :384]

    def ingest_and_index(self, passage_dicts):
        """Phase 1-3 Data Ingestion: Standardizes documents into Qdrant PointStructs."""
        self.doc_passages = [p["text"] for p in passage_dicts]
        self.doc_categories = [p.get("category", "tech") for p in passage_dicts]
        self.doc_ids = [p.get("id", f"doc_{i}") for i, p in enumerate(passage_dicts)]

        # Generate vectors
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_passages).toarray()
        self.doc_vectors = self._pad_vector(tfidf_matrix)

        # Step 1: Re-create Collection schema in Qdrant
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Step 2 & 3: Format PointStructs & Ingest via Upsert
        points = [
            PointStruct(
                id=idx,  # Integer point ID
                vector=self.doc_vectors[idx].tolist(),
                payload={
                    "doc_id": self.doc_ids[idx],
                    "text": self.doc_passages[idx],
                    "category": self.doc_categories[idx]
                }
            )
            for idx in range(len(passage_dicts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

        # BM25 Lexical Engine initialization
        tokenized_corpus = [doc.lower().split() for doc in self.doc_passages]
        self.bm25_engine = BM25Okapi(tokenized_corpus)

        return self.svd_entropy_analysis(self.doc_vectors)

    def svd_entropy_analysis(self, embeddings: np.ndarray) -> int:
        if embeddings.shape[0] < 2:
            return 1
        centered = embeddings - np.mean(embeddings, axis=0)
        _, s, _ = svd(centered, full_matrices=False)
        total_var = np.sum(s)
        if total_var < 1e-9:
            return 1
        normalized_variance = s / total_var
        cumulative_variance = np.cumsum(normalized_variance)
        critical_dims = np.where(cumulative_variance >= 0.90)[0]
        return int(critical_dims[0] + 1) if len(critical_dims) > 0 else len(s)

    def upsert_passage(self, doc_id: str, text: str, category: str = "tech"):
        """Live mutation helper to append or update a passage in Qdrant."""
        all_dicts = [
            {"id": self.doc_ids[i], "text": self.doc_passages[i], "category": self.doc_categories[i]}
            for i in range(len(self.doc_passages))
            if self.doc_ids[i] != doc_id
        ]
        all_dicts.append({"id": doc_id, "text": text, "category": category})
        self.ingest_and_index(all_dicts)

    def delete_passage(self, doc_id: str) -> bool:
        """Deletes a passage by ID and re-indexes the corpus."""
        if doc_id not in self.doc_ids:
            return False

        remaining_dicts = [
            {"id": self.doc_ids[i], "text": self.doc_passages[i], "category": self.doc_categories[i]}
            for i in range(len(self.doc_passages))
            if self.doc_ids[i] != doc_id
        ]
        
        if len(remaining_dicts) > 0:
            self.ingest_and_index(remaining_dicts)
        else:
            self.doc_passages = []
            self.doc_categories = []
            self.doc_ids = []
            self.doc_vectors = np.empty((0, 384))
        return True


class HybridRetrievalPipeline:
    def __init__(self, engine: QdrantVectorEngine):
        self.engine = engine

    def dense_search(self, query: str, top_k: int = 5, category_filter: str = None):
        """Qdrant Dense Vector Similarity Search."""
        if len(self.engine.doc_passages) == 0:
            return []

        query_vec = self.engine.vectorizer.transform([query]).toarray()
        query_vec = self.engine._pad_vector(query_vec)[0].tolist()

        # Query Qdrant Collection
        search_results = self.engine.client.search(
            collection_name=self.engine.collection_name,
            query_vector=query_vec,
            limit=top_k * 3 if category_filter else top_k
        )

        results = []
        for hit in search_results:
            payload = hit.payload
            if category_filter and payload["category"] != category_filter:
                continue
            results.append((payload["text"], float(hit.score), hit.id))
            if len(results) == top_k:
                break
        return results

    def sparse_search(self, query: str, top_k: int = 15):
        """BM25 Lexical Keyword Search."""
        if len(self.engine.doc_passages) == 0:
            return []

        tokenized_query = query.lower().split()
        scores = self.engine.bm25_engine.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(scores):
            results.append((self.engine.doc_passages[idx], float(score), idx))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def hybrid_rrf_search(self, query: str, top_k: int = 5, category_filter: str = None, k_rrf: int = 60):
        """Reciprocal Rank Fusion combining Qdrant Dense + BM25 Sparse."""
        dense_res = self.dense_search(query, top_k=30, category_filter=category_filter)
        sparse_res = self.sparse_search(query, top_k=30)
        
        rrf_scores = {}
        for rank, item in enumerate(dense_res):
            doc_idx = item[2]
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        for rank, item in enumerate(sparse_res):
            doc_idx = item[2]
            if category_filter and self.engine.doc_categories[doc_idx] != category_filter:
                continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_rrf + rank + 1))

        fused_results = [
            (self.engine.doc_passages[idx], score, idx)
            for idx, score in rrf_scores.items()
        ]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:top_k]


def run_latency_benchmark(pipeline: HybridRetrievalPipeline, num_queries: int = 20):
    """Measures median (p50) and tail (p95) latency over consecutive queries."""
    sample_queries = [
        "BM25 term frequency", "Singular Value Decomposition SVD", 
        "Reciprocal Rank Fusion RAG", "Cross-Encoder reranking precision",
        "Pre-retrieval metadata payload filtering", "Dense bi-encoder vectors"
    ]
    latencies = []
    for i in range(num_queries):
        q = sample_queries[i % len(sample_queries)]
        start = time.perf_counter()
        _ = pipeline.hybrid_rrf_search(q, top_k=5)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        
    return np.percentile(latencies, 50), np.percentile(latencies, 95)


def run_ragas_eval(
    pipeline: HybridRetrievalPipeline,
    query: str,
    api_key: str = None,
    mode: str = "hybrid"
):
    """If a Groq API Key is supplied or found in .env, overrides local evaluation with Groq + RAGAS."""
    effective_groq_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()

    is_hybrid = not ("phase 1" in mode.lower() or mode.lower() == "dense")
    def _search(q: str, top_k: int):
        return pipeline.hybrid_rrf_search(q, top_k=top_k) if is_hybrid else pipeline.dense_search(q, top_k=top_k)

    # Key Override Path: Invoke Groq LLM
    if effective_groq_key:
        try:
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall
            from langchain_groq import ChatGroq
            from datasets import Dataset

            eval_llm = ChatGroq(
                temperature=0,
                groq_api_key=effective_groq_key,
                model_name="llama-3.1-70b-versatile"
            )

            retrieved = _search(query, top_k=5)
            contexts = [[item[0] for item in retrieved]]

            dataset = Dataset.from_dict({
                "question": [query],
                "contexts": contexts,
                "ground_truth": [query]
            })

            result = evaluate(
                dataset=dataset,
                metrics=[context_precision, context_recall],
                llm=eval_llm
            )

            cp = result.get("context_precision")
            cr = result.get("context_recall")

            if cp is not None and cr is not None:
                return {
                    "precision": float(cp),
                    "recall": float(cr),
                    "provider": "Groq LLM (RAGAS)"
                }
        except Exception:
            pass  # Fall back on failure

    # Local Fallback Math Computation
    return _compute_local_fallback_metrics(pipeline, query, is_hybrid, _search)


def _compute_local_fallback_metrics(pipeline, query, is_hybrid, search_fn):
    retrieved = search_fn(query, top_k=5)
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "provider": "Local Math Engine"}

    words = set(query.lower().split())
    matches = 0
    for passage, _, _ in retrieved:
        p_words = set(passage.lower().split())
        if words.intersection(p_words):
            matches += 1

    prec = round(matches / len(retrieved), 4)
    rec = round(min(1.0, matches / max(1, len(words))), 4)
    return {"precision": prec, "recall": rec, "provider": "Local Math Engine"}
