import time
import os
import numpy as np
from scipy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class MathematicalVectorEngine:
    def __init__(self):
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

    def ingest_and_index(self, passage_dicts):
        """Indexes passages using TF-IDF spatial vectors and BM25 lexicals."""
        self.doc_passages = [p["text"] for p in passage_dicts]
        self.doc_categories = [p.get("category", "tech") for p in passage_dicts]
        self.doc_ids = [p.get("id", f"doc_{i}") for i, p in enumerate(passage_dicts)]

        # Dense TF-IDF Matrix (acting as high-dimensional embedding space)
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_passages).toarray()
        
        # Pad to exactly 384 dimensions if vocabulary is smaller
        if tfidf_matrix.shape[1] < 384:
            padding = np.zeros((tfidf_matrix.shape[0], 384 - tfidf_matrix.shape[1]))
            self.doc_vectors = np.hstack([tfidf_matrix, padding])
        else:
            self.doc_vectors = tfidf_matrix[:, :384]

        # BM25 Lexical Engine
        tokenized_corpus = [doc.lower().split() for doc in self.doc_passages]
        self.bm25_engine = BM25Okapi(tokenized_corpus)

        return self.svd_entropy_analysis(self.doc_vectors)

    def svd_entropy_analysis(self, embeddings: np.ndarray) -> int:
        """Computes SVD cumulative variance entropy on context/candidate vectors."""
        if embeddings.shape[0] < 2:
            return 1
        
        # Mean-center vectors to isolate direction variance
        centered = embeddings - np.mean(embeddings, axis=0)
        _, s, _ = svd(centered, full_matrices=False)
        
        total_var = np.sum(s)
        if total_var < 1e-9:
            return 1
            
        normalized_variance = s / total_var
        cumulative_variance = np.cumsum(normalized_variance)
        
        # Minimum dimensions holding 90% cumulative variance
        critical_dims = np.where(cumulative_variance >= 0.90)[0]
        return int(critical_dims[0] + 1) if len(critical_dims) > 0 else len(s)

    def upsert_passage(self, doc_id: str, text: str, category: str = "tech"):
        """Live mutation helper (FR-5) to append or update a passage in the index."""
        all_dicts = [
            {"id": self.doc_ids[i], "text": self.doc_passages[i], "category": self.doc_categories[i]}
            for i in range(len(self.doc_passages))
            if self.doc_ids[i] != doc_id
        ]
        all_dicts.append({"id": doc_id, "text": text, "category": category})
        self.ingest_and_index(all_dicts)

    def delete_passage(self, doc_id: str) -> bool:
        """Deletes a passage by ID and re-indexes the remaining corpus."""
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
    def __init__(self, engine: MathematicalVectorEngine):
        self.engine = engine

    def dense_search(self, query: str, top_k: int = 5, category_filter: str = None):
        """Phase 1: Dense Vector Similarity Search (Cosine)."""
        if len(self.engine.doc_passages) == 0:
            return []

        query_vec = self.engine.vectorizer.transform([query]).toarray()
        if query_vec.shape[1] < 384:
            padding = np.zeros((1, 384 - query_vec.shape[1]))
            query_vec = np.hstack([query_vec, padding])
        else:
            query_vec = query_vec[:, :384]

        scores = cosine_similarity(query_vec, self.engine.doc_vectors)[0]
        
        results = []
        for idx, score in enumerate(scores):
            if category_filter and self.engine.doc_categories[idx] != category_filter:
                continue
            results.append((self.engine.doc_passages[idx], float(score), idx))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

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
        """Phase 2: Reciprocal Rank Fusion (Dense + Sparse BM25) + Reranking."""
        dense_res = self.dense_search(query, top_k=30, category_filter=category_filter)
        sparse_res = self.sparse_search(query, top_k=30)
        
        rrf_scores = {}
        
        # Accumulate Dense Ranks
        for rank, item in enumerate(dense_res):
            doc_idx = item[2]
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_rrf + rank + 1))
            
        # Accumulate Sparse BM25 Ranks
        for rank, item in enumerate(sparse_res):
            doc_idx = item[2]
            if category_filter and self.engine.doc_categories[doc_idx] != category_filter:
                continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_rrf + rank + 1))

        # Sort combined candidate space by unified RRF score
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
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    return p50, p95


def _content_terms(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in ENGLISH_STOP_WORDS and len(t) > 1}


def compute_local_context_metrics(pipeline, query, top_k=5, category_filter=None,
                                  mode="hybrid", cos_threshold=0.15, coverage_threshold=0.5):
    """
    Query-dependent proxy metrics (no LLM, no ground-truth labels needed).

    - Precision: rank-aware context precision (RAGAS-style average precision@k).
      A retrieved passage counts as relevant if its cosine similarity to the query
      >= cos_threshold, or it covers >= coverage_threshold of the query's content terms.
    - Recall: fraction of the query's content terms found in the union of the
      retrieved passages.
    """
    if mode == "dense":
        retrieved = pipeline.dense_search(query, top_k=top_k, category_filter=category_filter)
    else:
        retrieved = pipeline.hybrid_rrf_search(query, top_k=top_k, category_filter=category_filter)

    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "source": "local proxy"}

    q_terms = _content_terms(query)
    n_docs = len(pipeline.engine.doc_passages)
    cos_by_idx = {idx: s for _, s, idx in pipeline.dense_search(query, top_k=n_docs)}

    relevance, covered = [], set()
    for text, _score, idx in retrieved:
        p_terms = _content_terms(text)
        covered |= (q_terms & p_terms)
        coverage = len(q_terms & p_terms) / len(q_terms) if q_terms else 0.0
        relevance.append(cos_by_idx.get(idx, 0.0) >= cos_threshold or coverage >= coverage_threshold)

    hits, ap = 0, 0.0
    for k, is_rel in enumerate(relevance, start=1):
        if is_rel:
            hits += 1
            ap += hits / k
    precision = ap / hits if hits else 0.0
    recall = len(covered) / len(q_terms) if q_terms else 0.0

    return {"precision": precision, "recall": recall, "source": "local proxy"}


def run_ragas_eval(pipeline, query, api_key=None, category_filter=None, mode="hybrid"):
    """Uses RAGAS + Groq when a key is given; otherwise (or on failure) uses local metrics."""
    local = compute_local_context_metrics(pipeline, query, category_filter=category_filter, mode=mode)
    if not api_key:
        return local

    try:
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
        from langchain_groq import ChatGroq
        from datasets import Dataset

        eval_llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

        retrieved = pipeline.hybrid_rrf_search(query, top_k=5, category_filter=category_filter)
        contexts = [[item[0] for item in retrieved]]

        # No labeled ground truth exists for free-form queries, so the top-ranked
        # passage is used as a reference. This is a weak proxy, not a true gold label.
        dataset = Dataset.from_dict({
            "question": [query],
            "contexts": contexts,
            "ground_truth": [retrieved[0][0] if retrieved else ""],
        })

        result = evaluate(dataset=dataset, metrics=[context_precision, context_recall], llm=eval_llm)
        df = result.to_pandas()
        return {
            "precision": float(df["context_precision"].mean()),
            "recall": float(df["context_recall"].mean()),
            "source": "RAGAS (Groq)",
        }
    except Exception as e:
        local["source"] = f"local proxy (RAGAS failed: {type(e).__name__})"
        return local
