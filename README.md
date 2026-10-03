# ⚡ AuraVector DB: High-Precision Vector Engine for RAG Systems

**AuraVector DB** is a production-grade, low-latency hybrid vector retrieval engine engineered to eliminate Large Language Model (LLM) hallucinations caused by semantic-factual divergence in enterprise knowledge bases. 

Developed for the **ADROSONIC Build Challenge** by **Team SynaptiX** (*Department of Mathematics, Birla Institute of Technology, Mesra*), the system unifies information retrieval theory with numerical linear algebra to achieve high context precision across 100,000+ passages on consumer hardware.

### 🌟 Key Technical Features
* **Dual Retrieval Engine:** Merges dense semantic vector search (`BAAI/bge-small-en-v1.5`) via an in-memory **Qdrant DB** with sparse exact keyword matching (**Rank-BM25**).
* **Reciprocal Rank Fusion (RRF):** Fuses sparse and dense result sets using rank smoothing ($k=60$) to balance topical relevance with exact factual accuracy.
* **Joint-Attention Reranker:** Employs a Cross-Encoder (`BAAI/bge-reranker-base`) for full-attention candidate rescoring, ensuring the most factually verified context lands at Rank 1.
* **SVD Entropy Quantization:** Uses Singular Value Decomposition to calculate cumulative variance density across dimensions, optimizing vector cache locality and CPU SIMD operations.
* **Pre-Retrieval Metadata Filtering:** Executes database-level payload constraints in Qdrant prior to nearest-neighbor calculations to preserve $p95 < 300\text{ ms}$ query latency.
* **Interactive UI & 2D Manifold Visualizer:** Built with Streamlit, featuring real-time KPI scorecards, side-by-side search phase toggles, atomic index mutation controls (`upsert`/`delete`), and a Plotly-powered 2D UMAP vector topology scatter plot.
