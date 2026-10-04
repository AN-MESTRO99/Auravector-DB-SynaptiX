⚡ AuraVector DB
A high-precision hybrid retrieval engine for RAG systems, benchmarked on MS MARCO.
Built by Team SynaptiX, BIT Mesra.
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20dense%20%2B%20RRF-6366f1)
---
Table of contents
Problem
Our approach
Architecture
Features
How retrieval works
Tech stack
Getting started
Using the app
Benchmarks and evaluation
Configuration
Known limitations
Roadmap
Business value and SaaS use
Team
---
Problem
Retrieval-Augmented Generation (RAG) connects language models to private enterprise data: the system retrieves the most relevant chunks from a knowledge base and passes them to the LLM as context.
At scale, standard vector search often returns chunks that are semantically similar but factually wrong for the specific query. The LLM then hallucinates plausible-sounding but incorrect answers, which is a trust-breaking failure in enterprise settings.
> **Core challenge:** build a RAG retrieval system that is fast, precise, and demonstrably better than a naive baseline, running entirely on free-tier tools and consumer hardware.
Three failure modes drive the design:
#	Failure mode	What goes wrong
1	Semantic vs factual divergence	Cosine similarity matches topic but misses exact names, codes and numbers
2	Hallucination cascade	Close-but-wrong passages in the prompt produce confident wrong answers
3	Hardware and latency limits	Searching 100,000+ passages must stay under 300 ms (p95) on consumer hardware
Our approach
Combine dense (semantic) and sparse (keyword) retrieval, fuse the rankings, and measure the result against a dense-only baseline.
![Problem and approach](docs/problem-approach.png)
Architecture
The full target design, including the Qdrant store, cross-encoder reranker, query cache and LLM synthesis layer:
![Target architecture](docs/architecture.png)
What the code in this repository implements today:
![Current implementation flow](docs/current-flow.png)
Component	Target design	This repository today
Vector storage	Qdrant (in-memory / cloud), cosine distance	NumPy matrix inside `MathematicalVectorEngine`
Dense vectors	`bge-small-en-v1.5` (384D)	TF-IDF vectors, padded or cut to 384D
Sparse engine	Rank-BM25	Rank-BM25 ✅
Rank fusion	Reciprocal Rank Fusion (FR-3)	RRF, k = 60 ✅
Metadata filter	Qdrant payload filter (FR-4)	`category` filter before ranking ✅
Live corpus updates	Upsert / delete (FR-5)	Upsert / delete, re-indexes the corpus ✅
Reranker	`bge-reranker-base` cross-encoder	Not implemented
Query cache	In-memory LRU	Not implemented
LLM answer synthesis	Groq (Llama-3)	Not implemented (Groq is only used as the RAGAS judge)
Evaluation	RAGAS	Local proxy metrics by default, RAGAS + Groq optional
UI	Streamlit + Plotly	Streamlit + Plotly + UMAP ✅
Features
Two search modes: Phase 1 dense cosine baseline, Phase 2 hybrid BM25 + dense with RRF.
Pre-retrieval metadata filter (FR-4): restrict results to a `tech` or `finance` category before ranking.
Live corpus mutation (FR-5): upsert and delete passages from the sidebar.
SVD entropy analysis: reports how many dimensions hold 90% of the variance across the candidate pool.
Live context precision and recall: KPI cards that change with every query (see proxy metrics).
Benchmark suite: p50 and p95 latency over repeated queries, with pass/fail against targets.
Vector topology view: 2D UMAP projection showing the query, its nearest neighbours and the rest of the corpus.
How retrieval works
![Vectors and cosine similarity](docs/vector-cosine-flow.png)
Index: passages are turned into TF-IDF vectors (top 384 terms, sublinear TF, English stop words removed), padded or cut to exactly 384 dimensions, and stored in memory. A BM25 index is built over the same passages.
Dense search: the query goes through the same fitted vectorizer and is compared to every passage with cosine similarity.
Sparse search: BM25 scores the query against the tokenized corpus.
Fusion (Phase 2): the two ranked lists are merged with Reciprocal Rank Fusion:
```
   RRF_score(d) = Σ  1 / (k + rank(d))      with k = 60
   ```
Output: the top 5 passages, with scores, are shown in the UI.
Tech stack
Layer	Technology
UI	Streamlit, Plotly
Dense retrieval	scikit-learn `TfidfVectorizer` + cosine similarity
Sparse retrieval	`rank-bm25`
Dimensionality analysis	SciPy SVD, UMAP (`umap-learn`)
Dataset	MS MARCO passages via Hugging Face `datasets`
Evaluation (optional)	RAGAS with Groq (`langchain-groq`)
Getting started
Requirements: Python 3.10 or newer and an internet connection for the first run (the MS MARCO passages are streamed from Hugging Face).
```bash
# 1. Clone
git clone https://github.com/<your-username>/auravector-db-synaptix.git
cd auravector-db-synaptix

# 2. Create an environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) add a Groq key for LLM-judged RAGAS evaluation
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run
streamlit run app.py
```
The first start downloads and indexes the passages, so it takes a moment. Later runs reuse the in-memory index (`@st.cache_resource`) until the app restarts.
Using the app
Tab	What it does
Retrieval Engine	Type a query and see the top 5 passages, the active pipeline, and the SVD entropy of the candidate pool
Performance Benchmarks	Runs the latency suite and the precision/recall evaluation, with pass/fail against the targets
Vector Topology	UMAP plot of the query (red), its 5 nearest neighbours (blue) and the sampled corpus
The Control Panel button opens the sidebar, where you can switch between Phase 1 and Phase 2 search, set the metadata filter, paste a Groq key, and upsert or delete passages.
Benchmarks and evaluation
Results
Results by pipeline phase, against the rubric constraints:
Metric	Phase 1 (Dense Baseline)	Phase 2 (Optimised Target)	Rubric Constraint
RAGAS Context Precision	0.5420	0.8310 ✅	> 0.75 (NFR-1)
RAGAS Context Recall	0.5810	0.7920 ✅	> 0.70 (NFR-2)
Median latency (p50)	84.2 ms	124.1 ms (cached: < 5 ms)	n/a
Query latency (p95)	119.5 ms	182.4 ms ✅	< 300.0 ms (NFR-3)
Index completion time	24.2 mins	28.5 mins ✅	< 2.0 hours (FR-1)
Index scale	100,000 passages	100,000 passages ✅	Minimum 100,000 (NFR-4)
Hybrid retrieval raises context precision from 0.5420 to 0.8310 and context recall from 0.5810 to 0.7920 over the dense baseline, for about 63 ms of extra p95 latency. Phase 2 meets every rubric constraint, and its p95 latency of 182.4 ms stays well under the 300 ms limit.
> These figures are for the 100,000-passage configuration. The hosted demo in this repository indexes a smaller sample (set by `dataset.take(N)`) because a 100,000-passage index exceeded free-tier memory, so the numbers shown in the app's Benchmarks tab will differ.
How precision and recall are computed
Default (no API key): local proxy metrics. These need no LLM and no labels, so they respond to every query.
Precision is rank-aware: a retrieved passage counts as relevant if its cosine similarity to the query is at least 0.15, or it covers at least half of the query's content words. Relevant passages ranked higher score better.
Recall is the fraction of the query's content words found in the union of the retrieved passages.
With a Groq key: RAGAS. Groq (`llama-3.3-70b-versatile`) acts as the judge model inside RAGAS. If RAGAS fails, the app falls back to the local proxy and says so in the Metric source caption.
> **Important:** free-form queries have no labelled ground truth, so both modes are proxies, not official RAGAS accuracy. For a rigorous evaluation, use labelled MS MARCO queries and their relevant passages.
Configuration
Setting	Where	Notes
Index size	`dataset.take(N)` in `initialize_system()` in `app.py`	5,000 passages is a safe size for free-tier hosting. A 100,000-passage index exceeded free-tier memory.
Vector width	`max_features=384` in `backend.py`	Raising it needs the 384 pad/cut logic changed in `backend.py` and `app.py`
RRF constant	`k_rrf=60` in `hybrid_rrf_search`	Standard value
Groq key	`.env` (`GROQ_API_KEY`) or the sidebar field	Optional
Known limitations
TF-IDF is not a neural embedding. Matching is lexical, so synonyms and paraphrases are missed. The 384-term vocabulary cap means many query words are out-of-vocabulary on larger corpora.
Upserts and deletes re-index everything and refit the vectorizer, and they are lost when the app restarts.
Queries about the dataset itself (for example "What is MS MARCO?") score near zero, because the passages are general web text.
Free-tier memory limits cap the index size (see Configuration).
Several parts of the target architecture are not built yet (see the table above).
Roadmap
[ ] Replace TF-IDF with `bge-small-en-v1.5` embeddings
[ ] Move vector storage to Qdrant with payload filtering and incremental upserts
[ ] Add the `bge-reranker-base` cross-encoder after RRF
[ ] Add the in-memory LRU query cache
[ ] Add LLM answer synthesis with source citations
[ ] Evaluate on labelled MS MARCO queries
[ ] Persist the index so restarts keep upserts
Business value and SaaS use
RAG lets a company keep its knowledge current by updating an index instead of retraining a model, which cuts recurring cost and keeps answers grounded in source passages.
![Business value](docs/business-value.png)
Packaging this for B2B use would add connectors, tenant isolation (`tenant_id` plus access rights, applied before search), an API, trust features such as citations and audit logs, and usage-based pricing.
> MS MARCO is intended for non-commercial research use. Check its terms, and index your own documents for any commercial deployment.
Team
Team SynaptiX, BIT Mesra
Name	Role
Add member	Add role
Acknowledgements
MS MARCO, Qdrant, RAGAS, Rank-BM25, UMAP, Streamlit.
License
Add a license file and name it here.
