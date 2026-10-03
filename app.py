import os
import time
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import umap
import plotly.express as px
from datasets import load_dataset
from groq import Groq
from backend import (MathematicalVectorEngine, HybridRetrievalPipeline,
                     run_latency_benchmark, run_ragas_eval, compute_local_context_metrics)

load_dotenv()

st.set_page_config(
    page_title="AuraVector DB | High-Precision Vector Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1250px; }
    .app-title {
        font-weight: 900; font-size: 2.3rem; line-height: 1.2;
        background: linear-gradient(90deg, #4F46E5 0%, #0284C7 50%, #059669 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-subtitle { color: #64748B; font-size: 0.88rem; margin-top: 4px; font-weight: 500; }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px; text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #059669; }
    .metric-label { font-size: 0.72rem; color: #64748B; text-transform: uppercase; font-weight: 700; margin-bottom: 6px; }
    .passage-card {
        background: #FFFFFF; border-radius: 12px; padding: 20px;
        border-left: 5px solid #4F46E5; border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;
    }
    .passage-card-hybrid { border-left-color: #059669; }
    .score-badge {
        background-color: #EFF6FF; color: #0284C7; border: 1px solid #BAE6FD;
        font-size: 0.75rem; font-weight: 700; padding: 4px 14px; border-radius: 20px; display: inline-block; margin-bottom: 12px;
    }
    .cache-badge {
        background-color: #ECFDF5; color: #059669; border: 1px solid #A7F3D0;
        font-size: 0.75rem; font-weight: 700; padding: 4px 14px; border-radius: 20px; display: inline-block; margin-bottom: 12px; margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    # Load sample corpus from MS MARCO
    dataset = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
    sample_passages = []
    for i, row in enumerate(dataset.take(1000)):
        official_id = row.get("_id") or row.get("passage_id")
        official_text = row.get("text") or row.get("passage")
        if not official_text:
            continue
        sample_passages.append({
            "id": str(official_id),
            "text": official_text,
            "category": "tech" if i % 2 == 0 else "finance"
        })

    engine = MathematicalVectorEngine()
    svd_dims = engine.ingest_and_index(sample_passages)
    pipeline = HybridRetrievalPipeline(engine)
    return engine, pipeline, svd_dims

engine, pipeline, svd_dims = initialize_system()

@st.cache_data(show_spinner=False)
def compute_umap_projection(query_vec: np.ndarray, corpus_vecs: np.ndarray):
    all_vecs = np.vstack([query_vec, corpus_vecs])
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    projected = reducer.fit_transform(all_vecs)
    return projected[0], projected[1:]

def trigger_sidebar_toggle():
    components.html("""
    <script>
        const doc = window.parent.document;
        const btn = doc.querySelector('button[data-testid="stSidebarCollapseButton"]') 
                 || doc.querySelector('button[aria-label="Expand sidebar"]')
                 || doc.querySelector('button[aria-label="Close sidebar"]')
                 || doc.querySelector('section[data-testid="stSidebar"] button');
        if (btn) { btn.click(); }
    </script>
    """, height=0, width=0)

head_col1, head_col2 = st.columns([3.2, 1])
with head_col1:
    st.markdown("""
        <div>
            <div class="app-title">⚡ AuraVector DB</div>
            <div class="app-subtitle">High-Precision Vector Engine for RAG Systems • Team SynaptiX • BIT Mesra</div>
        </div>
    """, unsafe_allow_html=True)

with head_col2:
    st.write("")
    if st.button("🎛️ Control Panel", type="primary", use_container_width=True):
        trigger_sidebar_toggle()

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("🎛️ Engine Control Panel")
    if st.button("❌ Close Control Panel", use_container_width=True):
        trigger_sidebar_toggle()

    st.markdown("---")
    retrieval_mode = st.radio(
        "Search Pipeline Mode:",
        ["Phase 1: Dense Vector Search (Baseline)", "Phase 2: Hybrid RRF Search + Reranker"]
    )

    st.markdown("---")
    category_filter = st.selectbox("Pre-Retrieval Metadata Filter (FR-4):", [None, "tech", "finance"])

    st.markdown("---")
    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = os.getenv("GROQ_API_KEY", "")

    groq_key = st.text_input("Groq API Key", type="password", key="groq_api_key")

    st.markdown("---")
    st.subheader("Live Corpus Mutation (FR-5)")
    mutation_tab_upsert, mutation_tab_delete = st.tabs(["➕ Upsert", "🗑 Delete"])

    with mutation_tab_upsert:
        new_id = st.text_input("Doc ID", "ms_marco_999", key="upsert_id_input")
        new_text = st.text_area("Passage Text", "MS MARCO Passage #999: SVD dimensional entropy quantization accelerates retrieval.", key="upsert_text_input")
        if st.button("➕ Upsert Passage", use_container_width=True):
            engine.upsert_passage(new_id, new_text)
            st.toast(f"Upserted document `{new_id}`!", icon="⚡")

    with mutation_tab_delete:
        del_id = st.text_input("Doc ID to Delete", "ms_marco_999", key="delete_id_input")
        if st.button("🗑️ Delete Passage", type="primary", use_container_width=True):
            if engine.delete_passage(del_id):
                st.toast(f"Deleted `{del_id}` from corpus!", icon="🗑️")
            else:
                st.error(f"Document `{del_id}` not found.")

# Top Scoreboard KPIs
kpi_query = st.session_state.get("live_query_input", "What is MS MARCO passage ranking?")
mode_key = "dense" if "Phase 1" in retrieval_mode else "hybrid"
live = compute_local_context_metrics(pipeline, kpi_query, mode=mode_key, category_filter=category_filter)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Indexed Scale</div><div class="metric-value">{len(engine.doc_passages):,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="metric-card"><div class="metric-label">Target p95 Latency</div><div class="metric-value">&lt; 300 ms</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">RAGAS Context Precision</div><div class="metric-value">{live["precision"]:.3f}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">RAGAS Context Recall</div><div class="metric-value">{live["recall"]:.3f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_search, tab_benchmark, tab_umap = st.tabs(["🚀 Retrieval Engine", "⚡ Performance Benchmarks", "📊 Vector Topology"])

with tab_search:
    query = st.text_input("🔍 Search MS MARCO Corpus", value="What is MS MARCO passage ranking?", key="live_query_input")

    if query:
        start_time = time.perf_counter()
        cache_hit = False

        if "Phase 1" in retrieval_mode:
            results = pipeline.dense_search(query, top_k=5, category_filter=category_filter)
            candidates = pipeline.dense_search(query, top_k=30, category_filter=category_filter)
            card_class = "passage-card"
        else:
            results, cache_hit = pipeline.hybrid_rrf_search(query, top_k=5, category_filter=category_filter, use_reranker=True)
            dense_cand = pipeline.dense_search(query, top_k=15, category_filter=category_filter)
            sparse_cand = pipeline.sparse_search(query, top_k=15)
            candidates = dense_cand + sparse_cand
            card_class = "passage-card passage-card-hybrid"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if candidates:
            cand_indices = list(set([res[2] for res in candidates]))
            cand_vecs = engine.doc_vectors[cand_indices]
            local_svd_dims = engine.svd_entropy_analysis(cand_vecs)
        else:
            local_svd_dims = svd_dims

        st.markdown("<br>", unsafe_allow_html=True)
        if not results:
            st.warning("No passages found matching criteria.")
        else:
            col_res, col_info = st.columns([1.4, 0.6])
            with col_res:
                st.markdown("##### Top Relevant Passages")
                for idx, res in enumerate(results):
                    text, score = res[0], res[1]
                    cache_badge_html = '<div class="cache-badge">⚡ Sub-5ms LRU Cache Hit</div>' if cache_hit else ""
                    st.markdown(f"""
                        <div class="{card_class}">
                            <div class="score-badge">Rank #{idx+1} • Score: {score:.4f}</div>{cache_badge_html}
                            <div style="color: #334155; font-size: 0.95rem; line-height: 1.5;">{text}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # LLM Answer Card with Citations
                if groq_key:
                    st.markdown("---")
                    st.markdown("##### 🤖 LLM Answer Card (Groq Llama-3)")
                    if st.button("Generate Natural Language Answer"):
                        try:
                            client = Groq(api_key=groq_key)
                            context_str = "\n\n".join([f"Passage [{i+1}]: {r[0]}" for i, r in enumerate(results)])
                            prompt = (
                                "You are an enterprise knowledge assistant. Answer the user query using strictly "
                                "the provided context passages. Cite source passage numbers in brackets, e.g., [1].\n\n"
                                f"Context:\n{context_str}\n\nQuery: {query}"
                            )
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0
                            )
                            st.success(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Groq Answer Generation Error: {str(e)}")

            with col_info:
                st.markdown("##### 🛠 Execution Metadata")
                st.info(f"**Pipeline Mode:**\n\n{retrieval_mode}")
                st.metric("Query Response Time", f"{elapsed_ms:.2f} ms")
                st.success(f"**SVD Entropy Concentration:**\n\n90% Entropy in top **{local_svd_dims} / 384** dimensions.")
                if category_filter:
                    st.warning(f"**Payload Filter:** `category == '{category_filter}'`")

with tab_benchmark:
    st.markdown("##### NFR Latency & Evaluation Suite")
    if st.button("▶ Run Full System Benchmark Suite", type="primary"):
        with st.spinner("Running latency and precision evaluation..."):
            bench_query = query if query else "What is MS MARCO passage ranking?"
            p50, p95 = run_latency_benchmark(pipeline)
            eval_res = run_ragas_eval(pipeline, bench_query, groq_key, category_filter=category_filter, mode=mode_key)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median Latency (p50)", f"{p50:.2f} ms")
            m2.metric("Tail Latency (p95)", f"{p95:.2f} ms", delta="PASSED (<300ms)" if p95 < 300 else "FAILED")
            m3.metric("RAGAS Precision", f"{eval_res['precision']:.4f}", delta="PASSED (>0.75)" if eval_res['precision'] > 0.75 else "BELOW TARGET")
            m4.metric("RAGAS Recall", f"{eval_res['recall']:.4f}", delta="PASSED (>0.70)" if eval_res['recall'] > 0.70 else "BELOW TARGET")
            st.caption(f"Evaluation Engine Source: {eval_res['source']}")

with tab_umap:
    st.markdown("##### 2D UMAP Vector Space Topology Explorer")
    active_query = query if query else "What is MS MARCO passage ranking?"
    if len(engine.doc_passages) > 0:
        from sklearn.metrics.pairwise import cosine_distances

        query_vec = engine.dense_model.encode([active_query], normalize_embeddings=True)
        corpus_limit = min(100, len(engine.doc_passages))
        corpus_vecs = engine.doc_vectors[:corpus_limit]
        
        dists = cosine_distances(query_vec, corpus_vecs)[0]
        sims = (1.0 - dists) * 100.0
        nearest_indices = set(np.argsort(dists)[:5])

        q_coords, doc_coords = compute_umap_projection(query_vec, corpus_vecs)

        categories, hover_texts, sizes = [], [], []
        for idx in range(corpus_limit):
            doc_id = engine.doc_ids[idx]
            sim_score = sims[idx]
            text_snippet = engine.doc_passages[idx][:80] + "..."
            if idx in nearest_indices:
                categories.append("Nearest Neighbor")
                sizes.append(10)
            else:
                categories.append("Unselected Corpus")
                sizes.append(5)
            hover_texts.append(f"<b>Doc ID:</b> {doc_id}<br><b>Similarity:</b> {sim_score:.2f}%<br><b>Passage:</b> {text_snippet}")

        x_pts = [q_coords[0]] + list(doc_coords[:, 0])
        y_pts = [q_coords[1]] + list(doc_coords[:, 1])

        fig = px.scatter(
            x=x_pts, y=y_pts,
            color=["Query Vector"] + categories,
            size=[14] + sizes,
            hover_name=[f"<b>Query:</b> {active_query}"] + hover_texts,
            color_discrete_map={"Query Vector": "#EF4444", "Nearest Neighbor": "#2563EB", "Unselected Corpus": "#CBD5E1"},
            template="plotly_white"
        )
        for idx in nearest_indices:
            fig.add_shape(
                type="line", x0=q_coords[0], y0=q_coords[1],
                x1=doc_coords[idx, 0], y1=doc_coords[idx, 1],
                line=dict(color="#3B82F6", width=1.5, dash="dash"), layer="below"
            )
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
