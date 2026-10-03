import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import umap
import plotly.express as px
from backend import QdrantVectorEngine, HybridRetrievalPipeline, run_latency_benchmark, run_ragas_eval

# Load environment variables (.env)
load_dotenv()

st.set_page_config(
    page_title="AuraVector DB | Qdrant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def initialize_system():
    base_domains = [
        "Singular Value Decomposition (SVD) performs dimensionality reduction.",
        "BM25 keyword search calculates term frequency for sparse lexical ranking.",
        "Reciprocal Rank Fusion (RRF) unifies sparse keyword search and dense similarity scores.",
        "Pre-retrieval metadata filtering executes structural constraints in Qdrant.",
        "Dense vector embeddings compress text into high-dimensional spatial representations."
    ]

    sample_passages = []
    for i in range(100):
        domain_text = base_domains[i % len(base_domains)]
        sample_passages.append({
            "id": f"ms_marco_{i}",
            "text": f"MS MARCO Passage #{i}: {domain_text} [variant-{i}]",
            "category": "tech" if i % 2 == 0 else "finance"
        })

    # Instantiate Qdrant Engine
    engine = QdrantVectorEngine()
    svd_dims = engine.ingest_and_index(sample_passages)
    pipeline = HybridRetrievalPipeline(engine)
    return engine, pipeline, svd_dims

engine, pipeline, svd_dims = initialize_system()

# Sidebar Setup
with st.sidebar:
    st.subheader("🎛️ Engine Control Panel")
    
    retrieval_mode = st.radio(
        "Search Pipeline Mode:",
        ["Phase 1: Dense Vector Search (Baseline)", "Phase 2: Hybrid RRF Search + Reranker"]
    )

    category_filter = st.selectbox("Pre-Retrieval Metadata Filter:", [None, "tech", "finance"])

    st.markdown("---")
    st.subheader("🔑 API Key Override Configuration")
    
    # Read key from .env or manual user entry
    env_groq_key = os.getenv("GROQ_API_KEY", "")
    groq_key = st.text_input(
        "Groq API Key Override", 
        type="password",
        value=st.session_state.get("groq_api_key", env_groq_key),
        key="groq_api_key_input",
        help="Paste a key to override local evaluation metrics with Groq LLM RAGAS evaluation."
    )

    if groq_key.strip():
        st.success("⚡ Groq API Key Active: LLM Override Enabled")
    else:
        st.info("ℹ️ No Groq Key: Falling back to local precision/recall scoring")

    st.markdown("---")
    st.subheader("Live Corpus Mutation")

    mutation_tab_upsert, mutation_tab_delete = st.tabs(["➕ Upsert", "🗑 Delete"])

    with mutation_tab_upsert:
        new_id = st.text_input("Doc ID", "ms_marco_999", key="upsert_id")
        new_text = st.text_area("Passage Text", "MS MARCO Passage #999: SVD dimensional entropy.", key="upsert_text")
        if st.button("➕ Upsert Passage", use_container_width=True):
            engine.upsert_passage(new_id, new_text)
            st.toast(f"Upserted document `{new_id}` into Qdrant!", icon="⚡")

    with mutation_tab_delete:
        del_id = st.text_input("Doc ID to Delete", "ms_marco_0", key="delete_id")
        if st.button("🗑️ Delete Passage", type="primary", use_container_width=True):
            if engine.delete_passage(del_id):
                st.toast(f"Deleted `{del_id}` from Qdrant!", icon="🗑️")
            else:
                st.error(f"Document `{del_id}` not found.")

# Interface Benchmark Trigger
st.title("⚡ AuraVector DB (Qdrant Powered)")
query = st.text_input("🔍 Search MS MARCO Corpus", value="What is MS MARCO passage ranking?")

if st.button("▶ Run Full Benchmark & RAGAS Eval", type="primary"):
    with st.spinner("Executing query pipeline..."):
        p50, p95 = run_latency_benchmark(pipeline)
        
        # Passes the entered / env groq_key to activate override
        eval_res = run_ragas_eval(
            pipeline,
            query,
            api_key=groq_key,
            mode=retrieval_mode
        )

        st.write("### Evaluation Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Latency (p50)", f"{p50:.2f} ms")
        m2.metric("Target Latency (p95)", f"{p95:.2f} ms")
        m3.metric(f"Precision ({eval_res['provider']})", f"{eval_res['precision']:.4f}")
        m4.metric(f"Recall ({eval_res['provider']})", f"{eval_res['recall']:.4f}")
