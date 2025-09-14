from pathlib import Path
import sys

# Make repo root importable when running from streamlit_app/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import streamlit as st
from pathlib import Path

from src.rag.store import load_index
from src.rag.retrieval import search_raw, retrieve, K_FIRST, K_FINAL
from src.rag.answer import ask_openai, citations
from src.rag.chunking import load_pdf_chunks
from src.rag.store import build_index, load_index

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def get_model():
    # cache the SentenceTransformer model across reruns
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL_NAME)

def ensure_index(dataset: str):
    """Load index if present; otherwise build from data/<dataset> (demo-sized only)."""
    data_dir  = Path("data") / dataset
    index_dir = Path("rag_index") / dataset
    index_dir.mkdir(parents=True, exist_ok=True)

    if (index_dir / "faiss.index").exists():
        # fast path: index already there
        return load_index(index_dir, EMB_MODEL_NAME)

    # build path: small demo only (OK for Streamlit Cloud)
    chunks = load_pdf_chunks(data_dir)
    if not chunks:
        raise RuntimeError(f"No PDFs in {data_dir}. Add a small demo set.")
    # pass a cached encoder by name inside build_index (it will instantiate internally);
    # if you want to force reuse of get_model(), you can modify build_index to accept a model instance.
    return build_index(chunks, EMB_MODEL_NAME, index_dir)


def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


st.set_page_config(page_title="What do they say? - RAG MVP", layout="wide")
st.title("🔎 What do they say? - RAG MVP")

with st.sidebar:
    st.header("Dataset")
    dataset = st.text_input("Folder under /data", "demo")
    st.caption("Put PDFs in data/<dataset> and build an index locally first.")
    st.divider()
    st.header("Prompts (read-only for now)")
    st.text("System + guide enforce verbatim quotes in original language.\nSee src/rag/answer.py")

question = st.text_area("Your question", "What are the harmful effects of PFAS?")
run = st.button("Run")

if run:
    if not get_api_key():
        st.error("Please set OPENAI_API_KEY (.env or Streamlit Secrets).")
        st.stop()

    with st.status("Loading/Building index...", expanded=False):
        store, emb_model, emb, metas = ensure_index(dataset)

    with st.spinner("Searching & reranking..."):
        hits, _ = search_raw(store, emb_model, question, K_FIRST)
        final_hits = retrieve(emb_model, emb, hits, question, K_FINAL)

    with st.expander("Retrieved contexts"):
        for h in final_hits[:10]:
            st.markdown(f"- **{h['source']}** p.{h['page']} · score={h.get('rerank', h['score']):.3f}")

    with st.spinner("Asking OpenAI..."):
        ans = ask_openai(question, final_hits)

    st.markdown("## Answer")
    st.markdown(ans)

    st.markdown("### Sources")
    st.code(citations(final_hits))
