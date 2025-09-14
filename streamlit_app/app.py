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

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


st.set_page_config(page_title="What do they say? — RAG MVP", layout="wide")
st.title("🔎 What do they say? — RAG MVP")

with st.sidebar:
    st.header("Dataset")
    dataset = st.text_input("Folder under /data", "bund_pfas")
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

    index_dir = Path("rag_index") / dataset
    if not (index_dir / "faiss.index").exists():
        st.error(f"No index found at {index_dir}. Build it locally with cli.py first.")
        st.stop()

    with st.status("Loading index & model...", expanded=False):
        store, emb_model, emb, metas = load_index(index_dir, EMB_MODEL_NAME)

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
