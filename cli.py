from pathlib import Path
from src.rag.chunking import load_pdf_chunks
from src.rag.store import build_index, load_index
from src.rag.retrieval import search_raw, retrieve, K_FIRST, K_FINAL
from src.rag.answer import ask_openai, bullets_have_single_citation, citations

# ---- Config (same as your original) ----
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_ROOT  = Path("data")
INDEX_ROOT = Path("rag_index")
DATASET    = "bund_pfas"     # e.g., "bund_pfas", "tz_studium"
QUESTION   = "What are the harmful effects of PFAS?"
REBUILD    = False            # set False to reuse an existing index
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
# ----------------------------------------

DATA_DIR  = DATA_ROOT  / DATASET
INDEX_DIR = INDEX_ROOT / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load or build index
    if (INDEX_DIR / "faiss.index").exists() and not REBUILD:
        print(f"[INFO] Loading existing index for dataset='{DATASET}' ...")
        store, emb_model, emb, metas = load_index(INDEX_DIR, EMB_MODEL_NAME)
    else:
        print(f"[INFO] Building a new index from '{DATA_DIR}' (dataset='{DATASET}') ...")
        chunks = load_pdf_chunks(DATA_DIR, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            print(f"[ERROR] No PDFs in {DATA_DIR}. Please add files.")
            return
        store, emb_model, emb, metas = build_index(chunks, EMB_MODEL_NAME, INDEX_DIR)

    # 2) Query
    print("\n[QUERY]", QUESTION)

    # 3) Retrieval
    raw_hits, _ = search_raw(store, emb_model, QUESTION, K_FIRST)
    final_hits = retrieve(emb_model, emb, raw_hits, QUESTION, K_FINAL)

    print("\n[Contexts for prompt]")
    for h in final_hits:
        print(f"- {h['source']} p.{h['page']} | score={h.get('rerank', h['score']):.3f}")

    # 4) LLM answer
    ans = ask_openai(QUESTION, final_hits)
    print("\n===== ANSWER =====\n")
    print(ans)

    # 5) Guard (optional)
    ok = bullets_have_single_citation(ans)
    print(f"\n[Guard] Bullets have exactly one citation? -> {ok}")

    print("\n===== USED SOURCES =====")
    print(citations(final_hits))

if __name__ == "__main__":
    main()
