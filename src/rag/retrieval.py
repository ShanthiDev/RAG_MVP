from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from functools import lru_cache

K_FIRST, K_FINAL = 40, 15
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

def search_raw(store, emb_model: SentenceTransformer, query: str, k: int = K_FIRST):
    """Vector search with inner product on normalized embeddings. Returns (hits, qvec)."""
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    hits = store.search(qv, k)
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits, qv

@lru_cache(maxsize=1)
def get_reranker():
    """Lazy, cached cross-encoder for reranking."""
    return CrossEncoder(RERANK_MODEL)

def rerank(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_RERANKER:
        return hits
    ce = get_reranker()
    scores = ce.predict([(query, h["text"]) for h in hits]).tolist()
    for h, s in zip(hits, scores):
        h["rerank"] = float(s)
    return sorted(hits, key=lambda x: x["rerank"], reverse=True)

def mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k=K_FINAL, lambda_=0.6) -> List[int]:
    """Maximal Marginal Relevance for diversity; assumes cand_vecs aligned to reranked list."""
    selected = []
    if len(cand_vecs) == 0:
        return selected
    selected.append(0)
    candidates = list(range(1, len(cand_vecs)))
    while len(selected) < min(k, len(cand_vecs)):
        best_i, best_val = None, -1e9
        for i in candidates:
            sim_to_q = float(np.dot(query_vec, cand_vecs[i]))
            sim_to_sel = max(float(np.dot(cand_vecs[i], cand_vecs[j])) for j in selected)
            val = lambda_ * sim_to_q - (1 - lambda_) * sim_to_sel
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i); candidates.remove(best_i)
    return selected

def retrieve(emb_model, full_emb, hits, query, k_final=K_FINAL):
    """Rerank → select diverse set via MMR → stable sort by source/page/doc."""
    rer = rerank(query, hits)
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    cand_indices = [h["idx"] for h in rer]
    cand_vecs = full_emb[cand_indices]
    keep = mmr_select(qv, cand_vecs, k=k_final, lambda_=0.6)
    final = [rer[i] for i in keep]
    final.sort(key=lambda h: (h["source"], h["page"], h["doc_id"]))
    return final
