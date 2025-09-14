from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np, pickle
import faiss
from sentence_transformers import SentenceTransformer

class FaissStore:
    """Minimal inner-product FAISS index with Python-side metadata."""
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []
        self.ids:  List[str] = []

    def upsert(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.meta.extend(metadatas)
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        hits = []
        for j, idx in enumerate(I[0]):
            if idx < 0:
                continue
            m = self.meta[idx]
            hits.append({"score": float(D[0][j]), "idx": int(idx), **m})
        return hits

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump({"meta": self.meta, "ids": self.ids, "dim": self.dim}, f)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path / "faiss.index"))
        store = pickle.load(open(path / "meta.pkl", "rb"))
        self.meta, self.ids, self.dim = store["meta"], store["ids"], store["dim"]

def build_index(chunks, emb_model_name: str, index_dir: Path) -> Tuple[FaissStore, SentenceTransformer, np.ndarray, List[Dict[str, Any]]]:
    """Encode chunks, build FAISS, persist index + embeddings + metadata."""
    emb_model = SentenceTransformer(emb_model_name)
    texts = [c.text for c in chunks]
    metas = [{"doc_id": c.doc_id, "source": c.source, "page": c.page, "text": c.text} for c in chunks]
    emb = emb_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    store = FaissStore(emb.shape[1])
    store.upsert(emb, metas, [f"{m['doc_id']}-{i}" for i, m in enumerate(metas)])
    store.save(index_dir)
    np.save(index_dir / "embeddings.f32.npy", emb.astype(np.float32))
    with open(index_dir / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)
    return store, emb_model, emb, metas

def load_index(index_dir: Path, emb_model_name: str):
    """Reload FAISS and memory-map the embeddings (no re-encode)."""
    emb_model = SentenceTransformer(emb_model_name)
    store = FaissStore(emb_model.get_sentence_embedding_dimension())
    store.load(index_dir)
    emb = np.load(index_dir / "embeddings.f32.npy", mmap_mode="r")
    metas = pickle.load(open(index_dir / "metas.pkl", "rb"))
    return store, emb_model, emb, metas
