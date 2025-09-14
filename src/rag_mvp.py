# %% [markdown]
# RAG MVP: "What do they say?" – lokal, Cloud-ready
# - PDFs -> Chunks -> Embeddings -> FAISS -> Retrieval (+ Rerank + MMR)
# - Antwort via OpenAI (Summary -> Bullets mit [Dok S.Seite] -> Quellen)
# - Austauschbares VectorStore-Interface (FAISS heute, Azure AI Search später)

# %% Imports & Setup
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol, Tuple
import os, re, pickle, textwrap, sys
import numpy as np
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Optional: for auto-downloads (you can leave URLs empty)
import requests

# ---------- Config ----------
DATA_ROOT  = Path("data")
INDEX_ROOT = Path("rag_index")

DATASET   = "bund_pfas"   # <- z.B. "chemikalienpolitik", "verkehr", "energie"
QUESTION  = "Welche Forderungen stellt der BUND zu PFAS/REACH (Fristen, Klassenansatz)?"
REBUILD   = False         # True => Index für dieses Dataset neu bauen

DATA_DIR  = DATA_ROOT  / DATASET
INDEX_DIR = INDEX_ROOT / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)



EMB_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER     = True   # False = nur Vektor
CHUNK_SIZE       = 900
CHUNK_OVERLAP    = 150
K_FIRST          = 40     # grobe Vorauswahl
K_FINAL          = 15     # finale Vielfalt (MMR) in den Prompt
#OPENAI_CHAT_MODEL= "gpt-4o-mini"
OPENAI_CHAT_MODEL= "gpt-4o"
TEMPERATURE      = 0.1

# Falls du auto-downloaden willst: trage URLs ein (sonst lass Liste leer)
PDF_URLS = [
#  "https://www.bund.net/fileadmin/user_upload_bund/publikationen/ressourcen_und_technik/Herausforderungen_fuer_eine_nachhaltige_Stoffpolitik_Positionspapier_BUND_2023.pdf",
#  "https://www.bund.net/fileadmin/user_upload_bund/publikationen/chemie/chemie-pfas-verbot-manifest.pdf",
#  "https://www.bund.net/fileadmin/user_upload_bund/publikationen/chemie/PFAS-Brief-EU-Kommission-BUND.pdf",
#  "https://www.bund.net/fileadmin/user_upload_bund/publikationen/chemie/ToxFox-Test-PFAS-Lebensmittel-BUND-2025.pdf",
#  "https://www.bund.net/fileadmin/user_upload_bund/publikationen/chemie/nachhaltige-stoffpolitik-kurzfassung-bund.pdf",
]

# %% Utility: download PDFs (optional)
def maybe_download_pdfs(urls: List[str], dest: Path):
    for url in urls:
        fname = dest / url.split("/")[-1]
        if not fname.exists():
            print("Downloading:", fname.name)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            fname.write_bytes(r.content)
        else:
            print("Exists:", fname.name)

if PDF_URLS:
    maybe_download_pdfs(PDF_URLS, DATA_DIR)

# %% Chunking
@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    text: str

def clean_text(txt: str) -> str:
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def split_into_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        soft = text.rfind(".", start, end)
        if soft != -1 and soft > start + int(size*0.6):
            end = soft + 1
        out.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in out if c]

def load_pdf_chunks(pdf_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] Keine PDFs in {pdf_dir}. Bitte 5 PDFs einlegen oder PDF_URLS setzen.")
    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
        except Exception as e:
            print(f"[WARN] Konnte {pdf.name} nicht lesen: {e}")
            continue
        for p, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            txt = clean_text(raw)
            if not txt: 
                continue
            for c in split_into_chunks(txt):
                chunks.append(Chunk(pdf.stem, pdf.name, p, c))
    return chunks

# %% VectorStore Interface (Cloud-ready)
class VectorStore(Protocol):
    def upsert(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: List[str]) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

class FaissStore:
    def __init__(self, dim: int):
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []
        self.ids:  List[str] = []

    def upsert(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.meta.extend(metadatas)
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_vec.reshape(1,-1).astype(np.float32), k)
        hits = []
        for j, idx in enumerate(I[0]):
            if idx < 0: 
                continue
            m = self.meta[idx]
            hits.append({
                "score": float(D[0][j]),
                "idx": int(idx),
                **m
            })
        return hits

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump({"meta": self.meta, "ids": self.ids, "dim": self.dim}, f)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path / "faiss.index"))
        store = pickle.load(open(path / "meta.pkl","rb"))
        self.meta = store["meta"]; self.ids = store["ids"]; self.dim = store["dim"]

# (Später) Azure AI Search Adapter – nur Skeleton:
class AzureSearchStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Stub – später durch azure-search-documents implementieren.")

# %% Build Index (only if not exists)
def build_index(chunks: List[Chunk], emb_model_name: str, index_dir: Path) -> Tuple[FaissStore, SentenceTransformer, np.ndarray, List[str], List[Dict[str, Any]]]:
    print("[INFO] Lade Embedding-Modell:", emb_model_name)
    emb_model = SentenceTransformer(emb_model_name)
    texts = [c.text for c in chunks]
    metas = [{"doc_id": c.doc_id, "source": c.source, "page": c.page, "text": c.text} for c in chunks]
    ids   = [f"{c.doc_id}-{i}" for i, c in enumerate(chunks)]

    print("[INFO] Erzeuge Embeddings...")
    emb = emb_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dim = emb.shape[1]

    print("[INFO] Baue FAISS Index...")
    store = FaissStore(dim)
    store.upsert(emb, metas, ids)
    store.save(index_dir)
    np.save(index_dir / "embeddings.f32.npy", emb.astype(np.float32))
    with open(index_dir / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)

    return store, emb_model, emb, ids, metas

def load_index(index_dir: Path, emb_model_name: str):
    emb_model = SentenceTransformer(emb_model_name)
    dim = emb_model.get_sentence_embedding_dimension()

    store = FaissStore(dim)
    store.load(index_dir)

    # Embeddings & Metas direkt laden (kein Re-Encode!)
    emb = np.load(index_dir / "embeddings.f32.npy", mmap_mode="r")  # zero-copy
    with open(index_dir / "metas.pkl", "rb") as f:
        metas = pickle.load(f)

    ids = [f"id-{i}" for i in range(len(metas))]
    return store, emb_model, emb, ids, metas

# %% Retrieval (raw + rerank + MMR)
def search_raw(store: FaissStore, emb_model: SentenceTransformer, query: str, k: int) -> List[Dict[str, Any]]:
    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    hits = store.search(q[0], k)
    # index liefert Score (inner product), sortiere absteigend
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL)

def rerank(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not USE_RERANKER:
        return hits
    ce = get_reranker()  # cached
    pairs = [(query, h["text"]) for h in hits]
    scores = ce.predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h["rerank"] = float(s)
    return sorted(hits, key=lambda x: x["rerank"], reverse=True)

def mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k=K_FINAL, lambda_=0.6) -> List[int]:
    selected = []
    if len(cand_vecs) == 0:
        return selected
    selected.append(0)  # assume best already at 0
    candidates = list(range(1, len(cand_vecs)))
    while len(selected) < min(k, len(cand_vecs)):
        best_i, best_val = None, -1e9
        for i in candidates:
            sim_to_q = float(np.dot(query_vec, cand_vecs[i]))
            sim_to_sel = max(float(np.dot(cand_vecs[i], cand_vecs[j])) for j in selected)
            val = lambda_ * sim_to_q - (1 - lambda_) * sim_to_sel
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected

def retrieve(store, emb_model, full_emb, hits, query, k_final):
    # 1) Rerank (optional Cross-Encoder)
    rer = rerank(query, hits)

    # 2) Query-Vektor
    qv = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # 3) Kandidaten-Vektoren per 'idx' (FAISS Rückgabe-Index) holen – NICHT via .index(h)
    cand_indices = [h["idx"] for h in rer]
    cand_vecs = full_emb[cand_indices]

    # 4) MMR für Diversität
    keep = mmr_select(qv, cand_vecs, k=k_final, lambda_=0.6)
    final = [rer[i] for i in keep]

    # 5) hübsch sortieren
    final.sort(key=lambda h: (h["source"], h["page"], h["doc_id"]))
    return final

# %% Prompt & Answer
def citations(hits: List[Dict[str, Any]]) -> str:
    seen, out = set(), []
    for h in hits:
        tag = f"[{h['doc_id']} S.{h['page']}] {h['source']}"
        if tag not in seen:
            seen.add(tag); out.append(tag)
    return "\n".join(out)

SYS_PROMPT = (
    "Du bist ein tibetisch-buddhistischer Gelehrter. Antworte NUR auf Basis des gelieferten Kontextes.\n"
    "Erst eine kurze Fließtext-Zusammenfassung (3–6 Sätze), danach Bulletpoints,\n"
    "jede Zeile = genau EINE überprüfbare Einzelaussage mit genau EINEM Zitat, dass du bitte im Wortlaut wiedergibst [Dok S.Seite].\n"
    "Dannach die Langfassung der Antwort, die sich u.a. auf die gefundenen Zitate stützt."
    "Wenn dir Belege fehlen, sag es klar."
)

def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"### Kontext {i} — {h['source']} (S.{h['page']})\n{h['text']}")
    return "\n\n".join(blocks)

def ask_openai(query: str, hits: List[Dict[str, Any]], model=OPENAI_CHAT_MODEL, temperature=TEMPERATURE) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY fehlt (.env)")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    ctx = build_context(hits)
    guide = (
        "FORMAT:\n"
        "1) Kurz-Zusammenfassung (3–6 Sätze).\n"
        "2) Bulletpoints: pro Zeile genau EINE überprüfbare Aussage + genau EIN Zitat [Dok S.Seite].\n"
        "3) Langfassung der Antwort (so viele Sätze wie es braucht).\n"
        "4) Am Ende: 'Quellen' als Plaintext-Liste – jede Quelle in einer neuen Zeile, KEINE Bullets.\n"
        "5) Antworte gern bilingual DE/EN, wenn passend."
    )
    user = f"{ctx}\n\n---\nFrage: {query}\n\n{guide}\nQuellen:\n{citations(hits)}"

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens = 3000,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ]
    )
    return resp.choices[0].message.content

# Guard: jede Bullet soll genau ein Zitat haben (einfacher Check)
def bullets_have_single_citation(answer: str) -> bool:
    # Nur den Teil VOR "Quellen:" prüfen (case-insensitive)
    m = re.search(r"(?im)^quellen\s*:", answer)
    check_text = answer[:m.start()] if m else answer

    # Erlaubt: [Dok S.9] oder [Dok S. 9] sowie Bereiche [Dok S.9–10]
    pat = re.compile(r"\[[^\[\]]+ S\.\s*\d+(?:[-–]\d+)?\]")

    for ln in check_text.splitlines():
        s = ln.strip()
        if s.startswith(("-", "*")):     # echte Bullet-Zeilen
            matches = pat.findall(s)
            if len(matches) != 1:        # genau EIN Zitat pro Bullet
                # Debug optional:
                # print("FAIL BULLET:", s, "| matches:", matches)
                return False
    return True
# %% Main: Build/Load index, Ask

from pathlib import Path

DATA_ROOT  = Path("data")
INDEX_ROOT = Path("rag_index")

DATASET   = "bund_pfas"   # <- z.B. "chemikalienpolitik", "verkehr", "energie"
QUESTION  = "What are the harmful effects of pfas?"
REBUILD   = True         # True => Index für dieses Dataset neu bauen

DATA_DIR  = DATA_ROOT  / DATASET
INDEX_DIR = INDEX_ROOT / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) Index laden oder bauen (dataset-spezifisch)
    if (INDEX_DIR/"faiss.index").exists() and not REBUILD:
        print(f"[INFO] Lade existierenden Index für DATASET='{DATASET}' ...")
        store, emb_model, emb, ids, metas = load_index(INDEX_DIR, EMB_MODEL_NAME)
    else:
        print(f"[INFO] Baue neuen Index aus '{DATA_DIR}' (DATASET='{DATASET}') ...")
        chunks = load_pdf_chunks(DATA_DIR)
        if not chunks:
            print(f"[ERROR] Keine PDFs in {DATA_DIR}. Bitte Dateien reinlegen.")
            return
        store, emb_model, emb, ids, metas = build_index(chunks, EMB_MODEL_NAME, INDEX_DIR)

    # 2) Frage definieren
    user_query = QUESTION
    print("\n[QUERY]", user_query)

    # 3) Retrieval
    raw_hits = search_raw(store, emb_model, user_query, K_FIRST)
    final_hits = retrieve(store, emb_model, emb, raw_hits, user_query, K_FINAL)

    print("\n[Kontexte für Prompt]")
    for h in final_hits:
        print(f"- {h['source']} S.{h['page']} | score={h.get('rerank', h['score']):.3f}")

    # 4) OpenAI Antwort
    ans = ask_openai(user_query, final_hits)
    print("\n===== ANTWORT =====\n")
    print(ans)

    # 5) Guard (optional)
    ok = bullets_have_single_citation(ans)
    print(f"\n[Guard] Bullets haben genau ein Zitat? -> {ok}")

    print("\n===== GENUTZTE QUELLEN =====")
    print(citations(final_hits))

if __name__ == "__main__":
    main()

# %%
