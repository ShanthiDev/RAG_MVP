from pathlib import Path
from dataclasses import dataclass
from typing import List
import re
from pypdf import PdfReader

# Optional (only if you want auto-downloads)
import requests

@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    text: str

def clean_text(txt: str) -> str:
    """Light normalization to reduce weird whitespace and nulls."""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def split_into_chunks(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    """Greedy, sentence-aware chunking with a soft cut on '.' near the window end."""
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        soft = text.rfind(".", start, end)
        if soft != -1 and soft > start + int(size * 0.6):
            end = soft + 1
        out.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in out if c]

def load_pdf_chunks(pdf_dir: Path, size: int = 900, overlap: int = 150) -> List[Chunk]:
    """Load all PDFs from a folder, extract text per page, and chunk."""
    chunks: List[Chunk] = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No PDFs in {pdf_dir}. Put some PDFs there or set PDF_URLS.")
    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
        except Exception as e:
            print(f"[WARN] Could not read {pdf.name}: {e}")
            continue
        for p, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            txt = clean_text(raw)
            if not txt:
                continue
            for c in split_into_chunks(txt, size=size, overlap=overlap):
                chunks.append(Chunk(pdf.stem, pdf.name, p, c))
    return chunks

def maybe_download_pdfs(urls: List[str], dest: Path) -> None:
    """Optional helper: download PDFs to a folder."""
    for url in urls:
        fname = dest / url.split("/")[-1]
        if not fname.exists():
            print("Downloading:", fname.name)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            fname.write_bytes(r.content)
        else:
            print("Exists:", fname.name)
