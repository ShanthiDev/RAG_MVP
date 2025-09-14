import os, re
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

def build_context(hits: List[Dict[str, Any]]) -> str:
    """Render top snippets as context blocks."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"### Context {i} — {h['source']} (p.{h['page']})\n{h['text']}")
    return "\n\n".join(blocks)

def citations(hits: List[Dict[str, Any]]) -> str:
    """De-duplicated source list for the answer footer."""
    seen, out = set(), []
    for h in hits:
        tag = f"[{h['doc_id']} p.{h['page']}] {h['source']}"
        if tag not in seen:
            seen.add(tag); out.append(tag)
    return "\n".join(out)

def ask_openai(query: str, hits: List[Dict[str, Any]], model=OPENAI_CHAT_MODEL, temperature=TEMPERATURE) -> str:
    """Call OpenAI with strict format instructions and the retrieved context."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (.env)")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    SYS_PROMPT = (
        "You are a careful scholar answering ONLY from the provided context blocks. "
        "OUTPUT REQUIREMENTS:\n"
        "• Start with a short summary (3–6 sentences).\n"
        "• Then bullets: each bullet must contain (a) exactly ONE verifiable claim, "
        "(b) exactly ONE VERBATIM QUOTE from the source text IN ITS ORIGINAL LANGUAGE, "
        "and (c) exactly ONE citation tag in the form [Doc p.X].\n"
        "• After bullets, provide a long-form explanation.\n"
        "• If evidence is missing, say so clearly.\n"
        "• Do NOT translate the quotes; leave them exactly as in the source (German for current docs)."
    )
    guide = (
        "FORMAT:\n"
        "1) Short summary (3–6 sentences).\n"
        "2) Bullets (each line): Claim (EN) — \"<verbatim quote in original language>\" [Doc p.X]\n"
        "   - Exactly ONE quote per bullet, no paraphrase inside the quotes.\n"
        "3) Long-form answer.\n"
        "4) At the end: 'Sources' as plain text — one source per new line, NO bullets."
    )
    user = f"{build_context(hits)}\n\n---\nQuestion: {query}\n\n{guide}\nSources:\n{citations(hits)}"

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=3000,
        messages=[{"role":"system","content":SYS_PROMPT},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content


def bullets_have_single_citation(answer: str) -> bool:
    """Simple guard: each bullet line must contain exactly one [Doc p.X] pattern."""
    m = re.search(r"(?im)^sources\s*:", answer)
    check_text = answer[:m.start()] if m else answer
    pat = re.compile(r"\[[^\[\]]+ p\.\s*\d+(?:[-–]\d+)?\]")
    for ln in check_text.splitlines():
        s = ln.strip()
        if s.startswith(("-", "*")):
            if len(pat.findall(s)) != 1:
                return False
    return True
