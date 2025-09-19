from __future__ import annotations
import os, re
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from functools import lru_cache

# ===== Config =====
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION  = os.getenv("COLLECTION_NAME", "rare_diseases")

_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
_collection = _client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# ===== Helpers =====
def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def _embed(texts: List[str]) -> List[List[float]]:
    import ollama
    vecs: List[List[float]] = []
    for t in texts:
        r = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=t,
            options={"keep_alive": "30m"}
        )
        vecs.append(r["embedding"])
    return vecs

@lru_cache(maxsize=256)
def _embed_one(text: str) -> List[float]:
    import ollama
    r = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text,
        options={"keep_alive": "30m"}  # ⚡ cache + warm
    )
    return r["embedding"]

# --- Guardrails de dominio/tema ---
_HEALTH_KEYWORDS = {
    "salud","síntoma","sintoma","signo","diagnóstico","diagnostico","tratamiento",
    "prevención","prevencion","pronóstico","pronostico","enfermedad","síndrome","sindrome",
    "trastorno","gen","genética","genetica","herencia","medicamento","terapia",
    "fisioterapia","rehabilitación","rehabilitacion","biomarcador","cribado","tamizaje",
    "riesgo","epidemiología","epidemiologia","guía","guia","consenso","criterios","ICD","CIE"
}

_DISEASE_ALIASES = {
    r"\bdown\b": "down",
    r"\btrisom(ía|ia)?\s*21\b": "down",
    r"\bmucopolisacaridos(.*)?\b": "mps",
    r"\bmucopolisacaridosis\b": "mps",
    r"\bmps\b": "mps",
    r"\bmps\s*[i1]\b": "mps",
    r"\bmps\s*ii\b": "mps",
    r"\bmps\s*iii\b": "mps",
    r"\b(hurler|hunter|sanfilippo|morquio)\b": "mps",
    r"\bwilliams\b": "williams",
    r"\bsíndrome\s+de\s+williams\b": "williams",
    r"\bsindrome\s+de\s+williams\b": "williams",
}

def _is_health_related(text: str) -> bool:
    t = text.lower()
    if any(kw in t for kw in _HEALTH_KEYWORDS):
        return True
    return _extract_topic(t) is not None

def _extract_topic(text: str) -> str | None:
    t = text.lower()
    for pattern, topic in _DISEASE_ALIASES.items():
        if re.search(pattern, t):
            return topic
    return None

# ===== Indexación / RAG =====
def upsert_document(
    doc_id: str,
    source: str,
    full_text: str,
    extra_meta: Dict | None = None,
    topic: str | None = None,
    replace: bool = True,
) -> int:
    # 1) Chunks
    chunks = _chunk_text(full_text)
    if not chunks:
        return 0

    # 2) Embeddings
    vectors = _embed(chunks)

    # 3) Si replace=True, borra todo lo previo de ese doc_id (sin pedir ids)
    if replace:
        _collection.delete(where={"doc_id": doc_id})

    # 4) IDs + metadatos
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        m = {"doc_id": doc_id, "source": source, "chunk": i}
        if topic:
            m["topic"] = topic
        if extra_meta:
            m.update(extra_meta)
        metadatas.append(m)

    # 5) Upsert en Chroma
    _collection.upsert(
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )
    return len(chunks)

def query_context(query: str, k: int = 5, where: dict | None = None) -> Tuple[str, List[Dict]]:  # k=5 ⚡
    qvec = _embed_one(query)  # ⚡ cache de embedding del query
    res = _collection.query(query_embeddings=[qvec], n_results=k, where=where)
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ctx = ""
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        m = m or {}
        tag = f"{m.get('source','?')}"
        if m.get("year"):
            tag += f" {m['year']}"
        if m.get("type"):
            tag += f" · {m['type']}"
        ctx += f"[{i}] ({tag}) {d}\n"
    return ctx.strip(), metas

def format_apa6_list(metas: List[Dict], limit: int = 4) -> List[str]:
    seen = {}
    apa_list = []
    for m in metas:
        if not m: continue
        doc_id = m.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        seen[doc_id] = True
        src = m.get("source", "Fuente desconocida")
        year = m.get("year", "s.f.")
        tipo = f" ({m['type']})" if m.get("type") else ""
        country = f", {m['country']}" if m.get("country") else ""
        url = f" {m['url']}" if m.get("url") else ""
        apa = f"{src}. ({year}). {doc_id}{tipo}{country}.{url}"
        apa_list.append(apa)
        if len(apa_list) >= limit:
            break
    return apa_list

def generate_answer(
    user_msg: str,
    screen_context: str = "",
    topic: str | None = None,
    min_year: int | None = None,
    types: list[str] | None = None,
    lang: str | None = None,
) -> Tuple[str, List[Dict], List[str]]:
    if not _is_health_related(user_msg) and topic is None:
        return (
            "Me centro exclusivamente en temas médicos (en especial enfermedades raras). "
            "Tu consulta parece ser de otro ámbito.", [], []
        )
    inferred = _extract_topic(user_msg)
    effective_topic = topic or inferred
    where = _compose_where(
        topic=effective_topic,
        lang=lang,
        min_year=min_year,
        types=types,
    )
    rag_text, metas = query_context(user_msg, k=5, where=where)  # k=5 ⚡

    if not rag_text:
        return ("No recuperé información suficiente para responder con calidad.", [], [])
    SYSTEM_PROMPT = """
                    Eres un asistente EDUCATIVO de salud, especializado en enfermedades raras.
                    Objetivo: ofrecer información general basada en evidencia (definiciones, síntomas/signos,
                    causas, pruebas diagnósticas a alto nivel y opciones de manejo generales), sin dar
                    diagnósticos personalizados ni dosis de medicamentos.

                    Reglas:
                    - Responde SIEMPRE en español, claro y conciso (4-8 oraciones).
                    - SÍ puedes enumerar síntomas y signos cuando te lo pidan.
                    - Evita frases como “no puedo proporcionar asistencia médica directa”.
                    En lugar de eso, da información general segura y una breve advertencia si procede.
                    - No sustituyes a un profesional; incluye señales de alarma cuando sea pertinente.
                    """
    topic_line = f"Tópico: {effective_topic}\n" if effective_topic else ""
    CONTEXT_BLOCK = topic_line + f"Contexto recuperado:\n{rag_text}\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Pantalla: {screen_context or 'N/A'}\n{CONTEXT_BLOCK}\nPregunta: {user_msg}".strip()},
    ]
    import ollama
    out = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={
            "temperature": 0.2,
            "num_predict": 220,
            "top_p": 0.9,
            "keep_alive": "30m",  # ⚡ evita cold start del LLM
        }
    )
    reply = (out.get("message") or {}).get("content", "").strip()
    citations_apa = format_apa6_list(metas)
    return reply or "No pude generar una respuesta en este momento.", metas, citations_apa

# ===== Admin =====
def list_docs() -> list[dict]:
    res = _collection.get(include=["metadatas"]) or {}
    metas = res.get("metadatas", []) or []
    counts: Dict[str, int] = {}
    sources: Dict[str, set] = {}
    for m in metas:
        if not m: continue
        did = m.get("doc_id", "unknown")
        src = m.get("source", "unknown")
        counts[did] = counts.get(did, 0) + 1
        sources.setdefault(did, set()).add(src)
    out = []
    for did, c in counts.items():
        out.append({
            "doc_id": did,
            "count_chunks": c,
            "sources": sorted(list(sources.get(did, []))),
        })
    return sorted(out, key=lambda x: x["doc_id"])

def doc_stats(doc_id: str) -> dict:
    res = _collection.get(where={"doc_id": doc_id}, include=["metadatas", "documents"]) or {}
    ids  = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    if not isinstance(ids, list):    ids = list(ids)
    if not isinstance(metas, list):  metas = list(metas)
    if not isinstance(docs, list):   docs = list(docs)
    srcs = sorted({(m or {}).get("source", "desconocido") for m in metas if isinstance(m, dict)})
    sample = [str(docs[i])[:200] for i in range(min(2, len(docs)))]
    count = len(ids) if ids else len(metas)
    return {"doc_id": doc_id, "count": count, "sources": srcs, "sample": sample}

def delete_document(doc_id: str) -> int:
    # 1) contar cuántos hay de ese doc_id (sin include=["ids"])
    res = _collection.get(where={"doc_id": doc_id}) or {}
    ids = res.get("ids") or []
    count = len(ids) if ids else len(res.get("metadatas") or [])

    # 2) borrar por filtro (no por ids)
    if count > 0:
        _collection.delete(where={"doc_id": doc_id})

    return count

def _compose_where(
    topic: str | None = None,
    lang: str | None = None,
    min_year: int | None = None,
    types: list[str] | None = None,
    doc_id: str | None = None,
) -> dict | None:
    """Convierte filtros sueltos en un where válido para Chroma >= 0.5."""
    parts = []
    if topic:
        parts.append({"topic": {"$eq": topic}})
    if lang:
        parts.append({"lang": {"$eq": lang}})
    if min_year is not None:
        parts.append({"year": {"$gte": min_year}})
    if types:
        parts.append({"type": {"$in": types}})
    if doc_id:
        parts.append({"doc_id": {"$eq": doc_id}})

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
