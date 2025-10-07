from __future__ import annotations
import os, re, time, logging
from typing import List, Dict, Tuple
from functools import lru_cache
import threading

import chromadb
from chromadb.config import Settings

# ===== Config =====
LLM_MODEL    = os.getenv("LLM_MODEL", "phi3:mini-instruct")  # ligero por defecto
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR   = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION   = os.getenv("COLLECTION_NAME", "rare_diseases")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
KEEP_ALIVE   = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

log = logging.getLogger("uvicorn.error")

# ===== Chroma =====
_allow_reset = os.getenv("CHROMA_ALLOW_RESET", "false").lower() == "true"
_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=_allow_reset)
)
_collection = _client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

# ===== Ollama client =====
import ollama
# Evitamos depender de tipos internos del SDK
try:
    from httpx import ConnectError as _HttpxConnectError
except Exception:  # en caso de que cambie el import
    class _HttpxConnectError(Exception): ...

_OllamaResponseError = Exception

# ⬇️ Timeout explícito para que FastAPI no se congele
_ollama = ollama.Client(host=OLLAMA_HOST, timeout=60)

# Serializamos el acceso al runner para evitar picos y flapping
_LLM_LOCK = threading.Lock()

def _chat(messages, model, **opts):
    """Chat con reintento corto y lock para reducir flapping del runner."""
    for i in range(2):  # 1 reintento
        try:
            with _LLM_LOCK:
                return _ollama.chat(
                    model=model,
                    messages=messages,
                    options=opts,
                    keep_alive=KEEP_ALIVE
                )
        except Exception as e:
            log.warning("[chat] intento=%s error=%s", i+1, e)
            if i == 1:
                raise
            time.sleep(0.6)

# === Fallbacks de modelos ===
_LLM_PREFERENCES = [
    os.getenv("LLM_MODEL", "phi3:mini-instruct"),
    "llama3.2:1b-instruct",
    "tinyllama:latest",
]

def _choose_llm() -> str:
    for m in _LLM_PREFERENCES:
        try:
            _ollama.show(m)
            return m
        except Exception:
            continue
    return _LLM_PREFERENCES[0]

# ===== Helpers =====
def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    """Chunking simple con solape; normaliza espacios para estabilidad."""
    text = " ".join(text.split())
    chunks, start, N = [], 0, len(text)
    while start < N:
        end = min(start + chunk_size, N)
        cut = end
        # intenta cortar bonito en espacio o ". "
        for sep in [". ", " ", ""]:
            idx = text.rfind(sep, start + 200, end)  # evita trozos demasiado cortos
            if idx != -1:
                cut = idx + (0 if sep == "" else len(sep))
                break
        chunks.append(text[start:cut].strip())
        if cut >= N:
            break
        start = max(0, cut - overlap)
    return [c for c in chunks if c]

def _try_embed_once(prompt: str) -> List[float]:
    r = _ollama.embeddings(model=EMBED_MODEL, prompt=prompt, keep_alive=KEEP_ALIVE)
    return r["embedding"]

def _try_embed(prompt: str, tries: int = 2) -> List[float]:
    last = None
    for i in range(tries):
        try:
            return _try_embed_once(prompt)
        except (_OllamaResponseError, _HttpxConnectError, Exception) as e:
            last = e
            log.warning("[embed] intento=%s error=%s", i+1, e)
            time.sleep(0.6)
    raise last  # dejamos que el caller decida

def _embed(texts: List[str]) -> List[List[float]]:
    """Para indexar: si falla, aborta (mejor no indexar a medias)."""
    vecs: List[List[float]] = []
    for t in texts:
        try:
            vecs.append(_try_embed(t))
        except Exception as e:
            raise RuntimeError(
                f"Embedding model '{EMBED_MODEL}' no está disponible o falló en Ollama ({str(e)}). "
                f"Precárgalo con: ollama pull {EMBED_MODEL}"
            ) from e
    return vecs

@lru_cache(maxsize=256)
def _embed_one(text: str) -> List[float]:
    """Para consulta: *graceful fallback* → si falla devolvemos [] y no bloqueamos la respuesta."""
    try:
        return _try_embed(text)
    except Exception as e:
        log.error("[embed_one] fallo embeddings (usaré fallback sin RAG): %s", e)
        return []  # señal para query_context de que no hay vector

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
    r"\bmucopolisacaridos(?:is)?\b": "mps",
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
    if any(kw in t for kw in _HEALTH_KEYWORDS): return True
    return _extract_topic(t) is not None

def _extract_topic(text: str) -> str | None:
    t = text.lower()
    for pattern, topic in _DISEASE_ALIASES.items():
        if re.search(pattern, t): return topic
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
    chunks = _chunk_text(full_text)
    if not chunks: return 0
    vectors = _embed(chunks)
    if replace:
        _collection.delete(where={"doc_id": doc_id})
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        m = {"doc_id": doc_id, "source": source, "chunk": i}
        if topic: m["topic"] = topic
        if extra_meta: m.update(extra_meta)
        metadatas.append(m)
    _collection.upsert(documents=chunks, embeddings=vectors, metadatas=metadatas, ids=ids)
    return len(chunks)

def query_context(query: str, k: int = 5, where: dict | None = None) -> Tuple[str, List[Dict]]:
    """Consulta y arma un contexto diverso (evita repetir el mismo doc_id)."""
    qvec = _embed_one(query)
    if not qvec:  # fallback: sin embeddings no intentamos buscar
        return "", []

    try:
        res = _collection.query(
            query_embeddings=[qvec],
            n_results=max(k*2, 10),
            where=where,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        log.error("[query_context] fallo Chroma: %s", e)
        return "", []

    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    picked_ids, final_pairs = set(), []
    for d, m in zip(docs, metas):
        m = m or {}
        did = m.get("doc_id")
        if not d or did in picked_ids:
            continue
        picked_ids.add(did)
        final_pairs.append((d, m))
        if len(final_pairs) >= k:
            break

    if not final_pairs:
        return "", []

    ctx_lines, final_metas = [], []
    for i, (d, m) in enumerate(final_pairs, start=1):
        # Mantén el contexto “limpio” para el LLM
        ctx_lines.append(f"[{i}] {d}")
        final_metas.append(m)

    return "\n".join(ctx_lines).strip(), final_metas

def format_apa6_list(metas: List[Dict], limit: int = 4) -> List[str]:
    seen, apa_list = set(), []
    for m in metas or []:
        if not isinstance(m, dict): 
            continue
        doc_id = (m.get("doc_id") or "Documento").strip()
        if doc_id in seen:
            continue
        seen.add(doc_id)
        src = (m.get("source") or "Fuente desconocida").strip()
        year = m.get("year") or "s.f."
        tipo = f" ({m.get('type')})" if m.get("type") else ""
        country = f", {m.get('country')}" if m.get("country") else ""
        url = f" {m.get('url')}" if m.get("url") else ""
        apa_list.append(f"{src}. ({year}). {doc_id}{tipo}{country}.{url}".strip())
        if len(apa_list) >= limit:
            break
    return apa_list

# ---------- Limpieza de salida (anti "call-center") ----------
_STOP_PHRASES = [
    r"\bpuedo\s+proporcionar(le|te)\b",
    r"\bpuedo\s+ofrecer(le|te)\b",
    r"\b(contáct|contacta|contacte|escríb|escribe)\b",
    r"\bnuestras?\s+fuentes\b",
    r"\bpara\s+obtener\s+m[aá]s\s+informaci[oó]n\b",
    r"\bproporci[oó]nanos\b",
]

def _tidy_output(text: str, max_sentences: int = 8) -> str:
    if not text:
        return text
    for pat in _STOP_PHRASES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    if len(sents) > max_sentences:
        text = " ".join(sents[:max_sentences]).strip()
    if text and text[-1] not in ".?!":
        text += "."
    return text

# ===== Fallbacks genéricos (cuando RAG no está disponible) =====
def _generic_definition(topic: str | None) -> str | None:
    t = (topic or "").lower()
    if t == "down":
        return ("El síndrome de Down es un trastorno genético causado por material extra del cromosoma 21 "
                "(habitualmente trisomía 21 completa). Se asocia a rasgos faciales característicos, "
                "discapacidad intelectual de grado variable y mayor riesgo de ciertas condiciones médicas. "
                "El diagnóstico se confirma con estudios cromosómicos (cariotipo o técnicas equivalentes).")
    if t == "williams":
        return ("El síndrome de Williams es una enfermedad genética rara debida a una microdeleción en 7q11.23 "
                "que afecta genes como ELN. Suele cursar con rasgos faciales característicos, estenosis supravalvular "
                "aórtica u otras vasculopatías, perfil cognitivo y conductual particular y talla baja.")
    if t == "mps":
        return ("Las mucopolisacaridosis (MPS) son trastornos metabólicos hereditarios por déficit enzimático lisosomal "
                "que impide la degradación de glucosaminoglucanos. Generan afectación multiorgánica progresiva; "
                "el diagnóstico es bioquímico y molecular y existen terapias de reemplazo enzimático para algunos tipos.")
    return None

def generate_answer(
    user_msg: str,
    screen_context: str = "",
    topic: str | None = None,
    min_year: int | None = None,
    types: list[str] | None = None,
    lang: str | None = None,
) -> Tuple[str, List[Dict], List[str]]:
    if not _is_health_related(user_msg) and topic is None:
        return ("Me centro exclusivamente en temas médicos (en especial enfermedades raras). "
                "Tu consulta parece ser de otro ámbito.", [], [])

    t = user_msg.lower()
    if re.search(r"\bdown\b", t):
        topic = topic or "down"

    inferred = _extract_topic(user_msg)
    effective_topic = topic or inferred
    where = _compose_where(topic=effective_topic, lang=lang, min_year=min_year, types=types)

    rag_text, metas = query_context(user_msg, k=5, where=where)

    # Si no hubo contexto (por fallo de embeddings o colección vacía), respondemos con fallback genérico
    if not rag_text:
        generic = _generic_definition(effective_topic)
        if generic:
            return (generic, [], [])
        return ("No recuperé información suficiente para responder con calidad.", [], [])

    SYSTEM_PROMPT = """
Eres un asistente EDUCATIVO de salud, especializado en enfermedades raras.
Objetivo: ofrecer información general basada en evidencia (definiciones, síntomas/signos,
causas, pruebas diagnósticas a alto nivel y opciones de manejo generales), sin dar
diagnósticos personalizados ni dosis de medicamentos.

ESTÁ TERMINANTEMENTE PROHIBIDO:
- Mencionar o parafrasear palabras como: "tópico", "topic", "fuentes recuperadas",
"contexto", "pantalla", "documento", "chunk", "ruta", "doc_id" o similares.
- Describir el proceso, el sistema, los pasos seguidos o el origen del contexto.
- Usar primera persona del plural ("podemos", "nuestras fuentes") o frases comerciales.

Salida SIEMPRE en español, clara, concisa (4-8 oraciones). Usa títulos cortos y viñetas si aplica.

Plantilla orientativa (ajústala al caso):
- Definición breve (1-2 oraciones).
- Causas/genética (1-2 oraciones).
- Manifestaciones frecuentes (3-6 viñetas compactas).
- Diagnóstico a alto nivel (1-2 oraciones).
- Acompañamiento/alertas (1 oración con señales de alarma o derivación).
""".strip()


    topic_line = f"Tópico sugerido: {effective_topic}\n" if effective_topic else ""
    context_block = f"{topic_line}Fuentes recuperadas (extractos):\n{rag_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_block},
        {"role": "user", "content": user_msg.strip()},
    ]

    try:
        model_to_use = _choose_llm()
        # parámetros modestos para entornos con recursos limitados
        out = _chat(
            messages,
            model_to_use,
            temperature=0.10,
            num_predict=220,
            num_ctx=2048,
            num_batch=16,
            top_p=0.9,
        )
    except _OllamaResponseError as e:
        raise RuntimeError(
            f"LLM '{LLM_MODEL}' no está disponible en Ollama ({str(e)}). "
            f"Precárgalo con: ollama pull {LLM_MODEL}"
        ) from e
    except Exception as e:
        log.error("[chat] fallo final: %s", e)
        generic = _generic_definition(effective_topic)
        if generic:
            return (generic, [], [])
        raise RuntimeError("No se pudo completar la generación en este momento.") from e

    raw = (out.get("message") or {}).get("content", "").strip()
    reply = _tidy_output(raw)
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
    out = [{"doc_id": did, "count_chunks": c, "sources": sorted(list(sources.get(did, [])))} for did, c in counts.items()]
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
    res = _collection.get(where={"doc_id": doc_id}) or {}
    ids = res.get("ids") or []
    count = len(ids) if ids else len(res.get("metadatas") or [])
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
    parts = []
    if topic: parts.append({"topic": {"$eq": topic}})
    if lang: parts.append({"lang": {"$eq": lang}})
    if min_year is not None: parts.append({"year": {"$gte": min_year}})
    if types: parts.append({"type": {"$in": types}})
    if doc_id: parts.append({"doc_id": {"$eq": doc_id}})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
