import os, time, hashlib, warnings, re
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import anthropic
import gradio as gr
import numpy as np

# ── Config ─────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COLLECTION_NAME   = "research_papers"
EMBED_MODEL       = "all-MiniLM-L6-v2"
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
TOP_K_RETRIEVE    = 10
TOP_K_FINAL       = 5
MAX_CONTEXT_CHARS = 6000

# ── Load models ────────────────────────────────────────────
print("Loading embedding model…")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedder ready")

print("Loading reranker model…")
reranker = CrossEncoder(RERANK_MODEL)
print("✅ Reranker ready")

# ── ChromaDB (in-memory — persists for the lifetime of the Space) ──
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"✅ Existing collection: {collection.count()} chunks")
except Exception:
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("✅ New collection created")


# ══════════════════════════════════════════════════════════
# BACKEND
# ══════════════════════════════════════════════════════════

def _is_noise(text: str) -> bool:
    text = text.strip()
    if len(text) < 80: return True
    if re.match(r'^[\d\s\.\-\(\)]+$', text): return True
    if text.count('\n') > 5 and re.search(r'\[\d+\]', text): return True
    return False


def process_pdf(pdf_path: str) -> dict:
    source = Path(pdf_path).name
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    if not pages:
        return {"source": source, "pages": 0, "chunks": 0, "error": "empty"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks
              if len(c.page_content.strip()) > 80 and not _is_noise(c.page_content)]
    if not chunks:
        return {"source": source, "pages": len(pages), "chunks": 0,
                "error": "no content after filtering"}

    texts      = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  normalize_embeddings=True).tolist()
    ids, metadatas = [], []
    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{source}_{i}_{chunk.page_content[:50]}".encode()).hexdigest()
        ids.append(uid)
        metadatas.append({
            "source":   source,
            "page":     str(chunk.metadata.get("page", "?")),
            "chunk_id": str(i),
            "length":   str(len(chunk.page_content)),
        })

    existing = set(collection.get()["ids"])
    new_ids  = [i for i in ids if i not in existing]
    if new_ids:
        idx = [ids.index(i) for i in new_ids]
        collection.add(
            ids=new_ids,
            embeddings=[embeddings[j] for j in idx],
            documents=[texts[j] for j in idx],
            metadatas=[metadatas[j] for j in idx],
        )
    return {"source": source, "pages": len(pages),
            "chunks": len(new_ids), "total": collection.count()}


def expand_query(query: str) -> List[str]:
    if not ANTHROPIC_API_KEY:
        return [query]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=300,
        system=("You are a search query expert for agricultural research. "
                "Generate 3 search queries from the user question. "
                "Return ONLY the 3 queries, one per line, no numbering, no explanation."),
        messages=[{"role": "user",
                   "content": f"Original question: {query}\n\nGenerate 3 search queries:"}],
    )
    lines    = response.content[0].text.strip().split("\n")
    expanded = [q.strip() for q in lines if q.strip()][:3]
    all_q    = [query] + expanded
    seen, unique = set(), []
    for q in all_q:
        if q.lower() not in seen:
            seen.add(q.lower()); unique.append(q)
    return unique


def vector_search(queries: List[str], top_k: int = TOP_K_RETRIEVE,
                  source_filter: str = None) -> List[dict]:
    candidate_pool = {}
    where = {"source": source_filter} if source_filter else None
    for query in queries:
        q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
        n = min(top_k, collection.count())
        if n == 0:
            continue
        results = collection.query(
            query_embeddings=[q_emb], n_results=n,
            include=["documents", "metadatas", "distances"], where=where,
        )
        for i in range(len(results["documents"][0])):
            text       = results["documents"][0][i]
            similarity = round(1 - results["distances"][0][i], 3)
            uid        = hashlib.md5(text[:50].encode()).hexdigest()
            if uid not in candidate_pool:
                candidate_pool[uid] = {
                    "text":       text,
                    "source":     results["metadatas"][0][i]["source"],
                    "page":       results["metadatas"][0][i]["page"],
                    "similarity": similarity,
                }
            else:
                candidate_pool[uid]["similarity"] = max(
                    candidate_pool[uid]["similarity"], similarity)
    return sorted(candidate_pool.values(),
                  key=lambda x: x["similarity"], reverse=True)[:top_k]


def rerank_chunks(query: str, chunks: List[dict],
                  top_k: int = TOP_K_FINAL) -> List[dict]:
    if not chunks: return []
    pairs  = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def build_context(chunks: List[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        header = (f"[Source {i}: {chunk['source']}, Page {chunk['page']}, "
                  f"Score: {chunk.get('rerank_score', chunk['similarity']):.3f}]")
        entry = f"{header}\n{chunk['text']}"
        if total + len(entry) > max_chars:
            rem = max_chars - total
            if rem > 200: parts.append(entry[:rem] + "… [truncated]")
            break
        parts.append(entry); total += len(entry)
    return "\n\n---\n\n".join(parts)


def retrieve(query: str, source_filter: str = None) -> Tuple[List[dict], dict]:
    debug = {}
    t0 = time.time(); queries = expand_query(query)
    debug["query_expansion_s"] = round(time.time() - t0, 2)
    debug["expanded_queries"]  = queries
    t0 = time.time(); candidates = vector_search(queries, TOP_K_RETRIEVE, source_filter)
    debug["vector_search_s"]      = round(time.time() - t0, 2)
    debug["candidates_retrieved"] = len(candidates)
    t0 = time.time(); chunks = rerank_chunks(query, candidates, TOP_K_FINAL)
    debug["reranking_s"]         = round(time.time() - t0, 2)
    debug["chunks_after_rerank"] = len(chunks)
    return chunks, debug


def generate_answer(question: str, chunks: List[dict], history: List = []) -> str:
    if not ANTHROPIC_API_KEY:
        return "⚠️ No `ANTHROPIC_API_KEY` found. Add it in Space Settings → Secrets."
    context  = build_context(chunks)
    messages = []
    for human, asst in history[-3:]:
        messages.append({"role": "user",      "content": human})
        messages.append({"role": "assistant", "content": asst})
    messages.append({
        "role":    "user",
        "content": f"Context from research papers:\n\n{context}\n\n---\n\nQuestion: {question}",
    })
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1500,
        system="""You are AgriScholar, an expert research assistant for agricultural science.
Answer using ONLY the provided context from uploaded research papers.

Format every response as:

**Answer:**
[Detailed answer]

**Key Points:**
- [Point 1]
- [Point 2]
- [Point 3]

**📚 Sources:**
- [Paper name, Page X — what it contributed]

If context is insufficient say so clearly.""",
        messages=messages,
    )
    return response.content[0].text


# ── App functions ───────────────────────────────────────────

def chat_qa(question: str, history: List[Tuple],
            source_filter: str = None) -> Tuple[str, List, str]:
    if not question.strip(): return "", history, ""
    if collection.count() == 0:
        msg = "⚠️ No papers uploaded yet. Upload PDFs in the Upload Papers tab."
        history.append((question, msg)); return "", history, ""
    sf = None if (not source_filter or source_filter == "All Papers") else source_filter
    t_start = time.time()
    chunks, debug = retrieve(question, source_filter=sf)
    t_ret = time.time() - t_start
    if not chunks:
        history.append((question, "No relevant content found. Try rephrasing."))
        return "", history, ""
    t0 = time.time(); answer = generate_answer(question, chunks, history)
    t_gen = time.time() - t0
    debug_md = (
        f"**🔍 RAG Pipeline Trace**\n\n"
        f"**1. Query Expansion** — {debug['query_expansion_s']}s\n"
        + "\n".join([f"   - *{q}*" for q in debug['expanded_queries']])
        + f"\n\n**2. Vector Search** — {debug['vector_search_s']}s\n"
        f"   - Model: `{EMBED_MODEL}` (cosine, normalized)\n"
        f"   - Candidates: {debug['candidates_retrieved']}\n\n"
        f"**3. Reranking** — {debug['reranking_s']}s\n"
        f"   - Model: `{RERANK_MODEL}`\n"
        f"   - Final chunks: {debug['chunks_after_rerank']}\n\n"
        f"**Total: {t_ret + t_gen:.2f}s**"
    )
    answer += (f"\n\n---\n⏱️ *Retrieval {t_ret:.2f}s · "
               f"Generation {t_gen:.2f}s · {len(chunks)} chunks*")
    history.append((question, answer))
    return "", history, debug_md


def upload_papers(files) -> Tuple[str, object]:
    if not files: return "⚠️ No files selected.", _paper_dd(True)
    results = []
    for f in files:
        try:
            r = process_pdf(f.name)
            if "error" in r:
                results.append(f"❌ {r['source']} — {r['error']}")
            else:
                results.append(f"✅ {r['source']}\n"
                               f"   Pages: {r['pages']} · New chunks: {r['chunks']}")
        except Exception as e:
            results.append(f"❌ {Path(f.name).name} — {e}")
    results.append(f"\n📚 Total in database: {collection.count()} chunks")
    return "\n\n".join(results), _paper_dd(True)


def summarize_paper(source_name: str, section: str) -> str:
    if not source_name or source_name in ["No papers yet", "All Papers"]:
        return "⚠️ Please select a specific paper."
    all_data = collection.get(include=["documents", "metadatas"])
    chunks   = [doc for doc, meta in
                zip(all_data["documents"], all_data["metadatas"])
                if meta["source"] == source_name]
    if not chunks: return f"⚠️ No content found for: {source_name}"
    sample  = "\n\n".join(chunks[:8])
    prompts = {
        "Full Summary":           "Comprehensive summary: objective, methodology, key findings, conclusions, implications.",
        "Methods":                "Methodology and experimental design: study design, data collection, analysis, models.",
        "Findings":               "Main findings and results: key data, statistics, outcomes.",
        "Limitations":            "Limitations: constraints, gaps, caveats.",
        "Practical Implications": "Practical recommendations for farmers, agronomists, or researchers.",
    }
    if not ANTHROPIC_API_KEY:
        return "⚠️ No `ANTHROPIC_API_KEY` found. Add it in Space Settings → Secrets."
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1000,
        system="You are AgriScholar. Summarize agricultural papers clearly.",
        messages=[{"role": "user", "content":
                   f"Paper: {source_name}\n\nText:\n{sample}\n\n"
                   f"Task: {prompts.get(section, prompts['Full Summary'])}"}],
    )
    return response.content[0].text


def db_stats() -> str:
    total = collection.count()
    if total == 0: return "📭 No papers uploaded yet."
    meta    = collection.get(include=["metadatas"])["metadatas"]
    sources = {}
    for m in meta: sources[m["source"]] = sources.get(m["source"], 0) + 1
    out = f"**📚 {total} chunks across {len(sources)} papers**\n\n"
    for src, cnt in sorted(sources.items()): out += f"- `{src}` — {cnt} chunks\n"
    return out


def get_papers(include_all: bool = False) -> List[str]:
    if collection.count() == 0: return ["No papers yet"]
    meta    = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(set(m["source"] for m in meta))
    return (["All Papers"] + sources) if include_all else sources


def _paper_dd(include_all: bool = False):
    p = get_papers(include_all)
    return gr.Dropdown(choices=p, value=p[0] if p else None)


print("✅ AgriScholar backend ready")


# ══════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;500&family=Geist:wght@300;400;500;600&display=swap');

:root {
    --soil:      #1c1008;
    --bark:      #2e1f0e;
    --moss:      #2a4027;
    --fern:      #3d6b45;
    --sage:      #6b9e72;
    --mint:      #a8d5ae;
    --cream:     #f5f0e8;
    --parchment: #ede7d6;
    --warm-white:#faf8f3;
    --amber:     #c8831a;
    --gold:      #e8b84b;
    --text:      #1c1008;
    --text-mid:  #4a3728;
    --text-soft: #7a6555;
    --border:    rgba(44,31,14,0.12);
    --shadow-sm: 0 1px 3px rgba(28,16,8,0.08);
    --shadow-md: 0 4px 16px rgba(28,16,8,0.10);
    --shadow-lg: 0 12px 40px rgba(28,16,8,0.14);
    --r:         8px;
    --r-lg:      14px;
    --font-body: 'Geist', sans-serif;
    --font-mono: 'Geist Mono', monospace;
    --font-serif:'Instrument Serif', Georgia, serif;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--cream) !important;
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 0 0 3rem !important; }

.as-hero {
    background: var(--moss);
    background-image:
        radial-gradient(ellipse 80% 60% at 110% -10%, rgba(104,160,90,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 50% 80% at -10% 120%, rgba(200,131,26,0.12) 0%, transparent 60%);
    padding: 2.8rem 3rem 2.4rem;
    position: relative; overflow: hidden;
}
.as-hero::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--fern), var(--gold), var(--fern));
}
.as-wordmark { display: flex; align-items: center; gap: .7rem; margin-bottom: .5rem; }
.as-leaf { font-size: 2.2rem; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.2)); }
.as-title {
    font-family: var(--font-serif); font-size: 2.4rem; font-style: italic;
    color: var(--mint); letter-spacing: -0.01em; line-height: 1; margin: 0;
}
.as-sub {
    font-family: var(--font-mono); font-size: .72rem; color: var(--sage);
    letter-spacing: .12em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.as-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.as-chip {
    font-family: var(--font-mono); font-size: .6rem; font-weight: 500;
    letter-spacing: .1em; text-transform: uppercase; color: var(--mint);
    background: rgba(255,255,255,0.07); border: 1px solid rgba(168,213,174,0.25);
    padding: .22rem .6rem; border-radius: 3px;
}

.tab-nav { background: var(--warm-white) !important; border-bottom: 1px solid var(--border) !important; padding: 0 2rem !important; gap: 0 !important; }
.tab-nav button { font-family: var(--font-body) !important; font-size: .82rem !important; font-weight: 500 !important; color: var(--text-soft) !important; padding: .85rem 1.2rem !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; background: none !important; transition: color .2s, border-color .2s !important; margin: 0 !important; }
.tab-nav button:hover { color: var(--fern) !important; background: rgba(61,107,69,0.04) !important; }
.tab-nav button.selected { color: var(--moss) !important; border-bottom-color: var(--fern) !important; font-weight: 600 !important; }

.tabitem { padding: 2rem 2.5rem !important; }

.as-card { background: var(--warm-white); border: 1px solid var(--border); border-radius: var(--r-lg); padding: 1.4rem 1.6rem; box-shadow: var(--shadow-sm); }
.as-card-title { font-family: var(--font-mono); font-size: .68rem; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--fern); margin-bottom: .8rem; display: flex; align-items: center; gap: .4rem; }
.as-card-title::before { content: ''; display: block; width: 3px; height: 12px; background: var(--fern); border-radius: 2px; }

.as-info { background: rgba(61,107,69,0.06); border: 1px solid rgba(61,107,69,0.18); border-radius: var(--r); padding: .9rem 1.1rem; font-family: var(--font-mono); font-size: .76rem; color: var(--text-mid); line-height: 1.7; margin-bottom: 1rem; }
.as-info b { color: var(--moss); }
.as-warn { background: rgba(200,131,26,0.06); border: 1px solid rgba(200,131,26,0.22); border-radius: var(--r); padding: .9rem 1.1rem; font-family: var(--font-mono); font-size: .76rem; color: var(--text-mid); margin-bottom: 1rem; }

input[type=text], textarea { font-family: var(--font-body) !important; font-size: .88rem !important; background: var(--warm-white) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; color: var(--text) !important; transition: border-color .2s, box-shadow .2s !important; }
input[type=text]:focus, textarea:focus { border-color: var(--fern) !important; box-shadow: 0 0 0 3px rgba(61,107,69,0.10) !important; outline: none !important; }

#ask-btn { background: var(--moss) !important; color: var(--mint) !important; font-family: var(--font-mono) !important; font-size: .78rem !important; font-weight: 500 !important; letter-spacing: .05em !important; border: none !important; border-radius: var(--r) !important; padding: .75rem 1.4rem !important; transition: background .2s, transform .1s !important; box-shadow: var(--shadow-sm) !important; }
#ask-btn:hover { background: var(--bark) !important; transform: translateY(-1px) !important; box-shadow: var(--shadow-md) !important; }
#ask-btn:active { transform: translateY(0) !important; }

#upload-btn { background: var(--fern) !important; color: #fff !important; font-family: var(--font-mono) !important; font-size: .78rem !important; font-weight: 500 !important; letter-spacing: .05em !important; border: none !important; border-radius: var(--r) !important; padding: .8rem 1.6rem !important; width: 100% !important; transition: background .2s, box-shadow .2s !important; box-shadow: var(--shadow-sm) !important; }
#upload-btn:hover { background: var(--moss) !important; box-shadow: var(--shadow-md) !important; }

#sum-btn { background: var(--amber) !important; color: #fff !important; font-family: var(--font-mono) !important; font-size: .78rem !important; font-weight: 500 !important; letter-spacing: .05em !important; border: none !important; border-radius: var(--r) !important; padding: .8rem 1.2rem !important; width: 100% !important; transition: background .2s !important; }
#sum-btn:hover { background: #a8681a !important; }

.chatbot { background: var(--warm-white) !important; border: 1px solid var(--border) !important; border-radius: var(--r-lg) !important; box-shadow: var(--shadow-md) !important; }
.chatbot .message.user { background: var(--parchment) !important; color: var(--text) !important; border-radius: 12px 12px 4px 12px !important; font-size: .88rem !important; border: 1px solid var(--border) !important; }
.chatbot .message.bot { background: rgba(42,64,39,0.05) !important; color: var(--text) !important; border-radius: 12px 12px 12px 4px !important; font-size: .88rem !important; border: 1px solid rgba(61,107,69,0.12) !important; }

.as-pipeline { display: flex; flex-direction: column; gap: .5rem; margin-top: .6rem; }
.as-step { display: flex; align-items: center; gap: .6rem; font-family: var(--font-mono); font-size: .73rem; color: var(--text-mid); }
.as-step-num { width: 20px; height: 20px; background: var(--fern); color: #fff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: .62rem; font-weight: 600; flex-shrink: 0; }
.as-divider { height: 1px; background: var(--border); margin: 1.2rem 0; }

label span { font-family: var(--font-mono) !important; font-size: .7rem !important; font-weight: 500 !important; letter-spacing: .08em !important; text-transform: uppercase !important; color: var(--text-soft) !important; }

.gr-markdown { font-family: var(--font-body) !important; font-size: .88rem !important; color: var(--text) !important; line-height: 1.7 !important; }
.gr-markdown h2 { font-family: var(--font-serif) !important; font-style: italic !important; color: var(--moss) !important; font-size: 1.4rem !important; border-bottom: 1px solid var(--border) !important; padding-bottom: .4rem !important; margin-top: 1.5rem !important; }
.gr-markdown h3 { font-family: var(--font-mono) !important; font-size: .78rem !important; text-transform: uppercase !important; letter-spacing: .1em !important; color: var(--fern) !important; }
.gr-markdown code { background: rgba(42,64,39,0.08) !important; color: var(--fern) !important; font-family: var(--font-mono) !important; font-size: .82em !important; padding: .1em .35em !important; border-radius: 3px !important; }
.gr-markdown strong { color: var(--moss) !important; }

.examples-holder .examples table td { font-family: var(--font-body) !important; font-size: .8rem !important; color: var(--fern) !important; background: rgba(61,107,69,0.04) !important; border: 1px solid rgba(61,107,69,0.12) !important; border-radius: 4px !important; cursor: pointer !important; transition: background .15s !important; padding: .4rem .7rem !important; }
.examples-holder .examples table td:hover { background: rgba(61,107,69,0.10) !important; color: var(--moss) !important; }

.as-footer { background: var(--soil); padding: 1.2rem 3rem; display: flex; align-items: center; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 2rem; }
.as-footer-brand { font-family: var(--font-serif); font-style: italic; color: var(--mint); font-size: 1rem; }
.as-footer-meta { font-family: var(--font-mono); font-size: .62rem; color: var(--sage); letter-spacing: .08em; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--sage); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--fern); }
"""

EXAMPLES = [
    "What are the main findings of the uploaded papers?",
    "How do these papers address herbicide resistance?",
    "What methods were used for weed detection?",
    "What are the recommended weed management practices?",
    "What future research directions are suggested?",
    "How does integrated weed management work?",
    "What economic impacts of weeds are reported?",
    "What limitations do these papers acknowledge?",
    "How is NDVI used in weed detection?",
    "What datasets or field experiments were conducted?",
]

with gr.Blocks(css=CSS, title="AgriScholar") as demo:

    gr.HTML("""
    <div class="as-hero">
        <div class="as-wordmark">
            <span class="as-leaf">🌿</span>
            <h1 class="as-title">AgriScholar</h1>
        </div>
        <p class="as-sub">Agricultural Research Intelligence · Developed by Shah Md Abul Hasan</p>
        <div class="as-chips">
            <span class="as-chip">Query Expansion</span>
            <span class="as-chip">Vector Search</span>
            <span class="as-chip">Cross-Encoder Reranking</span>
            <span class="as-chip">Context Management</span>
            <span class="as-chip">Claude API</span>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes="tab-nav"):

        # ── Tab 1: Upload ──────────────────────────────────
        with gr.Tab("📤  Upload Papers"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.HTML("""
                    <div class="as-info">
                        <b>Getting started</b><br>
                        Upload one or more agricultural research PDFs below.
                        Each file is chunked, embedded, and indexed automatically.
                        Once uploaded, switch to <b>Ask Questions</b> to query across them.
                    </div>
                    """)
                    upload_files = gr.File(
                        label="PDF files", file_types=[".pdf"],
                        file_count="multiple", height=160,
                    )
                    upload_btn = gr.Button("📥  Upload & Index", elem_id="upload-btn", size="lg")
                    upload_status = gr.Textbox(
                        label="Processing log", lines=7, interactive=False,
                        placeholder="Upload status will appear here…",
                    )
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="as-card">
                        <div class="as-card-title">Processing pipeline</div>
                        <div class="as-pipeline">
                            <div class="as-step"><span class="as-step-num">1</span>Text extracted via PyPDF</div>
                            <div class="as-step"><span class="as-step-num">2</span>Noise &amp; header filtering</div>
                            <div class="as-step"><span class="as-step-num">3</span>Split into 1 000-char chunks</div>
                            <div class="as-step"><span class="as-step-num">4</span>Embedded (all-MiniLM-L6-v2)</div>
                            <div class="as-step"><span class="as-step-num">5</span>Stored in ChromaDB</div>
                        </div>
                        <div class="as-divider"></div>
                        <div class="as-card-title">Accepted formats</div>
                        <div style="font-family:var(--font-mono);font-size:.72rem;color:var(--text-soft);line-height:1.9;">
                            arXiv / journal articles<br>
                            USDA · FAO · extension reports<br>
                            Conference proceedings<br>
                            Any text-based agricultural PDF
                        </div>
                    </div>
                    """)
                    gr.HTML('<div style="height:.8rem;"></div>')
                    stats_btn = gr.Button("📊  Knowledge Base Stats", variant="secondary")
                    stats_out = gr.Markdown("*Click to see database stats*")

        # ── Tab 2: Chat ────────────────────────────────────
        with gr.Tab("💬  Ask Questions"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=240):
                    gr.HTML("""
                    <div class="as-card">
                        <div class="as-card-title">RAG Pipeline</div>
                        <div class="as-pipeline">
                            <div class="as-step"><span class="as-step-num">1</span>Query expansion ×4</div>
                            <div class="as-step"><span class="as-step-num">2</span>Vector search (cosine)</div>
                            <div class="as-step"><span class="as-step-num">3</span>Cross-encoder reranking</div>
                            <div class="as-step"><span class="as-step-num">4</span>Context trimming</div>
                            <div class="as-step"><span class="as-step-num">5</span>Claude generation</div>
                        </div>
                    </div>
                    """)
                    gr.HTML("""
                    <div class="as-warn" style="margin-top:.8rem;">
                        ⚠️ Upload papers first in the Upload Papers tab.
                    </div>
                    """)
                    source_filter  = gr.Dropdown(
                        choices=get_papers(include_all=True), value="All Papers",
                        label="Filter by paper", interactive=True,
                    )
                    refresh_filter = gr.Button("🔄  Refresh", variant="secondary")
                    gr.HTML('<div style="height:.4rem;"></div>')
                    gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[], label="Example questions")

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="AgriScholar",
                        height=460,
                    )
                    with gr.Row():
                        q_box   = gr.Textbox(
                            placeholder="Ask about methods, findings, weed control, crop recommendations…",
                            label="Question", lines=2, scale=5,
                        )
                        ask_btn = gr.Button("Ask ▶", elem_id="ask-btn", scale=1)
                    gr.HTML("""
                    <div class="as-info" style="margin-top:.6rem;margin-bottom:0;">
                        💡 Mention the crop, weed species, or growth stage for more precise answers.
                        Answers always cite the source paper and page number.
                    </div>
                    """)
                    clear_btn = gr.Button("🗑  Clear Chat", variant="secondary", size="sm")

            with gr.Accordion("🔍  RAG Pipeline Debug", open=False):
                debug_out = gr.Markdown("*Ask a question to see the pipeline trace here.*")

        # ── Tab 3: Summarize ───────────────────────────────
        with gr.Tab("📋  Summarize Paper"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML("""
                    <div class="as-card">
                        <div class="as-card-title">Section-Aware Summaries</div>
                        <div style="font-family:var(--font-mono);font-size:.72rem;color:var(--text-soft);line-height:1.9;">
                            <b style="color:var(--moss);">Full Summary</b> — whole paper<br>
                            <b style="color:var(--moss);">Methods</b> — how it was done<br>
                            <b style="color:var(--moss);">Findings</b> — key results<br>
                            <b style="color:var(--moss);">Limitations</b> — gaps &amp; caveats<br>
                            <b style="color:var(--moss);">Practical Implications</b> — real-world use
                        </div>
                    </div>
                    """)
                    gr.HTML('<div style="height:.6rem;"></div>')
                    paper_drop   = gr.Dropdown(choices=get_papers(), label="Select paper", interactive=True)
                    refresh_sum  = gr.Button("🔄  Refresh list", variant="secondary")
                    section_drop = gr.Dropdown(
                        choices=["Full Summary", "Methods", "Findings",
                                 "Limitations", "Practical Implications"],
                        value="Full Summary", label="Section", interactive=True,
                    )
                    sum_btn = gr.Button("📋  Generate Summary", elem_id="sum-btn", size="lg")
                with gr.Column(scale=2):
                    summary_out = gr.Markdown("*Select a paper and section, then click Generate Summary.*")

        # ── Tab 4: How It Works ────────────────────────────
        with gr.Tab("⚙️  How It Works"):
            gr.Markdown("""
## *How AgriScholar Works*

### Principle 1 — Smart Document Processing
Raw PDFs are parsed with PyPDF, split into 1 000-character overlapping chunks, with
short fragments, page headers, and reference lists filtered before embedding.

### Principle 2 — Query Expansion
Your question is sent to Claude, which generates three semantically related variants.
All four queries hit the vector store — closing the vocabulary gap between how you
phrase a question and how authors phrase their answers.

### Principle 3 — Vector Search
Each query is encoded by `all-MiniLM-L6-v2` (384-dim) and compared via cosine
similarity in ChromaDB. Top-10 candidates per query are pooled and deduplicated.

### Principle 4 — Cross-Encoder Reranking
`cross-encoder/ms-marco-MiniLM-L-6-v2` reads query and document together —
far more accurate than bi-encoder similarity for domain-specific language.

### Principle 5 — Context Window Management
Top-5 chunks assembled into a structured prompt capped at 6 000 characters,
prefixed with source name, page number, and relevance score.

### Principle 6 — Metadata Filtering
Optional paper filter restricts ChromaDB queries to a single document for
precise per-paper questions.
            """)

        # ── Tab 5: About ───────────────────────────────────
        with gr.Tab("ℹ️  About"):
            gr.Markdown("""
## *About AgriScholar*

A lightweight semantic retrieval system for exploring agricultural research literature,
developed by **Shah Md Abul Hasan**.

**Stack**

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Language model | Claude Sonnet 4 (Anthropic) |
| Framework | LangChain + Gradio |

> ⚠️ **To use this Space**, add your `ANTHROPIC_API_KEY` in Settings → Secrets.
            """)

    gr.HTML("""
    <div class="as-footer">
        <span class="as-footer-brand">AgriScholar</span>
        <span class="as-footer-meta">
            Developed by Shah Md Abul Hasan &nbsp;·&nbsp;
            Query Expansion · Vector Search · Cross-Encoder Reranking · Claude API
        </span>
    </div>
    """)

    # ── Wire events ─────────────────────────────────────────
    upload_btn.click(fn=upload_papers, inputs=[upload_files], outputs=[upload_status, source_filter])
    stats_btn.click(fn=db_stats, outputs=[stats_out])
    ask_btn.click(fn=chat_qa, inputs=[q_box, chatbot, source_filter], outputs=[q_box, chatbot, debug_out])
    q_box.submit(fn=chat_qa, inputs=[q_box, chatbot, source_filter], outputs=[q_box, chatbot, debug_out])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, debug_out])
    refresh_filter.click(fn=lambda: _paper_dd(True), outputs=[source_filter])
    refresh_sum.click(fn=lambda: _paper_dd(False), outputs=[paper_drop])
    sum_btn.click(fn=summarize_paper, inputs=[paper_drop, section_drop], outputs=[summary_out])

demo.launch(server_name="0.0.0.0", server_port=7860)
