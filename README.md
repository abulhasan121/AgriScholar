# 🌿 AgriScholar — Agricultural Research Intelligence

**Developed by Shah Md Abul Hasan**

AgriScholar is a semantic retrieval system for exploring agricultural research literature. Upload your own research PDFs and ask natural language questions across them. Every answer is grounded in the source documents and cites the exact paper and page number it came from.

---

## 🧠 How It Works — RAG Pipeline

AgriScholar is built on **Retrieval-Augmented Generation (RAG)**, a technique that grounds large language model outputs in a private knowledge base — in this case, your uploaded research papers. Rather than relying on the model's parametric memory, RAG retrieves relevant passages at query time and feeds them as context to the LLM, making answers accurate, traceable, and up-to-date with your documents.

The system implements six RAG principles:

---

### Principle 1 — Smart Document Processing

Raw PDFs are loaded with **PyPDF** and split into overlapping 1 000-character chunks using LangChain's `RecursiveCharacterTextSplitter` with a 200-character overlap. The splitter respects natural text boundaries (paragraphs → sentences → words) before splitting mid-word. A noise filter removes chunks shorter than 80 characters, pure number sequences, and reference-list fragments — ensuring only meaningful prose enters the index.

```
PDF → PyPDF → RecursiveCharacterTextSplitter → noise filter → embeddings → ChromaDB
```

---

### Principle 2 — Query Expansion

Short or vague queries often miss relevant chunks because the vocabulary doesn't match the paper's terminology. To close this gap, the user's question is sent to **Claude**, which generates three semantically related query variants using agricultural domain knowledge.

**Example:**
> User asks: *"weed control"*
> Expanded to: *"herbicide application methods"*, *"integrated weed management strategies"*, *"chemical vs mechanical weed suppression"*

All four queries are sent to the vector store in parallel, dramatically improving recall without any extra user effort.

---

### Principle 3 — Vector Search with Multi-Query Fusion

Each query is encoded into a 384-dimensional dense vector using **`sentence-transformers/all-MiniLM-L6-v2`** and compared to all indexed chunks via **cosine similarity** in **ChromaDB**. Embeddings are L2-normalized before storage so cosine distance equals dot product — fast and accurate.

Results from all four queries are pooled into a single candidate set, deduplicated by chunk identity, and the highest similarity score per chunk is kept. This multi-query fusion ensures that a chunk matching any of the expanded queries is surfaced.

```
query → normalize → cosine search → top-10 per query → pool → deduplicate → top-10 candidates
```

**Model:** `all-MiniLM-L6-v2` — 384 dimensions, 22M parameters, optimized for semantic similarity.

---

### Principle 4 — Cross-Encoder Reranking

Bi-encoder vector search is fast but approximate — it scores query and document independently. For higher accuracy, the top-10 candidates are passed through a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which reads the query and each document *together* in a single forward pass, attending to their interaction directly.

This two-stage retrieval pattern (fast bi-encoder recall → accurate cross-encoder precision) is a well-established RAG best practice. The cross-encoder is especially effective for agricultural domain language where subtle terminology differences (e.g. "herbicide tolerance" vs "herbicide resistance") carry significant meaning.

```
10 candidates → cross-encoder scores each (query, chunk) pair → ranked by relevance → top-5
```

---

### Principle 5 — Context Window Management

The top-5 reranked chunks are assembled into a structured prompt with a hard cap of **6 000 characters**. Each chunk is prefixed with its source filename, page number, and relevance score — giving the LLM provenance information to produce grounded, citable answers.

Chunks are inserted highest-score-first. If the budget is exceeded mid-chunk, the remainder is truncated gracefully rather than dropped entirely. This ensures the most relevant content always makes it into the context window.

```python
[Source 1: paper.pdf, Page 4, Score: 0.923]
Crop rotation with non-host species reduces soil weed seed banks...

[Source 2: review.pdf, Page 11, Score: 0.887]
Integrated weed management combines cultural, mechanical...
```

---

### Principle 6 — Metadata Filtering

An optional paper filter restricts all ChromaDB queries to a single source document using the `where` clause at query time. This enables precise per-paper questions ("What does *this* paper say about nitrogen?") without interference from other papers in the database — useful when cross-paper noise is undesirable.

---

## 🔁 Full Pipeline at a Glance

```
User Question
     │
     ▼  ~0.5s
Query Expansion via Claude  ──►  4 queries total
     │
     ▼  ~0.3s
Vector Search (ChromaDB)  ──►  up to 40 candidates, deduplicated to 10
     │
     ▼  ~0.2s
Cross-Encoder Reranking  ──►  top 5 chunks
     │
     ▼
Context Assembly (6 000-char budget, source headers)
     │
     ▼  ~1–2s
Claude Sonnet Generation
     │
     ▼
Answer + Key Points + Cited Sources + Timing trace
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Document loading | LangChain + PyPDF |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (cosine, normalized) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Query expansion & generation | Claude Sonnet 4 (Anthropic API) |
| Interface | Gradio |
| Deployment | Hugging Face Spaces (Docker) |

---

## 🚀 Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/AgriScholar.git
cd AgriScholar
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## ☁️ Hugging Face Space

The live demo is deployed at:
**[huggingface.co/spaces/sh61309/AgriScholar](https://huggingface.co/spaces/sh61309/AgriScholar)**

To run your own copy, fork the Space and add your `ANTHROPIC_API_KEY` as a Secret in Settings → Variables and Secrets.

---

## 📌 Design Decisions & Limitations

- **In-memory ChromaDB** — uploaded papers reset when the Space restarts. For persistence, a paid Space with disk storage or an external vector DB (Pinecone, Weaviate) would be needed.
- **No multi-hop reasoning** — answers are grounded in single retrieved passages. Complex questions requiring synthesis across many sections may be incomplete.
- **Context cap at 6 000 chars** — very long answers may be truncated. This is a deliberate tradeoff to maintain LLM output quality.
- **Free-tier CPU** — embedding and reranking run on CPU, adding ~1s latency vs GPU.

---

## 👤 Author

**Shah Md Abul Hasan**
Agricultural Research Intelligence · Built with LangChain, ChromaDB, Sentence Transformers & Claude API
