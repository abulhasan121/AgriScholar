# AgriScholar — Agricultural Research Intelligence

**Developed by Shah Md Abul Hasan**

AgriScholar is a semantic retrieval system designed for agricultural research literature exploration. Users can upload their own research papers and ask natural language questions across them. Every response is grounded in the uploaded documents and includes source citations with page references for traceability.

The system combines Retrieval-Augmented Generation (RAG), semantic search, query expansion, reranking, and large language models to create an intelligent agricultural research assistant capable of answering domain-specific scientific questions from private document collections.

---

# System Overview

AgriScholar follows a modern Retrieval-Augmented Generation (RAG) architecture.

Instead of relying entirely on the language model's internal knowledge, the system retrieves relevant passages from uploaded papers at query time and injects them into the prompt context. This produces answers that are:
- Grounded in source documents
- Traceable and citable
- More accurate for specialized agricultural terminology
- Updatable without retraining the model

The pipeline combines:
- Dense vector retrieval
- Query expansion
- Cross-encoder reranking
- Context-aware prompt assembly
- Claude Sonnet generation

---

# RAG Pipeline

```text
User Question
      ↓
Query Expansion
      ↓
Vector Embedding
      ↓
ChromaDB Similarity Search
      ↓
Multi-Query Fusion
      ↓
Cross-Encoder Reranking
      ↓
Context Assembly
      ↓
Claude Sonnet Generation
      ↓
Grounded Answer + Citations
```

---

# Core Retrieval Principles

## 1. Smart Document Processing

Uploaded PDFs are processed using PyPDF and LangChain document loaders.

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`:
- Chunk size: 1,000 characters
- Overlap: 200 characters

The splitter prioritizes:
1. Paragraph boundaries
2. Sentence boundaries
3. Word boundaries

before splitting raw text.

A preprocessing filter removes:
- Very short fragments
- Pure numeric sequences
- Reference-list noise
- Low-information chunks

This improves retrieval quality by ensuring only meaningful scientific text enters the vector database.

### Processing Pipeline

```text
PDF
  ↓
PyPDF Loader
  ↓
RecursiveCharacterTextSplitter
  ↓
Noise Filtering
  ↓
Embeddings
  ↓
ChromaDB
```

---

## 2. Query Expansion

Short user queries often fail because research papers may use different terminology than the user's wording.

To improve recall, the original question is sent to Claude Sonnet, which generates three semantically related query variants using agricultural domain knowledge.

### Example

```text
Original Query:
"weed control"

Expanded Queries:
- herbicide application methods
- integrated weed management strategies
- chemical vs mechanical weed suppression
```

All expanded queries are searched in parallel.

This increases retrieval robustness without requiring additional effort from the user.

---

## 3. Vector Search and Multi-Query Fusion

Each query is encoded into a 384-dimensional embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are:
- L2 normalized
- Stored in ChromaDB
- Compared using cosine similarity

Each query retrieves the top matching chunks independently.

The results are then:
1. Pooled together
2. Deduplicated
3. Ranked by highest similarity score

### Retrieval Flow

```text
query
   ↓
embedding generation
   ↓
cosine similarity search
   ↓
top-10 retrieval per query
   ↓
candidate pooling
   ↓
deduplication
   ↓
top candidates
```

### Embedding Model

| Model | Details |
|---|---|
| all-MiniLM-L6-v2 | 384 dimensions, 22M parameters |

The model was selected because it provides strong semantic similarity performance while remaining lightweight enough for CPU deployment.

---

## 4. Cross-Encoder Reranking

Dense vector retrieval is efficient but approximate because query and document embeddings are generated independently.

To improve precision, the top retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike bi-encoders, the cross-encoder reads the query and document together in a single forward pass, enabling direct interaction between terms.

This improves retrieval quality for:
- Agricultural terminology
- Scientific phrasing
- Context-sensitive language

### Reranking Pipeline

```text
10 retrieved candidates
        ↓
cross-encoder scoring
        ↓
relevance ranking
        ↓
top-5 chunks
```

This two-stage retrieval architecture follows modern RAG best practices:
- Fast recall via embeddings
- Accurate ranking via cross-encoder

---

## 5. Context Window Management

The top reranked chunks are assembled into a structured prompt before being sent to Claude Sonnet.

Each chunk includes:
- Source filename
- Page number
- Relevance score

A strict 6,000-character budget is enforced to maintain generation quality and reduce hallucination risk.

Chunks are inserted:
1. Highest relevance first
2. Until the context budget is reached

If the limit is exceeded mid-chunk, the remaining text is truncated gracefully rather than discarded entirely.

### Example Context Format

```python
[Source 1: paper.pdf, Page 4, Score: 0.923]
Crop rotation with non-host species reduces soil weed seed banks...

[Source 2: review.pdf, Page 11, Score: 0.887]
Integrated weed management combines cultural, mechanical...
```

---

## 6. Metadata Filtering

An optional metadata filter allows retrieval from a single selected paper.

This is implemented using ChromaDB's `where` clause during query execution.

This enables focused questions such as:

```text
"What does this paper say about nitrogen management?"
```

without interference from unrelated papers in the database.

---

# End-to-End Pipeline Timing

```text
User Question
     │
     ▼  ~0.5 s
Query Expansion via Claude
     │
     ▼  ~0.3 s
Vector Search in ChromaDB
     │
     ▼  ~0.2 s
Cross-Encoder Reranking
     │
     ▼
Context Assembly
     │
     ▼  ~1–2 s
Claude Sonnet Generation
     │
     ▼
Grounded Response with Citations
```

---

# Tech Stack

| Layer | Tool |
|---|---|
| Document Loading | LangChain + PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Similarity Metric | Cosine Similarity |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query Expansion | Claude Sonnet 4 |
| Answer Generation | Claude Sonnet 4 |
| Interface | Gradio |
| Deployment | Hugging Face Spaces (Docker) |

---

# Running Locally

## Clone Repository

```bash
git clone https://github.com/abulhasan121/AgriScholar.git
cd AgriScholar
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Set API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Launch Application

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

---

# Hugging Face Deployment

The live demo is deployed on Hugging Face Spaces:

https://huggingface.co/spaces/sh61309/AgriScholar

To deploy your own version:
1. Fork the Space
2. Add your `ANTHROPIC_API_KEY`
3. Configure it under:
   - Settings
   - Variables and Secrets

---

# Design Decisions and Limitations

## In-Memory Vector Database

The current implementation uses in-memory ChromaDB storage.

Uploaded documents are reset whenever the Hugging Face Space restarts.

For persistence, future versions could integrate:
- Pinecone
- Weaviate
- Persistent Chroma storage

---

## No Multi-Hop Reasoning

The system primarily answers questions using single retrieved passages.

Complex synthesis across multiple papers or sections may still be incomplete.

---

## Context Length Constraints

The context window is capped at 6,000 characters.

This tradeoff was intentionally selected to:
- Improve response quality
- Reduce hallucinations
- Maintain low latency

---

## CPU-Based Inference

The free-tier deployment runs entirely on CPU.

This increases:
- Embedding latency
- Reranking latency

GPU deployment would significantly improve response speed.

---

# Future Improvements

Potential future extensions include:
- Persistent vector database storage
- Multi-document reasoning
- Hybrid sparse + dense retrieval
- Citation-aware answer generation
- PDF highlighting and inline source previews
- GPU acceleration
- Fine-tuned agricultural embedding models

---

# Author

**Shah Md Abul Hasan**

Built with:
- LangChain
- ChromaDB
- Sentence Transformers
- Claude API
- Gradio

---

# License

MIT License
