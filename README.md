# 🧠 Ferdowsi University RAG Assistant

A Persian-language **Retrieval-Augmented Generation (RAG)** system designed to provide intelligent question-answering about professors and their research at **Ferdowsi University of Mashhad**.

Students can simply ask natural questions such as:

> “Who are the professors working on image processing?”
> “Show me publications by Dr. Sadoughi.”

…and receive accurate, concise, and well-sourced answers in Persian.

---

## 🚀 Features

* **Automated Data Extraction** – Scrapes and structures professors’ information and publications from the university’s official website.
* **Persian Text Normalization** – Cleans and standardizes Persian content (characters, spacing, digits, URLs, etc.) for consistent search quality.
* **Vector & Lexical Retrieval** – Uses **LaBSE** embeddings with **FAISS** for semantic retrieval, combined with **TF-IDF/BM25** lexical search.
* **Rank Fusion (RRF)** – Merges dense and lexical results for higher accuracy and robustness.
* **LLM Integration** – Generates fluent Persian answers using **DeepSeek-R1:1.5b** served through **Ollama**.
* **FastAPI Backend** – Provides deterministic, RAG, and chat endpoints with **Server-Sent Events (SSE)** for streaming responses.
* **Modern Frontend** – Built with **Next.js + TypeScript + TailwindCSS + shadcn/ui**, offering real-time streaming chat UI.

---

## 🧩 System Architecture

1. **Data Layer:**
   Scraped data is normalized, structured into JSONL corpus files, and uniquely identified (`prof_id`, `pub_id`).

2. **Embedding & Indexing:**
   Documents are embedded using `sentence-transformers/LaBSE` and indexed with **FAISS** for vector search.

3. **Retrieval & Routing:**

   * **Track A (Name-based):** Deterministic retrieval for specific professors.
   * **Track B (Topic-based):** Hybrid semantic + lexical retrieval with RRF.

4. **Answer Generation:**
   Retrieved documents are passed to **LLM (DeepSeek-R1:1.5b)** for concise Persian responses, enriched with source citations.

5. **Frontend:**
   Interactive chat interface with real-time streaming, conversation management, and expandable “Suggested Sources” panel.

---

## 🔧 Tech Stack

| Layer      | Technology                                  |
| ---------- | ------------------------------------------- |
| Backend    | FastAPI, FAISS, TF-IDF, BM25                |
| Frontend   | Next.js, TypeScript, TailwindCSS, shadcn/ui |
| LLM        | Ollama + DeepSeek-R1:1.5b                        |
| Embeddings | LaBSE (Multilingual Sentence Transformer)   |
| Storage    | JSONL Corpus, Local Artifacts               |
| Tools      | Python, Node.js, SSE (Streaming Responses)  |

---

## ⚙️ Setup Instructions

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Environment Variables

Create a `.env` file in the frontend:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## 🧩 Code Modules Overview

### **block2_normalize.py — Persian Text Normalization**

Cleans and standardizes all extracted Persian text to ensure consistent indexing and search quality.
Includes:

* Converting Arabic to Persian letters
* Removing diacritics, elongations, and invisible characters
* Managing `ZWNJ` and spaces
* Converting Persian/Arabic digits to Latin digits
* Protecting and restoring URLs and email addresses

🧠 *Goal: make all text machine-readable and uniform for vectorization.*

---

### **block3_lookup.py — Deterministic Name-Based Lookup**

Handles exact, rule-based retrieval of professor information.
When a user query clearly mentions a professor’s name (e.g. *“email of Dr. Naderi”*), this block returns:

* Verified metadata (email, homepage, department, etc.)
* The professor’s full list of publications

🧠 *Goal: fast and reliable name-based responses without using the RAG pipeline.*

---

### **block4_router.py — Query Routing Logic**

Analyzes the incoming query to determine its intent and routes it to the appropriate processing path:

* **Track A (deterministic):** Name-based or fact lookup
* **Track B (hybrid):** Semantic or topic-based question

🧠 *Goal: decide whether to run a direct lookup or trigger the RAG retrieval pipeline.*

---

### **block5_build_corpus.py — Corpus Construction**

Builds the core text corpus from normalized JSON data of professors and publications.
Creates:

* Stable identifiers (`prof_id`, `pub_id`)
* Document records (`publication` / `chunk`)
* Optional text chunking with overlap for long entries
* Statistics file for text length and coverage

🧠 *Goal: prepare a clean, structured corpus ready for embedding and FAISS indexing.*

---

### **block7_smoke_test.py — System Sanity Check**

Performs quick functional tests to ensure that all modules, indexes, and data artifacts are working correctly.
Checks:

* Corpus readability and key presence
* FAISS index loading and vector dimension match
* Sample query retrieval and LLM connection

🧠 *Goal: verify the integrity of the entire pipeline before deployment.*

---

### **block8_hybrid.py — Hybrid Retrieval and Rank Fusion**

Implements the hybrid retrieval mechanism combining:

* Dense retrieval (FAISS + LaBSE)
* Lexical retrieval (TF-IDF / BM25)
* Rank fusion using **Reciprocal Rank Fusion (RRF)**

Applies thresholding and re-ranking to produce the most relevant sources for the LLM.

🧠 *Goal: deliver the best of both semantic and keyword-based search for robust RAG results.*


## 🧠 Example Queries

| Input (Persian)            | Output                                            |
| -------------------------- | ------------------------------------------------- |
| "مقالات دکتر صدوقی چیست؟"  | List of Dr. Sadoughi’s publications               |
| "اساتید حوزه پردازش تصویر" | Professors and related papers in image processing |
| "ایمیل دکتر نادری"         | Deterministic email lookup for Dr. Naderi         |

---

## 📈 Future Work

* Expand corpus to other universities and research domains
* Improve semantic reranking and prompt optimization
* Integrate translation layer for bilingual Q&A (FA/EN)
* Deploy via Docker or cloud-based inference API

