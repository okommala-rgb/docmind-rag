# 🧠 DocMind RAG

**Production-ready Retrieval-Augmented Generation for large document collections**

DocMind RAG lets you upload thousands of pages across multiple file formats, index them into a persistent vector store, and chat with your documents using Google Gemini — with full source attribution, hybrid retrieval, and a clean Streamlit interface.

---

## ✨ Features

| Category | Highlights |
|---|---|
| **Document Ingestion** | PDF (PyMuPDF), DOCX, TXT, Markdown, HTML · Async batch processing · Duplicate detection · Incremental indexing |
| **Chunking** | Recursive (fast) + Semantic (embedding-aware) · Configurable size & overlap · Page reference preservation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` default · Switchable from UI · Batch generation · Persistent storage |
| **Vector Store** | ChromaDB (local persistence) · Full CRUD · Metadata filtering · Collection stats |
| **Retrieval** | Dense + Sparse (BM25) + Hybrid (RRF fusion) · TF-IDF re-ranking · MMR diversity · Confidence scores |
| **LLM** | Gemini 2.0 Flash / 1.5 Pro · Streaming responses · Citation-aware prompting · Hallucination reduction |
| **Chat** | Multi-turn with memory · Export JSON/MD/CSV · Source transparency · Streaming typing effect |
| **Tools** | Summarise document/pages · Generate quiz · Document insights · Compare two documents · Search-only mode |
| **Admin** | Document manager · Delete documents · Rebuild DB · Usage charts · Storage dashboard |
| **DevOps** | Docker + Compose · GitHub-ready · Full test suite |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Google API Key](https://aistudio.google.com/app/apikey) (free tier works)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/docmind-rag.git
cd docmind-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

docker-compose up --build

# Or plain Docker
docker build -t docmind-rag .
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key_here \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -v $(pwd)/uploads:/app/uploads \
  docmind-rag
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Google Gemini API key |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `DEFAULT_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Retrieved chunks per query |
| `DEFAULT_SIMILARITY_THRESHOLD` | `0.3` | Min similarity score |
| `DEFAULT_LLM_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `MAX_TOKENS` | `8192` | Max output tokens |
| `TEMPERATURE` | `0.1` | LLM temperature (0 = deterministic) |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max file upload size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 📁 Project Structure

```
docmind-rag/
├── app.py                  # Streamlit frontend (main entry point)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
├── docker-compose.yml      # Multi-container orchestration
├── .env.example            # Environment template
├── README.md
├── data/                   # Scratch space for processed data
├── vectorstore/            # ChromaDB persistent storage (auto-created)
├── uploads/                # Uploaded documents (auto-created)
├── logs/                   # Application logs (auto-created)
├── src/
│   ├── __init__.py
│   ├── utils.py            # Config, logging, shared helpers
│   ├── document_loader.py  # PDF/DOCX/TXT/MD/HTML parsers
│   ├── chunker.py          # Recursive + semantic chunking
│   ├── embeddings.py       # Sentence-transformers wrapper
│   ├── vector_store.py     # ChromaDB CRUD + management
│   ├── retriever.py        # Hybrid BM25+vector + re-ranking
│   ├── llm.py              # Gemini integration + prompts
│   └── chat.py             # Session management + export
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_document_loader.py
    ├── test_vector_store.py
    └── test_chat.py
```

---

## 🧪 Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pytest tests/ -v

# Run specific test file
pytest tests/test_chunker.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 💡 Example Queries

Once you've uploaded documents, try:

```
# Factual retrieval
"What are the key findings from the report?"
"Summarise the methodology section."
"What does the document say about X?"

# Comparative
"How does Chapter 2 differ from Chapter 5?"
"What changed between version 1 and version 2?"

# Synthesis
"What are the main risks mentioned across all documents?"
"List all action items from the meeting notes."

# Analytical
"What assumptions underpin the model described?"
"Are there any contradictions between the two papers?"
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Streamlit UI (app.py)           │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────▼────────┐
         │  Retriever       │
         │  ┌────────────┐  │
         │  │ Dense(vec) │  │   ChromaDB
         │  │ Sparse BM25│──┼──▶ (local)
         │  │ RRF Fusion │  │
         │  │ Re-ranking │  │
         │  └────────────┘  │
         └────────┬─────────┘
                  │  top-k chunks + scores
         ┌────────▼─────────┐
         │  LLM (Gemini)    │
         │  RAG prompt      │
         │  Citation gen    │
         │  Streaming       │
         └────────┬─────────┘
                  │
              Answer + Sources
```

---

## 🔑 Getting a Google API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy the key into your `.env` file as `GOOGLE_API_KEY=...`
4. The free tier includes generous quotas for Gemini 2.0 Flash

---

## 📸 Screenshots

> *After cloning, run the app and upload a PDF to see the interface in action.*

**Chat Interface** — streaming answers with source attribution  
**Upload & Index** — drag-and-drop with real-time progress  
**Admin Panel** — document manager with charts and storage stats  
**Document Tools** — summarise, quiz, insights, compare  

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration framework
- [ChromaDB](https://github.com/chroma-core/chroma) — vector database
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) — embeddings
- [Google Gemini](https://deepmind.google/technologies/gemini/) — language model
- [Streamlit](https://streamlit.io/) — UI framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF parsing
