# AGENTS.md

## Project Overview
DocMind-RAG is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents, process them into embeddings, store them in a vector database, and query them using an LLM.

## Architecture

Document Upload
→ Document Loader
→ Chunking
→ Embedding Generation
→ Vector Store
→ Similarity Search
→ LLM Response Generation

## Tech Stack

- Python
- Streamlit
- LangChain
- Vector Database
- LLM API
- Docker

## Project Structure

src/
- chat.py
- chunker.py
- document_loader.py
- embeddings.py
- llm.py
- retriever.py
- utils.py
- vector_store.py

uploads/
- User uploaded documents

vectorstore/
- Persistent vector database

tests/
- Test files

## Development Guidelines

- Follow PEP8 standards.
- Keep modules independent.
- Avoid hardcoded secrets.
- Use environment variables.
- Add type hints where possible.
- Write reusable functions.

## Agent Instructions

When modifying code:

1. Preserve the RAG pipeline.
2. Do not break retrieval functionality.
3. Keep chunking and embedding logic modular.
4. Update tests when changing functionality.
5. Maintain Docker compatibility.

## Running Locally

pip install -r requirements.txt

streamlit run app.py

## Future Improvements

- Hybrid search
- Multi-document conversations
- Citation support
- Authentication
- Performance optimization for large documents