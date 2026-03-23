# AI Product Support Chatbot

AI-powered product support chatbot using a Retrieval-Augmented Generation (RAG) architecture. The system retrieves relevant information from product documentation and uses a language model to generate context-aware responses.

The goal of this project is to enable users to quickly find answers from large technical documentation without manually searching through PDFs or manuals.

## Architecture
The system follows a Retrieval-Augmented Generation pipeline:

User Query → Embedding → Vector Search → Retrieve Documents → LLM → Response

## Pipeline Steps
1. Load product documentation (PDF)
2. Extract text from documents
3. Split text into smaller chunks
4. Convert text chunks into embeddings
5. Store embeddings in vector database (OpenSearch)
6. Retrieve relevant chunks based on user query
7. Generate final response using language model

## Project Structure

Ingestion/
document_loader.py

Processing/
chunking.py

Embeddings/
embedding generation (planned)

Retrieval/
vector search and retrieval (planned)

LLM/
response generation (planned)

App/
chatbot interface (planned)

test_loader.py
testing document loading and chunking


## Tech Stack
- Python
- Sentence Transformers
- HuggingFace Transformers
- OpenSearch (Vector Database)
- Streamlit
- Retrieval-Augmented Generation (RAG)

## Current Progress
Completed:
- Document loader
- Document chunking pipeline
- Project architecture and modular structure

In Progress:
- Embedding generation
- OpenSearch indexing
- Retrieval pipeline
- LLM integration
- Chatbot interface

## Future Improvements
- Conversation memory
- Web interface
- Deployment on cloud
- Multi-document support

## Goal
The goal of this project is to design a scalable architecture for AI-powered document question answering systems using Retrieval-Augmented Generation.
