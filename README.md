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
## System Architecture
The system is designed as a modular pipeline:

PDF Documents → Text Extraction → Chunking → Embeddings → Vector Database → Retriever → LLM → Response

Each module is separated to allow scalability and easy integration with different models or vector databases.

## How to Run the Project
1. Clone the repository
2. Install dependencies
   pip install -r requirements.txt
3. Place PDF documents inside the documents folder
4. Run the document loader and chunking pipeline
5. Generate embeddings and store them in OpenSearch
6. Run the retrieval and LLM module
7. Launch the Streamlit app (future interface)

## Example Use Case
This chatbot can be used for:
- Product documentation search
- Technical manuals Q&A
- Internal company knowledge base
- Customer support automation
- Document-based question answering systems
