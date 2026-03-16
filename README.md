# AI Product Support Chatbot

This project is an attempt to build an AI chatbot that can answer questions from product documentation. The goal is to make it easier to search through long technical documents and quickly find relevant information.

Instead of manually reading through large PDFs or manuals, the chatbot retrieves the most relevant sections from the documents and generates an answer based on that context.

The system uses a Retrieval-Augmented Generation (RAG) approach.

## How It Works

The pipeline follows these steps:

1. Load the document (PDF)
2. Extract text from the document
3. Split the text into smaller chunks
4. Convert the chunks into embeddings
5. Store the embeddings in a vector database (OpenSearch)
6. Retrieve the most relevant chunks when a user asks a question
7. Generate the final response using a language model

## Project Structure

Ingestion  
- document_loader.py  
Loads and extracts text from PDF documents

Processing  
- chunking.py  
Splits extracted text into smaller chunks

Embeddings  
Code for generating vector embeddings (to be implemented)

Retrieval  
Handles vector search and retrieving relevant chunks

LLM  
Responsible for generating final responses

App  
Future chatbot interface

test_loader.py  
Used for testing document loading and chunking

## Tech Stack

Python  
Sentence Transformers  
OpenSearch  
HuggingFace Transformers  
Streamlit

## Current Progress

Completed:
- Document loader
- Document chunking pipeline

Next Steps:
- Embedding generation
- OpenSearch indexing
- Retrieval pipeline
- LLM integration
- Chatbot interface

## Goal

The goal of this project is to build a chatbot that can answer questions directly from documentation and help users quickly find the information they need.