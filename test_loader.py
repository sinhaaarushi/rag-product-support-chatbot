from Ingestion.document_loader import load_documents
from Processing.chunking import chunk_text

docs = load_documents("Data")

for doc in docs:
    print("File:", doc["file_name"])

    chunks = chunk_text(doc["text"])

    print("Total chunks:", len(chunks))

    print("First chunk preview:")
    print(chunks[0][:200])