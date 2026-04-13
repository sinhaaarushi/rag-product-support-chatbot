"""
End-to-end indexing: PDF -> chunks -> embeddings -> OpenSearch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import config
from Embeddings.embedding_generator import embed_texts
from Ingestion.document_loader import load_pdf
from Processing.chunking import chunk_text
from Retrieval.retriever import infer_document_type_from_name
from Utils.logging_utils import get_logger
from VectorStore.faiss_store import add_embeddings

logger = get_logger("pipeline")


def _weight_for_type(document_type: str) -> float:
    return config.DOCUMENT_WEIGHTS.get(
        document_type, config.DOCUMENT_WEIGHTS["default"]
    )


def run_indexing_pipeline(
    file_path: str | Path,
    document_type: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Load one PDF, chunk, embed batches, index each chunk with metadata.

    If document_type is None, it is inferred from the filename (see retriever.infer_document_type_from_name).
    Returns a small summary dict for API responses.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported for indexing")

    doc_name = path.name
    if not config.is_valid_doc_name(doc_name):
        msg = (
            f"Document name does not match convention `{config.DOC_NAME_PATTERN}`: {doc_name}"
        )
        if config.ENFORCE_DOC_NAME_PATTERN:
            raise ValueError(msg)
        logger.warning(msg)
    dtype = document_type or infer_document_type_from_name(doc_name)
    weight = _weight_for_type(dtype)

    text = load_pdf(path)
    if progress_callback:
        progress_callback({"stage": "loaded_pdf", "document_name": doc_name})
    chunks = chunk_text(
        text,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    if progress_callback:
        progress_callback({"stage": "chunked_text", "total_chunks": len(chunks)})
    if not chunks:
        return {
            "document_name": doc_name,
            "document_type": dtype,
            "chunks_indexed": 0,
            "message": "No text extracted; nothing indexed.",
        }

    embeddings = embed_texts(chunks)
    if progress_callback:
        progress_callback({"stage": "generated_embeddings", "total_embeddings": len(embeddings)})
    records: list[dict] = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        body = {
            "text": chunk,
            "embedding": emb,
            "document_name": doc_name,
            "document_type": dtype,
            "weight": weight,
            "chunk_index": i,
        }
        records.append(body)
        indexed = i + 1
        if progress_callback and (indexed % 10 == 0 or indexed == len(chunks)):
            progress_callback(
                {
                    "stage": "indexing_chunks",
                    "indexed_chunks": indexed,
                    "total_chunks": len(chunks),
                }
            )
    indexed = add_embeddings(records)

    result = {
        "document_name": doc_name,
        "document_type": dtype,
        "weight": weight,
        "chunks_indexed": indexed,
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
    }
    if progress_callback:
        progress_callback({"stage": "completed", **result})
    logger.info(
        "Indexed document=%s type=%s chunks=%s", doc_name, dtype, indexed
    )
    return result


def run_batch_indexing_pipeline(
    folder_path: str | Path,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Index all PDFs in a folder. Document type is inferred per filename.
    """
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_paths = sorted(p for p in folder.glob("*.pdf") if p.is_file())
    total = len(pdf_paths)
    if total == 0:
        return {
            "folder": str(folder),
            "documents_found": 0,
            "documents_indexed": 0,
            "total_chunks_indexed": 0,
            "results": [],
            "message": "No PDF files found.",
        }

    results: list[dict] = []
    total_chunks = 0
    for i, pdf_path in enumerate(pdf_paths, start=1):
        if not config.is_valid_doc_name(pdf_path.name):
            msg = (
                f"Batch file name does not match convention `{config.DOC_NAME_PATTERN}`: "
                f"{pdf_path.name}"
            )
            if config.ENFORCE_DOC_NAME_PATTERN:
                raise ValueError(msg)
            logger.warning(msg)
        if progress_callback:
            progress_callback(
                {
                    "stage": "batch_indexing_document",
                    "current_document": i,
                    "total_documents": total,
                    "document_name": pdf_path.name,
                }
            )
        doc_result = run_indexing_pipeline(pdf_path)
        results.append(doc_result)
        total_chunks += int(doc_result.get("chunks_indexed", 0))

    summary = {
        "folder": str(folder),
        "documents_found": total,
        "documents_indexed": len(results),
        "total_chunks_indexed": total_chunks,
        "results": results,
    }
    if progress_callback:
        progress_callback({"stage": "batch_completed", **summary})
    return summary
