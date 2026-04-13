"""
OpenSearch client: create kNN index, index chunks, run vector search.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urlparse

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

import config


def _parse_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 9200)
    return host, port


def get_client() -> OpenSearch:
    """Build an OpenSearch client from config.OPENSEARCH_URL."""
    host, port = _parse_host(config.OPENSEARCH_URL)
    use_ssl = port == 443
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=use_ssl,
        verify_certs=use_ssl,
        ssl_show_warn=False,
    )


def index_mapping_body() -> dict[str, Any]:
    """Mapping for kNN vector field + metadata stored with each chunk."""
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                # Minimal kNN mapping; compatible with default OpenSearch kNN engine.
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.EMBEDDING_DIMENSION,
                },
                "text": {"type": "text"},
                "document_name": {"type": "keyword"},
                "document_type": {"type": "keyword"},
                "weight": {"type": "float"},
                "chunk_index": {"type": "integer"},
            }
        },
    }


def ensure_index(client: OpenSearch | None = None) -> None:
    """Create the index if it does not exist."""
    client = client or get_client()
    if client.indices.exists(index=config.INDEX_NAME):
        return
    client.indices.create(index=config.INDEX_NAME, body=index_mapping_body())


def index_chunk(
    body: dict[str, Any],
    doc_id: str | None = None,
    client: OpenSearch | None = None,
) -> str:
    """
    Index one chunk document. Returns the OpenSearch document id used.
    Expected keys in body: text, embedding, document_name, document_type, weight, chunk_index.
    """
    client = client or get_client()
    ensure_index(client)
    _id = doc_id or str(uuid.uuid4())
    client.index(index=config.INDEX_NAME, id=_id, body=body, refresh=True)
    return _id


def knn_search(
    query_vector: list[float],
    k: int,
    client: OpenSearch | None = None,
) -> list[dict[str, Any]]:
    """
    Run approximate kNN search; returns OpenSearch hits (with _score and _source).
    """
    client = client or get_client()
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }
    resp = client.search(index=config.INDEX_NAME, body=body)
    return resp.get("hits", {}).get("hits", [])


def delete_index(client: OpenSearch | None = None) -> None:
    """Drop the whole index (useful for dev resets)."""
    client = client or get_client()
    try:
        client.indices.delete(index=config.INDEX_NAME)
    except NotFoundError:
        pass


def get_index_stats(client: OpenSearch | None = None) -> dict[str, Any]:
    """
    Return lightweight index stats for diagnostics.
    """
    client = client or get_client()
    if not client.indices.exists(index=config.INDEX_NAME):
        return {"index_exists": False, "index_name": config.INDEX_NAME}

    count_resp = client.count(index=config.INDEX_NAME)
    stats_resp = client.indices.stats(index=config.INDEX_NAME)
    index_stats = (
        stats_resp.get("indices", {})
        .get(config.INDEX_NAME, {})
        .get("total", {})
        .get("store", {})
    )
    return {
        "index_exists": True,
        "index_name": config.INDEX_NAME,
        "document_count": int(count_resp.get("count", 0)),
        "store_size_bytes": int(index_stats.get("size_in_bytes", 0)),
    }
