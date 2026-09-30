from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from insurance_copilot.config import get_settings
from insurance_copilot.domain.models import (
    GuidanceSearchResult,
)
from insurance_copilot.retrieval.index import (
    GuidanceIndex,
)
from insurance_copilot.retrieval.semantic_index import (
    SemanticGuidanceIndex,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHUNKS_PATH = PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_chunks.jsonl"


@lru_cache(maxsize=1)
def get_guidance_index() -> GuidanceIndex:
    return GuidanceIndex(CHUNKS_PATH)


def search_insurance_guidance(
    query: str,
    top_k: int = 5,
) -> GuidanceSearchResult:
    index = get_guidance_index()

    return index.search(
        query=query,
        top_k=top_k,
    )


EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_embeddings.npy"

EMBEDDINGS_METADATA_PATH = (
    PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_embeddings_metadata.json"
)


@lru_cache(maxsize=1)
def get_semantic_guidance_index() -> SemanticGuidanceIndex:
    settings = get_settings()

    if not settings.embedding_model:
        raise RuntimeError("EMBEDDING_MODEL is not configured.")

    return SemanticGuidanceIndex(
        chunks_path=CHUNKS_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=EMBEDDINGS_METADATA_PATH,
        model_name=settings.embedding_model,
    )


def search_insurance_guidance_semantic(
    query: str,
    top_k: int = 5,
) -> GuidanceSearchResult:
    return get_semantic_guidance_index().search(
        query=query,
        top_k=top_k,
    )
