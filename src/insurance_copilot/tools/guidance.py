from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from insurance_copilot.domain.models import (
    GuidanceSearchResult,
)
from insurance_copilot.retrieval.index import (
    GuidanceIndex,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "docs"
    / "processed"
    / "guidance_chunks.jsonl"
)


@lru_cache(maxsize=1)
def get_guidance_index() -> GuidanceIndex:
    return GuidanceIndex(
        CHUNKS_PATH
    )


def search_insurance_guidance(
    query: str,
    top_k: int = 5,
) -> GuidanceSearchResult:
    index = (
        get_guidance_index()
    )

    return index.search(
        query=query,
        top_k=top_k,
    )
