from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from insurance_copilot.config import get_settings
from insurance_copilot.domain.models import GuidanceChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_chunks.jsonl"

EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_embeddings.npy"

METADATA_PATH = PROJECT_ROOT / "data" / "docs" / "processed" / "guidance_embeddings_metadata.json"


def load_chunks() -> list[GuidanceChunk]:
    chunks: list[GuidanceChunk] = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                chunks.append(GuidanceChunk.model_validate_json(line))

    return chunks


def main() -> None:
    settings = get_settings()

    if not settings.embedding_model:
        raise RuntimeError("EMBEDDING_MODEL is not configured.")

    chunks = load_chunks()

    model = SentenceTransformer(settings.embedding_model)

    texts = [(f"{chunk.document_title}. {chunk.section_title}. {chunk.text}") for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_name": settings.embedding_model,
                "chunk_count": len(chunks),
                "embedding_dimension": int(embeddings.shape[1]),
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Chunks: {len(chunks)}")
    print(f"Dimension: {embeddings.shape[1]}")
    print(f"Model: {settings.embedding_model}")


if __name__ == "__main__":
    main()
