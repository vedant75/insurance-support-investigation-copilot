from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from insurance_copilot.domain.models import (
    GuidanceChunk,
    GuidanceHit,
    GuidanceSearchResult,
)


class SemanticGuidanceIndex:
    def __init__(
        self,
        *,
        chunks_path: Path,
        embeddings_path: Path,
        metadata_path: Path,
        model_name: str,
    ) -> None:
        self.chunks_path = chunks_path
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError("Guidance corpus contains no chunks.")

        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_embeddings()

        if len(self.embeddings) != len(self.chunks):
            raise ValueError("Embedding count does not match guidance chunk count.")

    def _load_chunks(self) -> list[GuidanceChunk]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Guidance corpus not found: {self.chunks_path}")

        chunks: list[GuidanceChunk] = []

        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line:
                    chunks.append(GuidanceChunk.model_validate_json(line))

        return chunks

    def _load_embeddings(self) -> np.ndarray:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                "Semantic embeddings not found. Run scripts/build_semantic_index.py first."
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        if metadata["model_name"] != self.model_name:
            raise ValueError(
                "Configured embedding model differs from the model used to build the index."
            )

        return np.load(self.embeddings_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> GuidanceSearchResult:
        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        top_k = max(1, min(top_k, 20))

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        hits: list[GuidanceHit] = []

        for index in ranked_indices[:top_k]:
            chunk = self.chunks[int(index)]

            hits.append(
                GuidanceHit(
                    chunk_id=chunk.chunk_id,
                    document_title=chunk.document_title,
                    section_title=chunk.section_title,
                    source_url=chunk.source_url,
                    text=chunk.text,
                    score=round(float(scores[index]), 4),
                )
            )

        return GuidanceSearchResult(
            query=query,
            total_hits=len(hits),
            hits=hits,
        )
