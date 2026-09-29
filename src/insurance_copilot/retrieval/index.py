from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from insurance_copilot.domain.models import (
    GuidanceChunk,
    GuidanceHit,
    GuidanceSearchResult,
)


class GuidanceIndex:
    def __init__(
        self,
        chunks_path: Path,
    ) -> None:
        self.chunks_path = chunks_path

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise ValueError("Guidance corpus contains no chunks.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [self._search_text(chunk) for chunk in (self.chunks)]
        )

    def _load_chunks(
        self,
    ) -> list[GuidanceChunk]:
        if not (self.chunks_path.exists()):
            raise FileNotFoundError(
                f"Guidance corpus not found: "
                f"{self.chunks_path}. "
                "Run "
                "scripts/build_guidance_corpus.py "
                "first."
            )

        chunks: list[GuidanceChunk] = []

        with self.chunks_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                chunks.append(GuidanceChunk.model_validate_json(line))

        return chunks

    @staticmethod
    def _search_text(
        chunk: GuidanceChunk,
    ) -> str:
        return f"{chunk.document_title} {chunk.section_title} {chunk.text}"

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
    ) -> GuidanceSearchResult:
        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        top_k = max(
            1,
            min(top_k, 20),
        )

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1]

        hits: list[GuidanceHit] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                continue

            chunk = self.chunks[index]

            hits.append(
                GuidanceHit(
                    chunk_id=(chunk.chunk_id),
                    document_title=(chunk.document_title),
                    section_title=(chunk.section_title),
                    source_url=(chunk.source_url),
                    text=chunk.text,
                    score=round(
                        score,
                        4,
                    ),
                )
            )

            if len(hits) >= top_k:
                break

        return GuidanceSearchResult(
            query=query,
            total_hits=len(hits),
            hits=hits,
        )
