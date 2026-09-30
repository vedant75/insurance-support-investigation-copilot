from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
    search_insurance_guidance_semantic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES_PATH = PROJECT_ROOT / "data" / "evals" / "retrieval_cases.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "evals" / "results" / "retrieval_benchmark.json"


def load_cases() -> list[dict]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def relevant_rank(
    result,
    relevant_sections: list[str],
) -> int | None:
    expected = {section.casefold() for section in relevant_sections}

    for rank, hit in enumerate(
        result.hits,
        start=1,
    ):
        if hit.section_title.casefold() in expected:
            return rank

    return None


def metrics(
    rows: list[dict],
    rank_field: str,
) -> dict:
    ranks = [row[rank_field] for row in rows]

    total = len(ranks)

    hit_at_1 = sum(rank == 1 for rank in ranks) / total

    hit_at_3 = sum(rank is not None and rank <= 3 for rank in ranks) / total

    reciprocal_ranks = [1 / rank if rank is not None else 0.0 for rank in ranks]

    return {
        "hit_at_1": round(
            hit_at_1,
            4,
        ),
        "hit_at_3": round(
            hit_at_3,
            4,
        ),
        "mrr": round(
            float(np.mean(reciprocal_ranks)),
            4,
        ),
    }


def hit_payload(
    result,
) -> list[dict]:
    return [
        {
            "rank": rank,
            "section": hit.section_title,
            "document": hit.document_title,
            "score": hit.score,
            "chunk_id": hit.chunk_id,
        }
        for rank, hit in enumerate(
            result.hits,
            start=1,
        )
    ]


def main() -> None:
    cases = load_cases()

    rows: list[dict] = []

    for case in cases:
        query = case["query"]

        lexical = search_insurance_guidance(
            query=query,
            top_k=3,
        )

        semantic = search_insurance_guidance_semantic(
            query=query,
            top_k=3,
        )

        lexical_rank = relevant_rank(
            lexical,
            case["relevant_sections"],
        )

        semantic_rank = relevant_rank(
            semantic,
            case["relevant_sections"],
        )

        rows.append(
            {
                "id": case["id"],
                "query": query,
                "relevant_sections": (case["relevant_sections"]),
                "tfidf_rank": (lexical_rank),
                "semantic_rank": (semantic_rank),
                "tfidf_hits": (hit_payload(lexical)),
                "semantic_hits": (hit_payload(semantic)),
            }
        )

    tfidf_metrics = metrics(
        rows,
        "tfidf_rank",
    )

    semantic_metrics = metrics(
        rows,
        "semantic_rank",
    )

    payload = {
        "evaluated_at_utc": (datetime.now(timezone.utc).isoformat()),
        "case_count": len(rows),
        "tfidf": tfidf_metrics,
        "semantic": semantic_metrics,
        "cases": rows,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("RETRIEVAL BENCHMARK")
    print("=" * 72)

    print()
    print("TF-IDF")
    print(f"Hit@1: {tfidf_metrics['hit_at_1']:.1%}")
    print(f"Hit@3: {tfidf_metrics['hit_at_3']:.1%}")
    print(f"MRR:   {tfidf_metrics['mrr']:.4f}")

    print()
    print("SEMANTIC")
    print(f"Hit@1: {semantic_metrics['hit_at_1']:.1%}")
    print(f"Hit@3: {semantic_metrics['hit_at_3']:.1%}")
    print(f"MRR:   {semantic_metrics['mrr']:.4f}")

    print()
    print("CASE RANKS")

    for row in rows:
        print(f"{row['id']}: TF-IDF={row['tfidf_rank']} Semantic={row['semantic_rank']}")

    print()
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
