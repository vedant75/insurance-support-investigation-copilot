from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
)

QUERIES = [
    ("What does TDI say about total loss claims?"),
    ("What happens if I disagree with the insurer's claim decision?"),
    ("How can a consumer file an insurance complaint?"),
    ("What does collision coverage pay for?"),
]


def main() -> None:
    for query in QUERIES:
        print()
        print("=" * 70)
        print(query)
        print("=" * 70)

        result = search_insurance_guidance(
            query=query,
            top_k=3,
        )

        for rank, hit in enumerate(
            result.hits,
            start=1,
        ):
            print(f"\n[{rank}] {hit.chunk_id}")

            print(f"Document: {hit.document_title}")

            print(f"Section:  {hit.section_title}")

            print(f"Score:    {hit.score}")

            print(f"URL:      {hit.source_url}")

            preview = hit.text[:500]

            print(f"\n{preview}")


if __name__ == "__main__":
    main()
