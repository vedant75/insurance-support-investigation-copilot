from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
    search_insurance_guidance_semantic,
)


def test_semantic_retrieval_finds_total_loss_guidance():
    result = search_insurance_guidance_semantic(
        query=("The payout for my written-off vehicle seems too low."),
        top_k=3,
    )

    assert result.total_hits == 3

    sections = {hit.section_title for hit in result.hits}

    assert (
        "What if the insurance company totals my car?" in sections
        or "Resolving problems" in sections
    )


def test_collision_guidance_returns_results() -> None:
    result = search_insurance_guidance(
        "What does collision coverage pay for?",
        top_k=3,
    )

    assert result.total_hits > 0
    assert len(result.hits) > 0

    combined_text = " ".join(hit.text.lower() for hit in result.hits)

    assert "collision" in combined_text
