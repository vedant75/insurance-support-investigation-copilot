from insurance_copilot.tools.guidance import (
    search_insurance_guidance,
)


def test_collision_guidance_returns_results() -> None:
    result = search_insurance_guidance(
        "What does collision coverage pay for?",
        top_k=3,
    )

    assert result.total_hits > 0
    assert len(result.hits) > 0

    combined_text = " ".join(
        hit.text.lower()
        for hit in result.hits
    )

    assert "collision" in combined_text
