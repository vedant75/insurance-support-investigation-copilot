from google import genai

from insurance_copilot.config import get_settings


def main() -> None:
    settings = get_settings()

    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured in .env")

    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL is not configured in .env")

    client = genai.Client(
        api_key=settings.llm_api_key,
    )

    interaction = client.interactions.create(
        model=settings.llm_model,
        input=("Reply with exactly GEMINI_OK and nothing else."),
    )

    print("Model:", settings.llm_model)
    print("Response:", interaction.output_text)

    if interaction.usage:
        print("Usage:", interaction.usage)


if __name__ == "__main__":
    main()
