FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 \
    /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project

COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/evals ./data/evals

RUN uv sync \
    --locked \
    --no-dev

RUN mkdir -p \
    data/raw \
    data/processed \
    data/metadata \
    data/runtime \
    data/docs/raw \
    data/docs/processed

# Build a reproducible runtime evidence store.
RUN uv run python scripts/download_data.py \
    && uv run python scripts/build_database.py --full \
    && uv run python scripts/build_guidance_corpus.py \
    && rm -rf \
        data/raw \
        data/processed \
        data/metadata \
        data/docs/raw

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uv", "run", "uvicorn", "insurance_copilot.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]