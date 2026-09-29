from fastapi import FastAPI


app = FastAPI(
    title="Insurance Support Investigation Copilot",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Insurance Support Investigation Copilot",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}