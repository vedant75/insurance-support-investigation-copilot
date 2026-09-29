from fastapi import FastAPI

from insurance_copilot.api.analysis import (
    router as analysis_router,
)
from insurance_copilot.api.graph_analysis import (
    router as graph_router,
)

app = FastAPI(
    title=("InsureAssist — Insurance Complaint Intelligence Copilot"),
    version="0.2.0",
    description=(
        "Evidence-grounded complaint "
        "intelligence using Texas Department "
        "of Insurance complaint records and "
        "official consumer guidance."
    ),
)


app.include_router(analysis_router)

app.include_router(graph_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "InsureAssist",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
