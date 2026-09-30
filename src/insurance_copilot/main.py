from fastapi import FastAPI

from insurance_copilot.api.agent_analysis import (
    router as agent_router,
)
from insurance_copilot.api.analysis import (
    router as analysis_router,
)
from insurance_copilot.api.graph_analysis import (
    router as graph_router,
)
from insurance_copilot.api.system import (
    router as system_router,
)

app = FastAPI(
    title=("InsureAssist — Insurance Complaint Intelligence Copilot"),
    version="0.3.0",
    description=(
        "Evidence-grounded complaint intelligence "
        "using Texas Department of Insurance "
        "complaint records and official "
        "consumer guidance."
    ),
)


app.include_router(system_router)

app.include_router(analysis_router)

app.include_router(graph_router)

app.include_router(agent_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "InsureAssist",
        "version": "0.4.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }
