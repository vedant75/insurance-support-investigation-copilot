from __future__ import annotations

from fastapi import APIRouter, Response, status

from insurance_copilot.config import get_settings
from insurance_copilot.db.connection import connect_read_only
from insurance_copilot.tools.guidance import get_guidance_index

router = APIRouter(
    tags=["system"],
)


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@router.get("/ready")
def ready(
    response: Response,
) -> dict:
    settings = get_settings()

    checks = {
        "database": False,
        "guidance_index": False,
    }

    errors: dict[str, str] = {}

    try:
        connection = connect_read_only(settings.database_path)

        try:
            connection.execute("SELECT 1").fetchone()

            checks["database"] = True

        finally:
            connection.close()

    except Exception as exc:
        errors["database"] = str(exc)

    try:
        index = get_guidance_index()

        checks["guidance_index"] = len(index.chunks) > 0

    except Exception as exc:
        errors["guidance_index"] = str(exc)

    is_ready = all(checks.values())

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": ("ready" if is_ready else "not_ready"),
        "checks": checks,
        "errors": errors,
    }
