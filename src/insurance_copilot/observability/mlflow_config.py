from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlflow

from insurance_copilot.config import get_settings


@lru_cache(maxsize=1)
def configure_mlflow() -> None:
    settings = get_settings()

    Path("data/runtime").mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment(settings.mlflow_experiment_name)
