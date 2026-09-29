from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from insurance_copilot.agent.models import AgentToolSpec
from insurance_copilot.domain.models import ComplaintFilters, StatisticsGroupBy
from insurance_copilot.tools.complaints import (
    get_complaint,
    get_complaint_statistics,
    search_complaints,
)
from insurance_copilot.tools.guidance import search_insurance_guidance


class GetComplaintArgs(BaseModel):
    complaint_number: str


class SearchComplaintsArgs(BaseModel):
    filters: ComplaintFilters = Field(default_factory=ComplaintFilters)

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )


class GetComplaintStatisticsArgs(BaseModel):
    filters: ComplaintFilters = Field(default_factory=ComplaintFilters)

    group_by: StatisticsGroupBy

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )


class SearchGuidanceArgs(BaseModel):
    query: str = Field(min_length=3)

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "get_complaint": GetComplaintArgs,
    "search_complaints": SearchComplaintsArgs,
    "get_complaint_statistics": GetComplaintStatisticsArgs,
    "search_insurance_guidance": SearchGuidanceArgs,
}


_TOOL_DESCRIPTIONS = {
    "get_complaint": (
        "Retrieve one structured Texas Department of Insurance automobile "
        "complaint record using its complaint number."
    ),
    "search_complaints": (
        "Search structured TDI automobile complaint records using filters. "
        "Use this when the user asks for example or matching complaint records."
    ),
    "get_complaint_statistics": (
        "Calculate aggregated statistics over structured TDI automobile "
        "complaints, optionally using filters."
    ),
    "search_insurance_guidance": (
        "Search official TDI consumer insurance guidance. Use this for "
        "questions about coverage concepts, disputes, total loss, appraisal, "
        "or filing a complaint."
    ),
}


def get_agent_tool_specs() -> list[AgentToolSpec]:
    return [
        AgentToolSpec(
            name=name,
            description=_TOOL_DESCRIPTIONS[name],
            input_schema=model.model_json_schema(),
        )
        for name, model in _ARGUMENT_MODELS.items()
    ]


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> BaseModel:
    argument_model = _ARGUMENT_MODELS.get(tool_name)

    if argument_model is None:
        raise ValueError(f"Unknown agent tool: {tool_name}")

    return argument_model.model_validate(arguments)


def execute_agent_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> BaseModel | None:
    validated = validate_tool_arguments(
        tool_name,
        arguments,
    )

    if isinstance(validated, GetComplaintArgs):
        return get_complaint(
            validated.complaint_number,
        )

    if isinstance(validated, SearchComplaintsArgs):
        return search_complaints(
            filters=validated.filters,
            limit=validated.limit,
        )

    if isinstance(
        validated,
        GetComplaintStatisticsArgs,
    ):
        return get_complaint_statistics(
            filters=validated.filters,
            group_by=validated.group_by,
            limit=validated.limit,
        )

    if isinstance(validated, SearchGuidanceArgs):
        return search_insurance_guidance(
            query=validated.query,
            top_k=validated.top_k,
        )

    raise RuntimeError(f"Unhandled agent tool: {tool_name}")


def serialize_tool_result(
    result: BaseModel | None,
) -> dict[str, Any] | None:
    if result is None:
        return None

    return result.model_dump(
        mode="json",
    )
