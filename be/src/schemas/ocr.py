"""Pydantic schemas for OCR (Optical Character Recognition) functionality."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    """Schema for an OCR response."""

    filename: str = Field(..., description="Name of the processed file")
    ocr_text: str = Field(..., description="Text extracted from the file")


class OCRAnalysisRequest(BaseModel):
    """Schema for an OCR analysis request."""

    query: str = Field(..., description="The query to analyze OCR data with")
    ocr_text: Optional[str] = Field(None, description="OCR text if already extracted")


class OCRAnalysisResponse(BaseModel):
    """Schema for an OCR analysis response."""

    query: str = Field(..., description="The original query")
    analysis: str = Field(..., description="The analysis of the OCR text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class UploadAnalyzeResponse(BaseModel):
    """Schema for a combined upload and analyze response."""

    filename: str = Field(..., description="Name of the processed file")
    ocr_text: str = Field(..., description="Text extracted from the file")
    query: str = Field(..., description="The original query")
    analysis: str = Field(..., description="The analysis of the OCR text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class StructuredAnalysisResponse(BaseModel):
    """Schema for a structured analysis response."""

    filename: str = Field(..., description="Name of the processed file")
    ocr_text: str = Field(..., description="Text extracted from the file")
    structured_response: Dict[str, Any] = Field(
        ..., description="Structured data extracted from the OCR text"
    )
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


from typing import List, Optional

from pydantic import BaseModel, Field


class ProjectLocation(BaseModel):
    """
    Geographic scope and coordinates, as well as relevant
    administrative details.
    """

    country: str = Field(..., description="Country where project is located")
    provinces: List[str] = Field(
        ..., description="List of provinces, states, or districts in the project"
    )
    total_area_ha: float = Field(
        ..., description="Total project area in hectares, as stated by the document"
    )
    forest_area_ha: float = Field(
        ..., description="Forest area in hectares, if different from total_area_ha"
    )
    geojson_link_or_coords: Optional[str] = Field(
        None,
        description="Link to or snippet of KML/GeoJSON geometry describing the project boundary if available",
    )


class ProjectActivity(BaseModel):
    """
    Each key project intervention or measure to reduce
    deforestation or enhance carbon stocks.
    """

    name: str = Field(
        ..., description="Name or label of the activity, e.g. 'Improved Agriculture'"
    )
    description: str = Field(
        ..., description="Detailed explanation of how the activity works"
    )
    estimated_emission_reductions_tCO2e: Optional[float] = Field(
        None, description="Amount of CO2e this activity alone is expected to reduce"
    )
    timeframe_yrs: Optional[int] = Field(
        None,
        description="Over how many years is this activity implemented (if specified)?",
    )


class ClaimedReductions(BaseModel):
    """
    Claimed (ex-ante or ex-post) GHG emission reductions or
    removals, plus any relevant partition by year or period.
    """

    total_claimed_tCO2e: float = Field(
        ...,
        description="Total claimed emission reductions (tonnes of CO2e) over the crediting period",
    )
    average_annual_tCO2e: float = Field(
        ..., description="Average annual emission reductions"
    )
    # If needed, breakdown by monitoring period:
    monitoring_breakdown: Optional[List[dict]] = Field(
        None,
        description="Optional yearly or periodic breakdown of claimed emissions reductions, e.g. [{year: 2012, tCO2e: 6_896_913}, ...]",
    )


class ThinkingSpace(BaseModel):
    """
    Freeform internal reasoning, potential red flags,
    anomalies in the document, or any other commentary
    from the LLM about the project’s plausibility.
    """

    observations: List[str] = Field(
        description="List of direct observations or questionable items found in the doc"
    )
    possible_inconsistencies: List[str] = Field(
        description="List of inconsistent data or contradictory statements in the doc"
    )
    suggestions_for_further_verification: List[str] = Field(
        description="Suggested next steps for data validation (e.g. 'Check satellite imagery in Binga for 2012-2014')"
    )
    final_thoughts: Optional[str] = Field(
        None, description="Wrap-up or final remarks about the project’s claims"
    )


class ProjectAnalysisSchema(BaseModel):
    """
    The top-level schema that aggregates all relevant
    structured data from the REDD+ project documentation.
    """

    project_title: str = Field(..., description="Official project name/title")
    doc_version_or_issue_date: Optional[str] = Field(
        None, description="Document version or issuance date if stated"
    )

    # High-level summary and context
    summary_description: Optional[str] = Field(
        None, description="Concise summary of what the project claims to do"
    )

    # Nested sub-sections
    location: ProjectLocation
    claimed_reductions: ClaimedReductions
    key_activities: List[ProjectActivity] = Field(
        ...,
        description="List of primary interventions or activities the project implements",
    )

    # This is our "internal reasoning" or bigger picture field
    thinkingSpace: ThinkingSpace
