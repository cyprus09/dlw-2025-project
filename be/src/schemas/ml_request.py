"""Schemas for ML model requests and responses."""

from pydantic import BaseModel


class MLRequest(BaseModel):
    """Schema for ML model request."""

    input_data: str


class MLResponse(BaseModel):
    """Schema for ML model response."""

    result: str
    confidence: float
