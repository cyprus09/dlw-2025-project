"""Schemas for ML model requests and responses."""

from typing import Dict

from pydantic import BaseModel


class MLRequest(BaseModel):
    """Schema for ML model request."""

    location_name: str


class MLResponse(BaseModel):
    """Schema for ML model response."""

    result: Dict
    location: Dict
