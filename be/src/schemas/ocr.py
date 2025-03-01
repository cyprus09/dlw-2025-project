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
