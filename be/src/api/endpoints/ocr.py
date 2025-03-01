"""API endpoints for OCR (Optical Character Recognition) functionality."""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.schemas.ocr import (
    OCRResponse,
    ProjectAnalysisSchema,
    StructuredAnalysisResponse,
)
from src.services.ocr_service import process_document
from src.services.openai_service import analyze_text_with_schema

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=OCRResponse, status_code=200)
async def upload_document(file: UploadFile = File(...), file_type: str = Form(...)):
    """
    Upload a document and extract text using OCR.

    Args:
        file: The file to process
        file_type: Type of file ('image' or 'pdf')
    """
    if file_type not in ["image", "pdf"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported types: 'image', 'pdf'"
        )

    try:
        result = await process_document(file, file_type)
        return OCRResponse(filename=result["filename"], ocr_text=result["ocr_text"])
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upload-and-analyze", response_model=StructuredAnalysisResponse, status_code=200
)
async def upload_and_analyze(file: UploadFile = File(...), file_type: str = Form(...)):
    """
    Upload a document, extract text, and analyze it using a predefined schema.

    Args:
        file: The file to process
        file_type: Type of file ('image' or 'pdf')
    """
    logger.info(
        f"Starting upload_and_analyze with file: {file.filename}, type: {file_type}"
    )

    if file_type not in ["image", "pdf"]:
        logger.warning(f"Invalid file type provided: {file_type}")
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported types: 'image', 'pdf'"
        )
    try:
        # 1) OCR the file
        logger.info("Step 1: Starting OCR processing")
        ocr_result = await process_document(file, file_type)
        logger.info(
            f"OCR processing complete. Extracted {len(ocr_result['ocr_text'])} characters"
        )

        # 2) Use the more complex ProjectAnalysisSchema
        logger.info("Step 2: Starting text analysis with schema")
        analysis_result = await analyze_text_with_schema(
            ocr_result["ocr_text"], ProjectAnalysisSchema
        )
        logger.info("Text analysis complete")

        logger.info("Returning structured response")
        return StructuredAnalysisResponse(
            filename=ocr_result["filename"],
            ocr_text=ocr_result["ocr_text"],
            structured_response=analysis_result["structured_response"],
            usage=analysis_result["usage"],
        )
    except HTTPException as e:
        logger.error(f"HTTP exception in upload_and_analyze: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Exception in upload_and_analyze: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
