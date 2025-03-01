"""API endpoints for OCR (Optical Character Recognition) functionality."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ...schemas.ocr import OCRResponse, StructuredAnalysisResponse
from ...services.ocr_service import process_document
from ...services.openai_service import analyze_text_with_schema

router = APIRouter()


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
    if file_type not in ["image", "pdf"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported types: 'image', 'pdf'"
        )

    try:
        # First upload and extract OCR
        ocr_result = await process_document(file, file_type)

        class InvoiceData(BaseModel):
            """Pydantic model for invoice data extraction."""

            summary_of_invoice: str
            invoice_number: str
            date: str
            total_amount: float
            vendor: str

        # Then analyze with the schema type
        analysis_result = await analyze_text_with_schema(
            ocr_result["ocr_text"], InvoiceData
        )

        return StructuredAnalysisResponse(
            filename=ocr_result["filename"],
            ocr_text=ocr_result["ocr_text"],
            structured_response=analysis_result["structured_response"],
            usage=analysis_result["usage"],
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
