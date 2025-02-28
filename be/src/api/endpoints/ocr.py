from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends

from ...schemas.ocr import OCRResponse, UploadAnalyzeResponse
from ...services.ocr_service import process_document
from ...services.openai_service import analyze_text_with_query

router = APIRouter()


@router.post("/upload", response_model=OCRResponse, status_code=200)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and extract text using OCR.
    """
    try:
        result = await process_document(file)
        return OCRResponse(
            filename=result["filename"],
            ocr_text=result["ocr_text"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/upload-and-analyze", response_model=UploadAnalyzeResponse, status_code=200)
async def upload_and_analyze(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Upload a document, extract text, and analyze it in one step.
    """
    try:
        # First upload and extract OCR
        ocr_result = await process_document(file)
        
        # Then analyze with the query
        analysis_result = await analyze_text_with_query(query, ocr_result["ocr_text"])
        
        return UploadAnalyzeResponse(
            filename=ocr_result["filename"],
            ocr_text=ocr_result["ocr_text"],
            query=query,
            analysis=analysis_result["response"],
            usage=analysis_result["usage"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-pdf", response_model=OCRResponse, status_code=200)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and extract text using pymupdf4llm.
    """
    try:
        result = await process_document(file)
        return OCRResponse(
            filename=result["filename"],
            ocr_text=result["ocr_text"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-pdf", response_model=UploadAnalyzeResponse, status_code=200)
async def analyze_pdf(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Upload a PDF, extract text, and analyze it using OpenAI.
    """
    try:
        pdf_result = await process_document(file)
        analysis_result = await analyze_text_with_query(query, pdf_result["ocr_text"])

        return UploadAnalyzeResponse(
            filename=pdf_result["filename"],
            ocr_text=pdf_result["ocr_text"],
            query=query,
            analysis=analysis_result["response"],
            usage=analysis_result["usage"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))