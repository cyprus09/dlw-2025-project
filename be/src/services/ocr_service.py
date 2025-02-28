import os
import tempfile
import shutil
import fitz  # PyMuPDF
# from pymupdf4llm import extract_text
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
from PIL import Image
import pytesseract


async def extract_text_from_image(image_file: UploadFile) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_file: The uploaded image file
        
    Returns:
        Extracted text from the image
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(image_file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Perform OCR on the image
        img = Image.open(temp_path)
        ocr_text = pytesseract.image_to_string(img)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return ocr_text
    except Exception as e:
        # Make sure to clean up in case of error
        os.unlink(temp_path)
        raise Exception(f"Error extracting text from image: {str(e)}")

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Extract text from a PDF file using pymupdf (fitz).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(pdf_file.file, temp_file)
        temp_path = temp_file.name

    try:
        doc = fitz.open(temp_path)  # Open PDF
        text = "\n".join([page.get_text("text") for page in doc])  # Extract text from each page
        doc.close()

        os.unlink(temp_path)  # Remove temp file

        if not text.strip():
            raise ValueError("No text extracted. PDF may contain only images.")

        return text
    except Exception as e:
        os.unlink(temp_path)  # Ensure cleanup
        raise Exception(f"Error extracting text from PDF: {str(e)}")



async def process_document(file: UploadFile) -> Dict[str, Any]:
    """
    Process a document file and extract text if possible.
    
    Args:
        file: The uploaded file
        
    Returns:
        Dictionary containing the filename and extracted text
    """
    try:
        # Check if the file is an image
        if file.content_type.startswith('image/'):
            ocr_text = await extract_text_from_image(file)
        elif file.content_type == "application/pdf":
            ocr_text = await extract_text_from_pdf(file)
        else:
            raise HTTPException(
                status_code=400, 
                detail="File type not supported for OCR. Please upload an image / pdf."
            )
            
        return {
            "filename": file.filename,
            "ocr_text": ocr_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))