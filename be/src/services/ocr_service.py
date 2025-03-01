import os
import tempfile
import shutil
import fitz  # PyMuPDF
from fastapi import UploadFile, HTTPException
from typing import Dict, Any, List, Tuple
import pytesseract
from PIL import Image
import io
import numpy as np

async def extract_text_from_pdf(pdf_file: UploadFile) -> Dict[str, Any]:
    """
    Extract text and images from a PDF file using PyMuPDF (fitz).
    Processes both the text layer and performs OCR on images within the PDF.
    
    Returns:
        Dictionary containing extracted text and image information
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(pdf_file.file, temp_file)
        temp_path = temp_file.name

    try:
        doc = fitz.open(temp_path)
        result = {
            "text_content": [],
            "image_content": [],
            "page_count": len(doc),
            "has_text": False,
            "has_images": False
        }
        
        # Process each page
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            page_dict = {
                "page_num": page_num + 1,
                "text": page_text
            }
            
            # Track if we have actual text content
            if page_text.strip():
                result["has_text"] = True
                
            result["text_content"].append(page_dict)
            
            # Extract images using the get_images method
            image_list = page.get_images(full=True)
            
            if image_list:
                result["has_images"] = True
                
            page_images = []
            for img_index, img in enumerate(image_list):
                xref = img[0]  # Image reference
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Perform OCR on the image
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                
                image_info = {
                    "image_index": img_index,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "ocr_text": ocr_text
                }
                page_images.append(image_info)
                
                # Add image OCR text to the page text if it contains content
                if ocr_text.strip():
                    result["text_content"][page_num]["text"] += f"\n[IMAGE TEXT: {ocr_text.strip()}]"
                    result["has_text"] = True
            
            if page_images:
                result["image_content"].append({
                    "page_num": page_num + 1,
                    "images": page_images
                })
        
        # Extract document structure - using low-level interface for more details
        toc = doc.get_toc()
        if toc:
            result["toc"] = toc
        
        # Extract document metadata
        result["metadata"] = doc.metadata
        
        # Extract links
        all_links = []
        for page_num, page in enumerate(doc):
            links = page.get_links()
            if links:
                all_links.append({
                    "page_num": page_num + 1,
                    "links": links
                })
        
        if all_links:
            result["links"] = all_links
            
        # Close the document
        doc.close()
        
        # Clean up
        os.unlink(temp_path)
        
        # If no text was found in the document
        if not result["has_text"] and not result["has_images"]:
            raise ValueError("No text or images could be extracted from the PDF.")
            
        return result
    
    except Exception as e:
        # Ensure cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Error processing PDF: {str(e)}")


async def extract_text_from_image(image_file: UploadFile) -> str:
    """
    Extract text from an image file using OCR.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_file:
        shutil.copyfileobj(image_file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        image = Image.open(temp_path)
        text = pytesseract.image_to_string(image)
        os.unlink(temp_path)
        
        if not text.strip():
            return "No text could be extracted from this image."
        
        return text
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Error extracting text from image: {str(e)}")


async def process_document(file: UploadFile) -> Dict[str, Any]:
    """
    Process a document file and extract text and/or images if possible.
    
    Args:
        file: The uploaded file
        
    Returns:
        Dictionary containing the extracted content
    """
    try:
        # Check file type and process accordingly
        if file.content_type.startswith('image/'):
            ocr_text = await extract_text_from_image(file)
            return {
                "filename": file.filename,
                "type": "image",
                "ocr_text": ocr_text  # Keep the original field name for compatibility
            }
        elif file.content_type == "application/pdf":
            result = await extract_text_from_pdf(file)
            
            # For backward compatibility, extract all text into a single string
            all_text = ""
            if "text_content" in result:
                for page in result["text_content"]:
                    all_text += page["text"] + "\n\n"
            
            # Return with both new structured data and the original ocr_text field
            return {
                "filename": file.filename,
                "type": "pdf",
                "ocr_text": all_text,  # Add this for compatibility with your existing code
                "content": result
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="File type not supported. Please upload an image or PDF."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Advanced PDF processing with vector representation
async def process_pdf_with_vectors(pdf_file: UploadFile, vector_model=None) -> Dict[str, Any]:
    """
    Process a PDF and extract both text content and vector representations.
    This function uses the extract_text_from_pdf function and then applies
    a vector model to create embeddings of the text content.
    
    Args:
        pdf_file: The uploaded PDF file
        vector_model: Optional model for creating vector embeddings
        
    Returns:
        Dictionary with extracted text, images, and vector representations
    """
    # First extract all text and image content
    extraction_result = await extract_text_from_pdf(pdf_file)
    
    # If a vector model is provided, create embeddings
    if vector_model:
        # Combine all text from the document
        all_text = ""
        for page in extraction_result["text_content"]:
            all_text += page["text"] + "\n\n"
            
        # Create vector embedding for the entire document
        document_embedding = vector_model.encode(all_text)
        extraction_result["document_vector"] = document_embedding.tolist()
        
        # Create vectors for each page
        page_vectors = []
        for page in extraction_result["text_content"]:
            if page["text"].strip():
                page_vector = vector_model.encode(page["text"])
                page_vectors.append({
                    "page_num": page["page_num"],
                    "vector": page_vector.tolist()
                })
        
        if page_vectors:
            extraction_result["page_vectors"] = page_vectors
    
    return {
        "filename": pdf_file.filename,
        "type": "pdf_vectors",
        "content": extraction_result
    }