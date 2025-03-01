"""Service for OCR (Optical Character Recognition) processing of documents and images."""

import io
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
import pytesseract
from fastapi import HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger(__name__)


async def extract_text_from_pdf(pdf_file: UploadFile) -> Dict[str, Any]:
    """Extract text and images from a PDF file using PyMuPDF (fitz).

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
            "has_images": False,
        }

        # Process each page
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            page_dict = {"page_num": page_num + 1, "text": page_text}

            # Track if we have actual text content
            if page_text.strip():
                result["has_text"] = True

            # Initialize text_content as a list if it doesn't exist
            if "text_content" not in result:
                result["text_content"] = []

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
                    "ocr_text": ocr_text,
                }
                page_images.append(image_info)

                # Add image OCR text to the page text if it contains content
                if ocr_text.strip() and len(result["text_content"]) > page_num:
                    result["text_content"][page_num][
                        "text"
                    ] += f"\n[IMAGE TEXT: {ocr_text.strip()}]"
                    result["has_text"] = True

            if page_images:
                # Initialize image_content as a list if it doesn't exist
                if "image_content" not in result:
                    result["image_content"] = []

                result["image_content"].append(
                    {"page_num": page_num + 1, "images": page_images}
                )

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
                all_links.append({"page_num": page_num + 1, "links": links})

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
    """Extract text from an image file using OCR."""
    logger.info(f"Starting extract_text_from_image for file: {image_file.filename}")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(image_file.filename)[1]
    ) as temp_file:
        shutil.copyfileobj(image_file.file, temp_file)
        temp_path = temp_file.name
        logger.info(f"Created temporary file at: {temp_path}")

    try:
        logger.info("Opening image file")
        image = Image.open(temp_path)
        logger.info(
            f"Image opened successfully. Size: {image.size}, Mode: {image.mode}"
        )

        logger.info("Running OCR with pytesseract")
        text = pytesseract.image_to_string(image)
        logger.info(f"OCR complete. Extracted {len(text)} characters")

        os.unlink(temp_path)
        logger.info("Temporary file removed")

        if not text.strip():
            logger.warning("No text was extracted from the image")
            return "No text could be extracted from this image."

        return text
    except Exception as e:
        logger.error(f"Error in extract_text_from_image: {str(e)}", exc_info=True)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Cleaned up temporary file after error")
        raise Exception(f"Error extracting text from image: {str(e)}")


async def process_document(
    file: UploadFile, file_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a document file and extract text and/or images if possible.

    Args:
        file: The uploaded file
        file_type: The type of file ('image' or 'pdf'). If not provided,
                   will try to detect from content_type.

    Returns:
        Dictionary containing the extracted content
    """
    logger.info(f"Starting process_document for file: {file.filename}")

    try:
        # Use provided file_type or detect from content_type if not provided
        detected_file_type = file_type
        if not detected_file_type:
            logger.info("File type not provided, detecting from content type")
            if file.content_type.startswith("image/"):
                detected_file_type = "image"
                logger.info(
                    f"Detected file type: image from content type {file.content_type}"
                )
            elif file.content_type == "application/pdf":
                detected_file_type = "pdf"
                logger.info("Detected file type: pdf")
            else:
                logger.warning(f"Unsupported content type: {file.content_type}")
                raise HTTPException(
                    status_code=400,
                    detail="File type not supported. Please upload an image or PDF.",
                )
        else:
            logger.info(f"Using provided file type: {detected_file_type}")

        # Process according to file type
        if detected_file_type == "image":
            logger.info("Processing image file with OCR")
            ocr_text = await extract_text_from_image(file)
            logger.info(f"Image OCR complete, extracted {len(ocr_text)} characters")
            return {
                "filename": file.filename,
                "type": "image",
                "ocr_text": ocr_text,  # Keep the original field name for compatibility
            }
        elif detected_file_type == "pdf":
            logger.info("Processing PDF file")
            result = await extract_text_from_pdf(file)
            logger.info("PDF processing complete")

            # For backward compatibility, extract all text into a single string
            all_text = ""
            if "text_content" in result:
                logger.info(
                    f"Extracting text from {len(result['text_content'])} PDF pages"
                )
                for page in result["text_content"]:
                    all_text += page["text"] + "\n\n"

            logger.info(f"Total extracted text from PDF: {len(all_text)} characters")
            # Return with both new structured data and the original ocr_text field
            return {
                "filename": file.filename,
                "type": "pdf",
                "ocr_text": all_text,  # For compatibility with existing code
                "content": result,
            }
        else:
            logger.warning(f"Unsupported file type: {detected_file_type}")
            raise HTTPException(
                status_code=400,
                detail="File type not supported. Please use 'image' or 'pdf'.",
            )
    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
