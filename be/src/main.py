"""Main module for the FastAPI application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import ml, ocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("Starting DLW 2025 API")

app = FastAPI(
    title="DLW 2025 API", description="API for DLW 2025 ML Model", version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml.router, prefix="/api/ml", tags=["ml"])
app.include_router(ocr.router, prefix="/api/ocr", tags=["ocr"])


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "healthy", "message": "DLW 2025 API is running"}
