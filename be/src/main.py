"""Main module for the FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import ml, ocr

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
