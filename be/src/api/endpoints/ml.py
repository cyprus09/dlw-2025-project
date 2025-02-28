"""ML model endpoints."""

from fastapi import APIRouter

from src.schemas.ml_request import MLRequest, MLResponse
from src.services.ml_service import MLService

router = APIRouter()


@router.post("/predict", response_model=MLResponse)
async def predict(request: MLRequest) -> MLResponse:
    """Process input through ML model and return prediction."""
    result, confidence = await MLService.process_input(request.input_data)
    return MLResponse(result=result, confidence=confidence)

