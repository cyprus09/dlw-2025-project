"""ML model endpoints."""

from fastapi import APIRouter, HTTPException

from src.schemas.ml_request import MLRequest, MLResponse
from src.services.ml_service import MLService

router = APIRouter()


@router.post("/predict", response_model=MLResponse)
async def predict(request: MLRequest) -> MLResponse:
    """Process input through ML model and return prediction."""
    try:
        result, location = await MLService.process_input(request.location_name)
        return MLResponse(result=result, location=location)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/growth-rate")
async def get_growth_rate():
    """FastAPI endpoint to compute tree growth rate."""
    return await MLService.get_growth_rate()