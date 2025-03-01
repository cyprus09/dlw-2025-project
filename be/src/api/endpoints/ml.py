"""ML model endpoints."""

from fastapi import APIRouter, HTTPException

from ...schemas.ml_request import MLRequest, MLResponse
from ...services.ml_service import MLService

router = APIRouter()


@router.post("/predict", response_model=MLResponse)
async def predict(request: MLRequest) -> MLResponse:
    """Process input through ML model and return prediction."""
    try:
        result, location = await MLService.process_input(request.location_name)
        return MLResponse(result=result, location=location)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
