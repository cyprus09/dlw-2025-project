"""ML model endpoints."""

from fastapi import APIRouter, HTTPException, Body

from src.schemas.ml_request import MLRequest, MLResponse
from src.services.ml_service import MLService
from typing import Dict, Any
import traceback
import logging
from typing import Optional
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    location: str
    year1: int = 2019
    year2: int = 2022

router = APIRouter()

@router.post("/predict")
async def analyze_location(request: AnalysisRequest = Body(...)):
    """
    Complete analysis of a location, combining carbon sequestration and satellite data.
    """
    try:
        # Use the existing analyze_location method which handles both carbon and satellite analysis
        result = await MLService.analyze_location(
            location_name=request.location,
            year1=request.year1,
            year2=request.year2
        )
        
       
    
        return result
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"Unexpected error in location analysis: {e}")
        logging.error(traceback.format_exc())
        
        # If the whole analysis fails, raise an HTTP exception
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/growth-rate")
async def get_growth_rate():
    """FastAPI endpoint to compute tree growth rate."""
    return await MLService.get_growth_rate()