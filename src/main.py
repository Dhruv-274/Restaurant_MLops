# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import RestaurantRatingPredictor  # Corrected import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Rating Predictor API",
    description="Predict restaurant ratings based on reviews and metadata",
    version="1.0.0"
)

predictor = None

# Pydantic models
class RestaurantInput(BaseModel):
    name: str = Field(..., description="Restaurant name")
    location: str = Field(..., description="Restaurant location", example="Downtown")
    categories: str = Field(..., description="Restaurant category", example="Italian")
    price_range: str = Field(..., description="Price range", example="$$", pattern=r"^\${1,4}$")
    text: str = Field(..., description="Review text", min_length=10)

class PredictionResponse(BaseModel):
    predicted_rating: float
    confidence: str
    review_sentiment: float
    model_used: str

class BatchPredictionRequest(BaseModel):
    restaurants: List[RestaurantInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None

# Helper to get or load model
def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = RestaurantRatingPredictor()
            predictor.load_model()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return predictor

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Restaurant Rating Predictor API...")
    try:
        get_predictor()
        logger.info("API startup completed successfully!")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Restaurant Rating Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        pred = get_predictor()
        model_info = pred.get_model_info()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_info=model_info
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_info={"error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(restaurant: RestaurantInput):
    try:
        logger.info(f"Prediction request for: {restaurant.name}")
        pred = get_predictor()
        result = pred.predict_single(restaurant.dict())
        return PredictionResponse(
            predicted_rating=result['predicted_rating'],
            confidence=result['confidence'],
            review_sentiment=result['review_sentiment'],
            model_used=result['model_used']
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    try:
        logger.info(f"Batch prediction for {len(request.restaurants)} entries")
        pred = get_predictor()
        restaurant_data = [r.dict() for r in request.restaurants]
        results = pred.predict_batch(restaurant_data)

        predictions = []
        success = 0
        for r in results:
            if 'error' not in r:
                predictions.append(PredictionResponse(**r))
                success += 1
            else:
                predictions.append(PredictionResponse(
                    predicted_rating=0.0,
                    confidence="Error",
                    review_sentiment=0.0,
                    model_used="N/A"
                ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=success
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sample_data", response_model=dict)
async def get_sample_data():
    return {
        "sample_restaurants": [
            {
                "name": "Amazing Pizza Palace",
                "location": "Downtown",
                "categories": "Italian",
                "price_range": "$$",
                "text": "Absolutely fantastic pizza! The crust was perfect, sauce was flavorful, and the service was excellent. Highly recommend this place to anyone looking for authentic Italian pizza!"
            },
            {
                "name": "Terrible Burger Shack",
                "location": "Suburb",
                "categories": "American",
                "price_range": "$",
                "text": "Worst burger I've ever had. The meat was dry and overcooked, the bun was stale, and the service was incredibly slow. Would definitely not recommend this place."
            },
            {
                "name": "Decent Sushi Spot",
                "location": "Mall",
                "categories": "Japanese",
                "price_range": "$$$",
                "text": "Pretty good sushi overall. The fish was fresh and the presentation was nice. Service was okay, nothing extraordinary but decent quality for the price."
            }
        ]
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "See /docs for valid routes"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again later"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
