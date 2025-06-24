# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import RestaurantRatingPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Rating Predictor API",
    description="Predict restaurant ratings based on reviews and metadata",
    version="1.0.0"
)

# Initialize predictor (will be loaded on first request)
predictor = None

# Pydantic models for request/response
class RestaurantInput(BaseModel):
    name: str = Field(..., description="Restaurant name")
    location: str = Field(..., description="Restaurant location", example="Downtown")
    categories: str = Field(..., description="Restaurant category", example="Italian")
    price_range: str = Field(..., description="Price range", example="$$", regex="^[\$]{1,4}$")
    text: str = Field(..., description="Review text", min_length=10)

class PredictionResponse(BaseModel):
    predicted_rating: float = Field(..., description="Predicted rating (1-5)")
    confidence: str = Field(..., description="Prediction confidence level")
    review_sentiment: float = Field(..., description="Review sentiment score")
    model_used: str = Field(..., description="Model used for prediction")

class BatchPredictionRequest(BaseModel):
    restaurants: List[RestaurantInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None

def get_predictor():
    """Get or initialize the predictor"""
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
    """Load model on startup"""
    logger.info("Starting up Restaurant Rating Predictor API...")
    try:
        get_predictor()
        logger.info("API startup completed successfully!")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Restaurant Rating Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Predict rating for a single restaurant",
            "predict_batch": "/predict_batch - Predict ratings for multiple restaurants",
            "health": "/health - Check API health status",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
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
    """
    Predict rating for a single restaurant
    
    - **name**: Restaurant name
    - **location**: Restaurant location (e.g., Downtown, Suburb, Mall)
    - **categories**: Restaurant category (e.g., Italian, American, Japanese)
    - **price_range**: Price range ($, $$, $$$, $$$$)
    - **text**: Review text (minimum 10 characters)
    """
    try:
        logger.info(f"Prediction request for: {restaurant.name}")
        
        pred = get_predictor()
        
        # Convert Pydantic model to dict
        restaurant_data = restaurant.dict()
        
        # Make prediction
        result = pred.predict_single(restaurant_data)
        
        # Return response
        response = PredictionResponse(
            predicted_rating=result['predicted_rating'],
            confidence=result['confidence'],
            review_sentiment=result['review_sentiment'],
            model_used=result['model_used']
        )
        
        logger.info(f"Prediction completed: {result['predicted_rating']} stars")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict ratings for multiple restaurants
    
    Accepts a list of restaurant data and returns predictions for each.
    """
    try:
        logger.info(f"Batch prediction request for {len(request.restaurants)} restaurants")
        
        pred = get_predictor()
        
        # Convert Pydantic models to dicts
        restaurant_data_list = [restaurant.dict() for restaurant in request.restaurants]
        
        # Make batch predictions
        results = pred.predict_batch(restaurant_data_list)
        
        # Convert results to response format
        predictions = []
        successful_predictions = 0
        
        for result in results:
            if 'error' not in result:
                predictions.append(PredictionResponse(
                    predicted_rating=result['predicted_rating'],
                    confidence=result['confidence'],
                    review_sentiment=result['review_sentiment'],
                    model_used=result['model_used']
                ))
                successful_predictions += 1
            else:
                # For failed predictions, add a default response
                predictions.append(PredictionResponse(
                    predicted_rating=0.0,
                    confidence="Error",
                    review_sentiment=0.0,
                    model_used="N/A"
                ))
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=successful_predictions
        )
        
        logger.info(f"Batch prediction completed: {successful_predictions}/{len(request.restaurants)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sample_data", response_model=dict)
async def get_sample_data():
    """Get sample data for testing the API"""
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

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again later or contact support"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")