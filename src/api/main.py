"""
FastAPI REST API for battery prediction service

This API allows Laravel backend to communicate with the ML model
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BookingCreatedRequest,
    BookingCreatedResponse,
    TimelineResponse,
    HealthResponse,
    ModelInfoResponse
)

from ..models.prediction_service import BatteryPredictionService
from ..data.data_loader import BookingDataLoader
from ..utils.logger import logger
from ..utils.config_loader import config

# Import event router
from . import events

# Initialize FastAPI app
app = FastAPI(
    title="Battery Prediction API",
    description="REST API for predicting electric car battery levels at booking start",
    version="2.0.0"  # Updated for event-driven system
)

# CORS middleware for Laravel integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include event router
app.include_router(events.router)

# Global prediction service instance
prediction_service: Optional[BatteryPredictionService] = None


# Dependency for API authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if authentication is enabled"""
    auth_enabled = config.get('api.authentication.enabled', False)

    if auth_enabled:
        expected_key = config.get('api.authentication.api_key', '')
        if x_api_key != expected_key:
            raise HTTPException(status_code=403, detail="Invalid API key")

    return True


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup"""
    global prediction_service

    try:
        logger.info("Starting API server...")

        # Load historical data
        data_path = config.get('data.raw_data_path', 'data/raw/bookings.csv')
        data_path = Path(data_path)

        if data_path.exists():
            logger.info(f"Loading historical data from {data_path}")
            loader = BookingDataLoader(str(data_path))
            df = loader.load()
        else:
            logger.warning(f"Data file not found at {data_path}")
            df = None

        # Load trained model
        model_path = config.get('model.save_path', 'models/battery_predictor.pkl')
        model_path = Path(model_path)

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            prediction_service = BatteryPredictionService(
                model_path=str(model_path),
                historical_data=df
            )
            logger.info("Prediction service initialized successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
            logger.warning("API will start but predictions will fail until model is loaded")
            prediction_service = BatteryPredictionService(historical_data=df)

        # Pass prediction service to event handlers
        events.set_prediction_service(prediction_service)
        logger.info("Event handlers initialized with prediction service")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Battery Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "predict_batch": "/api/v1/predict/batch",
            "booking_created": "/api/v1/booking/created",
            "timeline": "/api/v1/vehicle/{vehicle_id}/timeline",
            "model_info": "/api/v1/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service is not None and prediction_service.model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, dependencies=[Depends(verify_api_key)])
async def get_model_info():
    """Get model information and statistics"""

    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats = prediction_service.get_prediction_stats()

    # Get feature importance
    feature_importance = prediction_service.model.get_feature_importance(top_n=10)

    return ModelInfoResponse(
        model_type="LightGBM",
        feature_count=stats['feature_count'],
        total_historical_bookings=stats['total_historical_bookings'],
        total_vehicles=stats['total_vehicles'],
        total_future_bookings=stats['total_future_bookings'],
        top_features=feature_importance.to_dict('records'),
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_battery(request: PredictionRequest):
    """
    Predict battery percentage at start of a booking

    This endpoint supports two prediction modes:

    1. **Historical mode** (default): Predicts from last known booking in database
       - Use when you don't have real-time battery data
       - Just pass vehicle_id, user_id, booking_start_time

    2. **Real-time mode** (recommended for production): Predicts from current battery level
       - Use when you have real-time vehicle telemetry
       - Pass current_battery_level to enable this mode
       - Optionally include intermediate_bookings for more accurate predictions

    Example real-time request:
    ```json
    {
      "vehicle_id": "V001",
      "user_id": "U0042",
      "booking_start_time": "2024-12-25T19:00:00Z",
      "current_battery_level": 65.5,
      "intermediate_bookings": [
        {"starts_at": "2024-12-24T13:00:00Z", "ends_at": "2024-12-24T15:00:00Z"}
      ]
    }
    ```
    """

    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse booking start time
        booking_start_time = datetime.fromisoformat(request.booking_start_time.replace('Z', '+00:00'))

        # Check if real-time prediction is requested
        if request.current_battery_level is not None:
            # REAL-TIME MODE: Use current battery level
            logger.info(f"Real-time prediction for vehicle {request.vehicle_id} with current battery {request.current_battery_level}%")

            # Parse current timestamp
            current_timestamp = None
            if request.current_timestamp:
                current_timestamp = datetime.fromisoformat(request.current_timestamp.replace('Z', '+00:00'))

            # Parse intermediate bookings
            intermediate_bookings = None
            if request.intermediate_bookings:
                intermediate_bookings = [
                    {
                        'starts_at': datetime.fromisoformat(b.starts_at.replace('Z', '+00:00')),
                        'ends_at': datetime.fromisoformat(b.ends_at.replace('Z', '+00:00')),
                        'user_id': b.user_id
                    }
                    for b in request.intermediate_bookings
                ]

            # Make real-time prediction
            result = prediction_service.predict_from_current_state(
                vehicle_id=request.vehicle_id,
                user_id=request.user_id,
                booking_start_time=booking_start_time,
                current_battery_level=request.current_battery_level,
                current_timestamp=current_timestamp,
                intermediate_bookings=intermediate_bookings,
                booking_id=request.booking_id
            )

        else:
            # HISTORICAL MODE: Use last known booking from database
            logger.info(f"Historical prediction for vehicle {request.vehicle_id}")

            result = prediction_service.predict_battery_at_start(
                vehicle_id=request.vehicle_id,
                user_id=request.user_id,
                booking_start_time=booking_start_time,
                booking_id=request.booking_id,
                update_timeline=request.update_timeline
            )

            # Add prediction method to result
            result['prediction_method'] = 'historical'

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_battery_batch(request: BatchPredictionRequest):
    """
    Predict battery percentage for multiple bookings

    Useful for batch processing or showing predictions for multiple vehicles
    """

    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to list of dicts
        bookings = []
        for booking_req in request.bookings:
            bookings.append({
                'vehicle_id': booking_req.vehicle_id,
                'user_id': booking_req.user_id,
                'booking_start_time': datetime.fromisoformat(
                    booking_req.booking_start_time.replace('Z', '+00:00')
                ),
                'booking_id': booking_req.booking_id
            })

        # Make predictions
        results = prediction_service.predict_batch(
            bookings=bookings,
            update_timeline=request.update_timeline
        )

        return BatchPredictionResponse(
            predictions=results,
            total_count=len(results),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/booking/created", response_model=BookingCreatedResponse, dependencies=[Depends(verify_api_key)])
async def booking_created(request: BookingCreatedRequest):
    """
    Handle when a new booking is created

    This updates the timeline and recalculates predictions for affected bookings.
    Call this from Laravel after a booking is confirmed.
    """

    if prediction_service is None or prediction_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        booking_start_time = datetime.fromisoformat(request.booking_start_time.replace('Z', '+00:00'))
        booking_end_time = datetime.fromisoformat(request.booking_end_time.replace('Z', '+00:00'))

        # Handle new booking
        result = prediction_service.handle_new_booking_created(
            vehicle_id=request.vehicle_id,
            user_id=request.user_id,
            booking_id=request.booking_id,
            booking_start_time=booking_start_time,
            booking_end_time=booking_end_time,
            actual_battery_at_start=request.actual_battery_at_start
        )

        return BookingCreatedResponse(**result)

    except Exception as e:
        logger.error(f"Error handling booking created: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/vehicle/{vehicle_id}/timeline", response_model=TimelineResponse, dependencies=[Depends(verify_api_key)])
async def get_vehicle_timeline(
    vehicle_id: str,
    include_historical: bool = True,
    include_future: bool = True
):
    """
    Get complete timeline for a vehicle with predictions

    Useful for displaying vehicle booking history and future predictions
    """

    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        timeline = prediction_service.get_vehicle_timeline_with_predictions(
            vehicle_id=vehicle_id,
            include_historical=include_historical,
            include_future=include_future
        )

        return TimelineResponse(
            vehicle_id=vehicle_id,
            timeline=timeline,
            total_bookings=len(timeline),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if config.get('dashboard.show_error_details', False) else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        "src.api.main:app",
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8000),
        reload=config.get('api.reload', True)
    )
