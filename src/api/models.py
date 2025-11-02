"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single battery prediction"""
    vehicle_id: str = Field(..., description="Unique identifier for the vehicle")
    user_id: str = Field(..., description="Unique identifier for the user")
    booking_start_time: str = Field(..., description="ISO format datetime when booking starts")
    booking_id: Optional[str] = Field(None, description="Unique booking identifier (optional)")
    update_timeline: bool = Field(True, description="Whether to add this booking to the timeline")

    class Config:
        schema_extra = {
            "example": {
                "vehicle_id": "V001",
                "user_id": "U0042",
                "booking_start_time": "2024-12-25T14:30:00Z",
                "booking_id": "BK123456",
                "update_timeline": True
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval for prediction"""
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(..., description="Confidence level (e.g., 0.95)")


class PredictionResponse(BaseModel):
    """Response model for battery prediction"""
    booking_id: Optional[str] = Field(None, description="Booking identifier")
    vehicle_id: str = Field(..., description="Vehicle identifier")
    user_id: str = Field(..., description="User identifier")
    booking_start_time: str = Field(..., description="ISO format datetime")
    predicted_battery_percentage: float = Field(..., description="Predicted battery percentage (0-100)")
    confidence_interval: ConfidenceInterval = Field(..., description="Prediction confidence interval")
    last_known_battery: Optional[float] = Field(None, description="Last known battery level for this vehicle")
    time_since_last_booking_hours: Optional[float] = Field(None, description="Hours since last booking")
    intermediate_bookings_count: int = Field(..., description="Number of bookings between last and this one")
    timestamp: str = Field(..., description="When this prediction was made")

    class Config:
        schema_extra = {
            "example": {
                "booking_id": "BK123456",
                "vehicle_id": "V001",
                "user_id": "U0042",
                "booking_start_time": "2024-12-25T14:30:00",
                "predicted_battery_percentage": 78.5,
                "confidence_interval": {
                    "lower": 68.5,
                    "upper": 88.5,
                    "confidence_level": 0.95
                },
                "last_known_battery": 65.2,
                "time_since_last_booking_hours": 8.5,
                "intermediate_bookings_count": 0,
                "timestamp": "2024-12-20T10:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    bookings: List[PredictionRequest] = Field(..., description="List of booking prediction requests")
    update_timeline: bool = Field(False, description="Whether to add these bookings to timeline")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total_count: int = Field(..., description="Total number of predictions")
    timestamp: str = Field(..., description="When predictions were made")


class BookingCreatedRequest(BaseModel):
    """Request when a new booking is confirmed"""
    vehicle_id: str = Field(..., description="Vehicle identifier")
    user_id: str = Field(..., description="User identifier")
    booking_id: str = Field(..., description="Booking identifier")
    booking_start_time: str = Field(..., description="ISO format datetime when booking starts")
    booking_end_time: str = Field(..., description="ISO format datetime when booking ends")
    actual_battery_at_start: Optional[float] = Field(None, description="Actual battery level if known")

    class Config:
        schema_extra = {
            "example": {
                "vehicle_id": "V001",
                "user_id": "U0042",
                "booking_id": "BK123456",
                "booking_start_time": "2024-12-25T14:30:00Z",
                "booking_end_time": "2024-12-25T18:30:00Z",
                "actual_battery_at_start": 78.5
            }
        }


class BookingCreatedResponse(BaseModel):
    """Response when booking created"""
    new_booking: Dict[str, Any] = Field(..., description="Prediction for the new booking")
    affected_bookings_count: int = Field(..., description="Number of bookings affected by this update")
    updated_predictions: List[Dict[str, Any]] = Field(..., description="Updated predictions for affected bookings")


class TimelineResponse(BaseModel):
    """Response with vehicle timeline"""
    vehicle_id: str = Field(..., description="Vehicle identifier")
    timeline: List[Dict[str, Any]] = Field(..., description="List of bookings with predictions")
    total_bookings: int = Field(..., description="Total bookings in timeline")
    timestamp: str = Field(..., description="When timeline was retrieved")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of ML model")
    feature_count: int = Field(..., description="Number of features")
    total_historical_bookings: int = Field(..., description="Total historical bookings")
    total_vehicles: int = Field(..., description="Total vehicles")
    total_future_bookings: int = Field(..., description="Total future bookings tracked")
    top_features: List[Dict[str, Any]] = Field(..., description="Top important features")
    timestamp: str = Field(..., description="Current timestamp")
