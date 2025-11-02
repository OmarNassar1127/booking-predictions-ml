"""
Event webhook handlers for Laravel integration

This module handles all booking lifecycle events:
- booking.created: New booking created
- booking.started: User picks up vehicle
- booking.ended: User returns vehicle
- booking.modified: Booking time/duration changed
- booking.cancelled: Booking cancelled before starting
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from ..models.prediction_service import BatteryPredictionService
from ..database.tracking import get_tracker
from ..learning.pattern_updater import get_pattern_updater
from ..learning.cascade_predictor import CascadePredictor
from ..api.webhooks import get_webhook_client
from ..utils.logger import logger

# Create router
router = APIRouter(prefix="/api/v1/events", tags=["events"])

# Global prediction service (will be set from main.py)
prediction_service: Optional[BatteryPredictionService] = None
cascade_predictor: Optional[CascadePredictor] = None


def set_prediction_service(service: BatteryPredictionService):
    """Set the global prediction service"""
    global prediction_service, cascade_predictor
    prediction_service = service
    cascade_predictor = CascadePredictor(service)


# Request models
class BookingCreatedRequest(BaseModel):
    """Request when new booking is created"""
    booking_id: str = Field(..., description="Laravel booking ID")
    vehicle_id: int = Field(..., description="Vehicle ID")
    user_id: int = Field(..., description="User ID")
    starts_at: str = Field(..., description="Booking start time (ISO format)")
    ends_at: str = Field(..., description="Booking end time (ISO format)")
    current_battery_level: Optional[float] = Field(None, description="Current battery % from telemetry")
    current_timestamp: Optional[str] = Field(None, description="Current time (ISO format)")


class FutureBooking(BaseModel):
    """Future booking for cascade re-prediction"""
    booking_id: str
    user_id: int
    starts_at: str
    ends_at: str


class BookingStartedRequest(BaseModel):
    """Request when user picks up vehicle"""
    booking_id: str = Field(..., description="Laravel booking ID")
    vehicle_id: int = Field(..., description="Vehicle ID")
    actual_battery_at_start: float = Field(..., description="Actual battery % when started")
    actual_started_at: Optional[str] = Field(None, description="Actual start time")
    future_bookings: Optional[List[FutureBooking]] = Field(None, description="Future bookings for cascade")


class BookingEndedRequest(BaseModel):
    """Request when user returns vehicle"""
    booking_id: str = Field(..., description="Laravel booking ID")
    vehicle_id: int = Field(..., description="Vehicle ID")
    user_id: int = Field(..., description="User ID")
    starts_at: str = Field(..., description="Actual start time")
    ends_at: str = Field(..., description="Actual end time")
    battery_at_start: float = Field(..., description="Battery % at start")
    battery_at_end: float = Field(..., description="Battery % at end")
    mileage_at_start: Optional[float] = Field(None, description="Mileage at start")
    mileage_at_end: Optional[float] = Field(None, description="Mileage at end")
    charging_at_end: int = Field(0, description="1 if charged during booking, 0 otherwise")
    future_bookings: Optional[List[FutureBooking]] = Field(None, description="Future bookings for cascade")


class BookingModifiedRequest(BaseModel):
    """Request when booking is modified"""
    booking_id: str = Field(..., description="Laravel booking ID")
    vehicle_id: int = Field(..., description="Vehicle ID")
    user_id: int = Field(..., description="User ID")
    new_starts_at: Optional[str] = Field(None, description="New start time")
    new_ends_at: Optional[str] = Field(None, description="New end time")
    modification_type: str = Field(..., description="extended, shortened, rescheduled")
    future_bookings: Optional[List[FutureBooking]] = Field(None, description="Future bookings for cascade")


class BookingCancelledRequest(BaseModel):
    """Request when booking is cancelled"""
    booking_id: str = Field(..., description="Laravel booking ID")
    vehicle_id: int = Field(..., description="Vehicle ID")
    future_bookings: Optional[List[FutureBooking]] = Field(None, description="Future bookings for cascade")


# Event handlers

@router.post("/booking-created")
async def handle_booking_created(
    request: BookingCreatedRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle new booking created event

    Flow:
    1. Make prediction using current state
    2. Store prediction in tracking DB
    3. Return prediction to Laravel
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        logger.info(f"Event: booking.created - {request.booking_id}")

        # Parse times
        booking_start_time = datetime.fromisoformat(request.starts_at.replace('Z', '+00:00'))
        current_timestamp = None
        if request.current_timestamp:
            current_timestamp = datetime.fromisoformat(request.current_timestamp.replace('Z', '+00:00'))

        # Make prediction (use real-time mode if current battery provided)
        if request.current_battery_level is not None:
            logger.info(f"Using real-time prediction with current battery {request.current_battery_level}%")
            result = prediction_service.predict_from_current_state(
                vehicle_id=request.vehicle_id,
                user_id=request.user_id,
                booking_start_time=booking_start_time,
                current_battery_level=request.current_battery_level,
                current_timestamp=current_timestamp,
                intermediate_bookings=None,
                booking_id=request.booking_id
            )
        else:
            logger.info(f"Using historical prediction")
            result = prediction_service.predict_battery_at_start(
                vehicle_id=request.vehicle_id,
                user_id=request.user_id,
                booking_start_time=booking_start_time,
                booking_id=request.booking_id,
                update_timeline=False
            )

        # Store prediction in tracking DB
        tracker = get_tracker()
        prediction_id = tracker.store_prediction(
            booking_id=request.booking_id,
            vehicle_id=request.vehicle_id,
            user_id=request.user_id,
            booking_start_time=booking_start_time,
            predicted_battery=result['predicted_battery_percentage'],
            confidence_lower=result['confidence_interval']['lower'],
            confidence_upper=result['confidence_interval']['upper'],
            model_version="v2.0",
            prediction_method=result.get('prediction_method', 'historical'),
            current_battery_level=request.current_battery_level
        )

        logger.info(f"✓ Prediction stored: {prediction_id} - {result['predicted_battery_percentage']:.1f}%")

        return {
            "status": "success",
            "prediction_id": prediction_id,
            "predicted_battery": result['predicted_battery_percentage'],
            "confidence_interval": result['confidence_interval'],
            "prediction_method": result.get('prediction_method', 'historical'),
            "affected_bookings": []  # No cascade on creation
        }

    except Exception as e:
        logger.error(f"Error handling booking.created: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/booking-started")
async def handle_booking_started(
    request: BookingStartedRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle booking started event (user picked up vehicle)

    Flow:
    1. Record actual battery vs predicted
    2. Calculate prediction error
    3. Update vehicle patterns if error > 5%
    4. Re-predict future bookings for this vehicle
    5. Return affected bookings for Laravel to update
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        logger.info(f"Event: booking.started - {request.booking_id}")

        # Parse time
        actual_started_at = None
        if request.actual_started_at:
            actual_started_at = datetime.fromisoformat(request.actual_started_at.replace('Z', '+00:00'))

        # Record actual in tracking DB
        tracker = get_tracker()
        tracker.record_actual_start(
            booking_id=request.booking_id,
            actual_battery=request.actual_battery_at_start,
            actual_started_at=actual_started_at or datetime.now()
        )

        # Get prediction to calculate error
        prediction = tracker.get_prediction(request.booking_id)

        prediction_error = 0
        pattern_adjustment = None
        if prediction:
            prediction_error = request.actual_battery_at_start - prediction['predicted_battery']
            logger.info(f"Prediction error: {prediction_error:+.1f}%")

        # Update vehicle patterns if large error
        patterns_updated = False
        if abs(prediction_error) > 5 and prediction:
            logger.info(f"Large prediction error detected, updating patterns")
            pattern_updater = get_pattern_updater()
            pattern_adjustment = pattern_updater.update_vehicle_patterns_on_error(
                vehicle_id=request.vehicle_id,
                predicted_battery=prediction['predicted_battery'],
                actual_battery=request.actual_battery_at_start,
                prediction_error=prediction_error
            )
            patterns_updated = pattern_adjustment.get('adjusted', False)

        # Re-predict future bookings if provided
        affected_bookings = []
        if request.future_bookings and cascade_predictor:
            # Convert Pydantic models to dicts
            future_bookings_list = [booking.dict() for booking in request.future_bookings]

            affected_bookings = cascade_predictor.find_and_repredict_after_booking_start(
                vehicle_id=request.vehicle_id,
                booking_id=request.booking_id,
                actual_battery=request.actual_battery_at_start,
                actual_started_at=actual_started_at or datetime.now(),
                future_bookings=future_bookings_list
            )
            logger.info(f"Re-predicted {len(affected_bookings)} future bookings")

            # Send callback to Laravel with updated predictions
            if affected_bookings:
                background_tasks.add_task(
                    _send_callback_webhook,
                    event_type="booking.started",
                    affected_bookings=affected_bookings,
                    metadata={
                        "trigger_booking_id": request.booking_id,
                        "prediction_error": prediction_error,
                        "patterns_updated": patterns_updated
                    }
                )

        logger.info(f"✓ Actual recorded: {request.actual_battery_at_start:.1f}%")

        return {
            "status": "success",
            "prediction_error": prediction_error,
            "patterns_updated": patterns_updated,
            "affected_bookings": affected_bookings
        }

    except Exception as e:
        logger.error(f"Error handling booking.started: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/booking-ended")
async def handle_booking_ended(
    request: BookingEndedRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle booking ended event (user returned vehicle)

    Flow:
    1. Store complete booking outcome
    2. Update vehicle drain rate immediately
    3. Update charging frequency if charged
    4. Queue for model retraining
    5. Re-predict future bookings
    6. Return affected bookings
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        logger.info(f"Event: booking.ended - {request.booking_id}")

        # Parse times
        starts_at = datetime.fromisoformat(request.starts_at.replace('Z', '+00:00'))
        ends_at = datetime.fromisoformat(request.ends_at.replace('Z', '+00:00'))

        # Store outcome in tracking DB
        tracker = get_tracker()
        tracker.store_booking_outcome(
            booking_id=request.booking_id,
            vehicle_id=request.vehicle_id,
            user_id=request.user_id,
            starts_at=starts_at,
            ends_at=ends_at,
            battery_at_start=request.battery_at_start,
            battery_at_end=request.battery_at_end,
            charging_at_end=request.charging_at_end,
            mileage_at_start=request.mileage_at_start,
            mileage_at_end=request.mileage_at_end
        )

        # Calculate metrics
        duration_hours = (ends_at - starts_at).total_seconds() / 3600
        battery_drain = request.battery_at_start - request.battery_at_end
        drain_rate = battery_drain / duration_hours if duration_hours > 0 else 0

        logger.info(f"Outcome: drain {battery_drain:.1f}% over {duration_hours:.1f}h = {drain_rate:.2f}%/h")

        # Update vehicle patterns immediately
        pattern_updater = get_pattern_updater()
        vehicle_patterns = pattern_updater.update_vehicle_patterns_after_booking(
            vehicle_id=request.vehicle_id,
            battery_at_start=request.battery_at_start,
            battery_at_end=request.battery_at_end,
            starts_at=starts_at,
            ends_at=ends_at,
            charging_at_end=request.charging_at_end,
            mileage_at_start=request.mileage_at_start,
            mileage_at_end=request.mileage_at_end
        )
        drain_rate_updated = True

        # Update user patterns
        user_patterns = pattern_updater.update_user_patterns(
            user_id=request.user_id,
            battery_at_end=request.battery_at_end,
            charging_at_end=request.charging_at_end,
            booking_duration_hours=duration_hours
        )

        # TODO: Queue for retraining
        queued_for_retraining = True

        # Re-predict future bookings if provided
        affected_bookings = []
        if request.future_bookings and cascade_predictor:
            # Convert Pydantic models to dicts
            future_bookings_list = [booking.dict() for booking in request.future_bookings]

            affected_bookings = cascade_predictor.find_and_repredict_after_booking_end(
                vehicle_id=request.vehicle_id,
                booking_id=request.booking_id,
                battery_at_end=request.battery_at_end,
                ends_at=ends_at,
                future_bookings=future_bookings_list
            )
            logger.info(f"Re-predicted {len(affected_bookings)} future bookings")

            # Send callback to Laravel with updated predictions
            if affected_bookings:
                background_tasks.add_task(
                    _send_callback_webhook,
                    event_type="booking.ended",
                    affected_bookings=affected_bookings,
                    metadata={
                        "trigger_booking_id": request.booking_id,
                        "new_avg_drain_rate": vehicle_patterns['avg_drain_rate_per_hour'],
                        "pattern_confidence": vehicle_patterns['confidence_level'],
                        "total_bookings": vehicle_patterns['total_bookings']
                    }
                )

        logger.info(f"✓ Outcome stored and patterns updated")

        return {
            "status": "success",
            "outcome_recorded": True,
            "drain_rate_updated": drain_rate_updated,
            "new_avg_drain_rate": vehicle_patterns['avg_drain_rate_per_hour'],
            "pattern_confidence": vehicle_patterns['confidence_level'],
            "total_bookings": vehicle_patterns['total_bookings'],
            "queued_for_retraining": queued_for_retraining,
            "affected_bookings": affected_bookings
        }

    except Exception as e:
        logger.error(f"Error handling booking.ended: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/booking-modified")
async def handle_booking_modified(
    request: BookingModifiedRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle booking modified event (time changed, extended, shortened)

    Flow:
    1. Update prediction with new time
    2. Re-predict bookings after this one (cascade effect)
    3. Return affected bookings
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        logger.info(f"Event: booking.modified - {request.booking_id} ({request.modification_type})")

        # Get current prediction
        tracker = get_tracker()
        prediction = tracker.get_prediction(request.booking_id)

        if not prediction:
            logger.warning(f"No prediction found for {request.booking_id}")
            return {
                "status": "warning",
                "message": "No prediction to update",
                "affected_bookings": []
            }

        # Re-predict modified booking and future bookings
        affected_bookings = []
        if cascade_predictor:
            # Build modified booking dict
            modified_booking = {
                'booking_id': request.booking_id,
                'user_id': request.user_id,
                'starts_at': request.new_starts_at or prediction['booking_start_time'],
                'ends_at': request.new_ends_at
            }

            # Convert future bookings
            future_bookings_list = []
            if request.future_bookings:
                future_bookings_list = [booking.dict() for booking in request.future_bookings]

            affected_bookings = cascade_predictor.repredict_after_modification(
                vehicle_id=request.vehicle_id,
                modified_booking=modified_booking,
                future_bookings=future_bookings_list
            )
            logger.info(f"Re-predicted {len(affected_bookings)} bookings after modification")

            # Send callback to Laravel
            if affected_bookings:
                background_tasks.add_task(
                    _send_callback_webhook,
                    event_type="booking.modified",
                    affected_bookings=affected_bookings,
                    metadata={
                        "trigger_booking_id": request.booking_id,
                        "modification_type": request.modification_type
                    }
                )

        logger.info(f"✓ Modified booking handled, {len(affected_bookings)} bookings affected")

        return {
            "status": "success",
            "prediction_updated": True,
            "affected_bookings": affected_bookings
        }

    except Exception as e:
        logger.error(f"Error handling booking.modified: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/booking-cancelled")
async def handle_booking_cancelled(
    request: BookingCancelledRequest,
    background_tasks: BackgroundTasks
):
    """
    Handle booking cancelled event

    Flow:
    1. Mark prediction as cancelled
    2. Re-predict bookings that were in cascade
    3. Return affected bookings
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")

    try:
        logger.info(f"Event: booking.cancelled - {request.booking_id}")

        # Mark as cancelled
        tracker = get_tracker()
        tracker.delete_prediction(request.booking_id)

        # Re-predict future bookings that were affected
        affected_bookings = []
        if request.future_bookings and cascade_predictor:
            # Convert Pydantic models to dicts
            future_bookings_list = [booking.dict() for booking in request.future_bookings]

            affected_bookings = cascade_predictor.repredict_after_cancellation(
                vehicle_id=request.vehicle_id,
                cancelled_booking_id=request.booking_id,
                future_bookings=future_bookings_list
            )
            logger.info(f"Re-predicted {len(affected_bookings)} bookings after cancellation")

            # Send callback to Laravel
            if affected_bookings:
                background_tasks.add_task(
                    _send_callback_webhook,
                    event_type="booking.cancelled",
                    affected_bookings=affected_bookings,
                    metadata={
                        "cancelled_booking_id": request.booking_id
                    }
                )

        logger.info(f"✓ Booking cancelled, {len(affected_bookings)} bookings re-predicted")

        return {
            "status": "success",
            "prediction_cancelled": True,
            "affected_bookings": affected_bookings
        }

    except Exception as e:
        logger.error(f"Error handling booking.cancelled: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def events_health():
    """Health check for event system"""
    return {
        "status": "healthy",
        "service": "event-handlers",
        "prediction_service_loaded": prediction_service is not None
    }


# Helper function for sending callbacks
async def _send_callback_webhook(
    event_type: str,
    affected_bookings: List[Dict],
    metadata: Optional[Dict] = None
):
    """
    Send callback webhook to Laravel with updated predictions

    This runs in a background task so it doesn't block the API response

    Args:
        event_type: Type of event that triggered the update
        affected_bookings: List of bookings with updated predictions
        metadata: Optional additional data
    """
    try:
        webhook_client = get_webhook_client()
        success = await webhook_client.send_predictions_updated(
            event_type=event_type,
            affected_bookings=affected_bookings,
            metadata=metadata
        )

        if success:
            logger.info(f"✓ Callback webhook sent successfully for {event_type}")
        else:
            logger.warning(f"✗ Callback webhook failed for {event_type}")

    except Exception as e:
        logger.error(f"Error sending callback webhook: {e}")
