"""
Cascade Re-prediction System

When a vehicle's battery state changes (actual differs from predicted, or booking completes),
this module re-predicts all affected future bookings for that vehicle.

Key Responsibilities:
- Find future bookings affected by state change
- Re-predict them in chronological order (cascade)
- Track which predictions were updated
- Return affected booking predictions for Laravel to update
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from ..models.prediction_service import BatteryPredictionService
from ..database.tracking import get_tracker
from ..utils.logger import logger


class CascadePredictor:
    """Handles cascade re-prediction of affected bookings"""

    def __init__(self, prediction_service: BatteryPredictionService):
        """
        Initialize cascade predictor

        Args:
            prediction_service: Battery prediction service instance
        """
        self.prediction_service = prediction_service
        self.tracker = get_tracker()

    def cascade_from_current_state(
        self,
        vehicle_id: int,
        current_battery_level: float,
        current_timestamp: datetime,
        future_bookings: List[Dict],
        last_booking_had_charging: bool = False
    ) -> List[Dict]:
        """
        Re-predict all future bookings starting from current battery state

        This is called when:
        - A booking just started and actual battery differs from predicted
        - A booking just ended and we have a new known battery level
        - Patterns were updated and predictions need refreshing

        Args:
            vehicle_id: Vehicle ID
            current_battery_level: Current/actual battery percentage
            current_timestamp: Current time (or time of last known state)
            future_bookings: List of future bookings to re-predict, each with:
                - booking_id: Laravel booking ID
                - user_id: User ID
                - starts_at: Booking start time (datetime or ISO string)
                - ends_at: Booking end time (datetime or ISO string)
                - charging_at_end: Optional - 1 if had charging, 0 otherwise (for completed bookings)
            last_booking_had_charging: Whether the last completed booking had charging_at_end = true

        Returns:
            List of updated predictions with booking_id and new predicted_battery
        """
        if not future_bookings:
            logger.info(f"No future bookings to re-predict for vehicle {vehicle_id}")
            return []

        logger.info(f"Cascade re-predicting {len(future_bookings)} bookings for vehicle {vehicle_id}")
        logger.info(f"Starting from battery: {current_battery_level:.1f}% at {current_timestamp}")

        # Sort bookings chronologically
        sorted_bookings = sorted(
            future_bookings,
            key=lambda b: self._parse_datetime(b['starts_at'])
        )

        updated_predictions = []
        simulated_battery = current_battery_level
        simulated_timestamp = current_timestamp
        had_charging = last_booking_had_charging

        for i, booking in enumerate(sorted_bookings):
            try:
                # Parse times
                starts_at = self._parse_datetime(booking['starts_at'])
                ends_at = self._parse_datetime(booking['ends_at'])

                # Calculate intermediate bookings (all previous bookings in this cascade)
                intermediate_bookings = []
                for prev_booking in sorted_bookings[:i]:
                    intermediate_bookings.append({
                        'starts_at': self._parse_datetime(prev_booking['starts_at']),
                        'ends_at': self._parse_datetime(prev_booking['ends_at']),
                        'user_id': prev_booking.get('user_id'),
                        'charging_at_end': prev_booking.get('charging_at_end')
                    })

                # Make prediction from current simulated state
                result = self.prediction_service.predict_from_current_state(
                    vehicle_id=vehicle_id,
                    user_id=booking['user_id'],
                    booking_start_time=starts_at,
                    current_battery_level=simulated_battery,
                    current_timestamp=simulated_timestamp,
                    intermediate_bookings=intermediate_bookings if intermediate_bookings else None,
                    booking_id=booking['booking_id'],
                    last_booking_had_charging=had_charging
                )

                # Store updated prediction
                self.tracker.store_prediction(
                    booking_id=booking['booking_id'],
                    vehicle_id=vehicle_id,
                    user_id=booking['user_id'],
                    booking_start_time=starts_at,
                    predicted_battery=result['predicted_battery_percentage'],
                    confidence_lower=result['confidence_interval']['lower'],
                    confidence_upper=result['confidence_interval']['upper'],
                    model_version="v2.0",
                    prediction_method='real_time_cascade',
                    current_battery_level=simulated_battery
                )

                updated_predictions.append({
                    'booking_id': booking['booking_id'],
                    'predicted_battery': result['predicted_battery_percentage'],
                    'confidence_interval': result['confidence_interval'],
                    'starts_at': starts_at.isoformat(),
                    'prediction_method': 'real_time_cascade'
                })

                logger.info(f"  [{i+1}/{len(sorted_bookings)}] Booking {booking['booking_id']}: {result['predicted_battery_percentage']:.1f}%")

                # Update simulated state for next booking
                # Get the cascade steps to find the final battery after this booking
                cascade_steps = result.get('cascade_steps', [])
                if cascade_steps:
                    # Find the last step's battery level
                    last_step = cascade_steps[-1]
                    simulated_battery = last_step.get('battery_after', result['predicted_battery_percentage'])
                else:
                    # Fallback: assume some drain during booking (rough approximation)
                    simulated_battery = result['predicted_battery_percentage'] * 0.85

                simulated_timestamp = ends_at

                # Track if this booking had charging for next iteration
                had_charging = booking.get('charging_at_end', False)

            except Exception as e:
                logger.error(f"Error re-predicting booking {booking['booking_id']}: {e}")
                continue

        logger.info(f"✓ Cascade complete: updated {len(updated_predictions)} predictions")

        return updated_predictions

    def find_and_repredict_after_booking_start(
        self,
        vehicle_id: int,
        booking_id: str,
        actual_battery: float,
        actual_started_at: datetime,
        future_bookings: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Re-predict future bookings after a booking starts

        This is called from the booking-started event when we have the actual
        battery level at start.

        Args:
            vehicle_id: Vehicle ID
            booking_id: Booking that just started
            actual_battery: Actual battery at start
            actual_started_at: Actual start time
            future_bookings: Optional list of future bookings (if not provided, returns empty)

        Returns:
            List of updated predictions
        """
        if not future_bookings:
            # No future bookings provided - Laravel would need to send these
            logger.info(f"No future bookings provided for cascade after {booking_id}")
            return []

        logger.info(f"Re-predicting bookings after {booking_id} started with {actual_battery:.1f}%")

        return self.cascade_from_current_state(
            vehicle_id=vehicle_id,
            current_battery_level=actual_battery,
            current_timestamp=actual_started_at,
            future_bookings=future_bookings
        )

    def find_and_repredict_after_booking_end(
        self,
        vehicle_id: int,
        booking_id: str,
        battery_at_end: float,
        ends_at: datetime,
        future_bookings: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Re-predict future bookings after a booking ends

        This is called from the booking-ended event when we know the final
        battery level.

        Args:
            vehicle_id: Vehicle ID
            booking_id: Booking that just ended
            battery_at_end: Battery level at end
            ends_at: End time
            future_bookings: Optional list of future bookings

        Returns:
            List of updated predictions
        """
        if not future_bookings:
            logger.info(f"No future bookings provided for cascade after {booking_id}")
            return []

        logger.info(f"Re-predicting bookings after {booking_id} ended with {battery_at_end:.1f}%")

        return self.cascade_from_current_state(
            vehicle_id=vehicle_id,
            current_battery_level=battery_at_end,
            current_timestamp=ends_at,
            future_bookings=future_bookings
        )

    def repredict_after_modification(
        self,
        vehicle_id: int,
        modified_booking: Dict,
        future_bookings: List[Dict]
    ) -> List[Dict]:
        """
        Re-predict after a booking is modified (time changed)

        Args:
            vehicle_id: Vehicle ID
            modified_booking: The booking that was modified with new times
            future_bookings: Future bookings that may be affected

        Returns:
            List of updated predictions (includes the modified booking)
        """
        logger.info(f"Re-predicting after booking {modified_booking['booking_id']} was modified")

        # Get the most recent known state for this vehicle
        # This would ideally come from Laravel or be stored in our DB
        # For now, we'll try to get it from recent predictions
        current_state = self._get_latest_vehicle_state(vehicle_id)

        if not current_state:
            logger.warning(f"No recent state found for vehicle {vehicle_id}, using defaults")
            current_state = {
                'battery_level': 80.0,  # Default assumption
                'timestamp': datetime.now()
            }

        # Include the modified booking in the cascade
        all_bookings = [modified_booking] + future_bookings

        return self.cascade_from_current_state(
            vehicle_id=vehicle_id,
            current_battery_level=current_state['battery_level'],
            current_timestamp=current_state['timestamp'],
            future_bookings=all_bookings
        )

    def repredict_after_cancellation(
        self,
        vehicle_id: int,
        cancelled_booking_id: str,
        future_bookings: List[Dict]
    ) -> List[Dict]:
        """
        Re-predict future bookings after a booking is cancelled

        When a booking is cancelled, the battery won't drain during that
        previously-scheduled time, so future bookings need updating.

        Args:
            vehicle_id: Vehicle ID
            cancelled_booking_id: Booking that was cancelled
            future_bookings: Remaining future bookings

        Returns:
            List of updated predictions
        """
        if not future_bookings:
            logger.info(f"No future bookings to re-predict after cancellation of {cancelled_booking_id}")
            return []

        logger.info(f"Re-predicting {len(future_bookings)} bookings after {cancelled_booking_id} was cancelled")

        # Get current state
        current_state = self._get_latest_vehicle_state(vehicle_id)

        if not current_state:
            current_state = {
                'battery_level': 80.0,
                'timestamp': datetime.now()
            }

        return self.cascade_from_current_state(
            vehicle_id=vehicle_id,
            current_battery_level=current_state['battery_level'],
            current_timestamp=current_state['timestamp'],
            future_bookings=future_bookings
        )

    # Helper methods

    def _parse_datetime(self, dt) -> datetime:
        """Parse datetime from various formats"""
        if isinstance(dt, datetime):
            return dt
        elif isinstance(dt, str):
            return datetime.fromisoformat(dt.replace('Z', '+00:00'))
        else:
            raise ValueError(f"Cannot parse datetime from {type(dt)}")

    def _get_latest_vehicle_state(self, vehicle_id: int) -> Optional[Dict]:
        """
        Get the latest known battery state for a vehicle

        This tries to find the most recent completed or started booking
        to determine current battery level.

        Args:
            vehicle_id: Vehicle ID

        Returns:
            Dictionary with battery_level and timestamp, or None
        """
        # Try to get most recent outcome
        recent_outcomes = self.tracker.get_recent_outcomes(vehicle_id=vehicle_id, limit=1)

        if not recent_outcomes.empty:
            latest = recent_outcomes.iloc[0]
            return {
                'battery_level': latest['battery_at_end'],
                'timestamp': pd.to_datetime(latest['ends_at'])
            }

        # Try to get most recent started booking
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT actual_battery, actual_started_at
                FROM predictions
                WHERE vehicle_id = ?
                  AND status = 'completed'
                  AND actual_battery IS NOT NULL
                ORDER BY actual_started_at DESC
                LIMIT 1
            """, (vehicle_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'battery_level': row['actual_battery'],
                    'timestamp': datetime.fromisoformat(row['actual_started_at'])
                }

        # No recent data found
        return None
