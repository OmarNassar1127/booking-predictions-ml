"""
Pattern Learning Engine

This module updates vehicle and user patterns immediately after each booking
ends. It uses exponential moving averages to give more weight to recent behavior
while maintaining historical context.

Key Responsibilities:
- Update vehicle drain rates (per hour, per km)
- Update vehicle charging frequencies
- Update user return battery patterns
- Calculate pattern confidence scores
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from ..database.tracking import PredictionTracker, get_tracker
from ..utils.logger import logger


class PatternUpdater:
    """Updates vehicle and user behavioral patterns"""

    def __init__(self, tracker: Optional[PredictionTracker] = None):
        """
        Initialize pattern updater

        Args:
            tracker: Prediction tracker instance (uses global if not provided)
        """
        self.tracker = tracker or get_tracker()

        # Exponential moving average alpha (0.3 = 30% weight to new data, 70% to history)
        self.ema_alpha = 0.3

        # Minimum samples needed for reliable patterns
        self.min_samples_for_confidence = 5

    def update_vehicle_patterns_after_booking(
        self,
        vehicle_id: int,
        battery_at_start: float,
        battery_at_end: float,
        starts_at: datetime,
        ends_at: datetime,
        charging_at_end: int,
        mileage_at_start: Optional[float] = None,
        mileage_at_end: Optional[float] = None
    ) -> Dict:
        """
        Update vehicle patterns immediately after a booking ends

        Args:
            vehicle_id: Vehicle ID
            battery_at_start: Battery % at start
            battery_at_end: Battery % at end
            starts_at: Booking start time
            ends_at: Booking end time
            charging_at_end: 1 if charged, 0 otherwise
            mileage_at_start: Odometer at start
            mileage_at_end: Odometer at end

        Returns:
            Dictionary with updated patterns and metrics
        """
        # Calculate new metrics from this booking
        duration_hours = (ends_at - starts_at).total_seconds() / 3600
        battery_drain = battery_at_start - battery_at_end

        drain_rate_per_hour = battery_drain / duration_hours if duration_hours > 0 else 0

        drain_rate_per_km = None
        distance_km = None
        if mileage_at_start and mileage_at_end:
            distance_km = mileage_at_end - mileage_at_start
            if distance_km > 0:
                drain_rate_per_km = battery_drain / distance_km

        # Get current patterns from database
        current_patterns = self._get_vehicle_patterns(vehicle_id)

        # Update drain rate patterns using EMA
        new_avg_drain_rate_per_hour = self._update_ema(
            current_value=current_patterns.get('avg_drain_rate_per_hour'),
            new_value=drain_rate_per_hour
        )

        # Update standard deviation (for confidence intervals)
        new_std_drain_rate = self._update_std(
            current_std=current_patterns.get('std_drain_rate_per_hour'),
            current_avg=current_patterns.get('avg_drain_rate_per_hour'),
            new_avg=new_avg_drain_rate_per_hour,
            new_value=drain_rate_per_hour,
            sample_count=current_patterns.get('total_bookings', 0)
        )

        # Update drain per km if available
        new_avg_drain_rate_per_km = None
        if drain_rate_per_km is not None:
            new_avg_drain_rate_per_km = self._update_ema(
                current_value=current_patterns.get('avg_drain_rate_per_km'),
                new_value=drain_rate_per_km
            )

        # Update charging patterns
        total_charging_events = current_patterns.get('total_charging_events', 0) + charging_at_end
        total_bookings = current_patterns.get('total_bookings', 0) + 1

        new_charging_frequency = total_charging_events / total_bookings if total_bookings > 0 else 0

        # Learn charging rate if vehicle was charged
        new_avg_charging_rate = current_patterns.get('avg_charging_rate_per_hour')
        if charging_at_end == 1:
            # Calculate charging rate from this booking
            # We need to look at next booking's battery_at_start to know how much it charged
            # For now, we'll estimate using typical EV charging rates or learn from historical data

            # Try to get the next booking after this one to calculate actual charging gain
            # If we can't, use default charging rate
            charging_rate_observed = self._calculate_charging_rate_from_history(vehicle_id, ends_at)

            if charging_rate_observed is not None:
                new_avg_charging_rate = self._update_ema(
                    current_value=current_patterns.get('avg_charging_rate_per_hour'),
                    new_value=charging_rate_observed
                )
                logger.info(f"  Learned charging rate: {charging_rate_observed:.2f}%/h → avg: {new_avg_charging_rate:.2f}%/h")
            elif new_avg_charging_rate is None:
                # First time charging observed, use default
                new_avg_charging_rate = 25.0  # Default: 25% per hour (typical Level 2 charging)

        # Calculate time since last booking
        last_booking_at = current_patterns.get('last_booking_at')
        time_between_charges = None
        if last_booking_at and charging_at_end == 1:
            # This booking had charging
            time_between_charges = (ends_at - datetime.fromisoformat(last_booking_at)).total_seconds() / 3600

        new_avg_time_between_charges = self._update_ema(
            current_value=current_patterns.get('avg_time_between_charges_hours'),
            new_value=time_between_charges
        ) if time_between_charges else current_patterns.get('avg_time_between_charges_hours')

        # Store updated patterns
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO vehicle_patterns (
                    vehicle_id,
                    avg_drain_rate_per_hour,
                    avg_drain_rate_per_km,
                    std_drain_rate_per_hour,
                    avg_charging_rate_per_hour,
                    charging_frequency,
                    avg_time_between_charges_hours,
                    total_bookings,
                    total_charging_events,
                    last_booking_at,
                    updated_at,
                    patterns_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vehicle_id,
                new_avg_drain_rate_per_hour,
                new_avg_drain_rate_per_km,
                new_std_drain_rate,
                new_avg_charging_rate,  # Now properly calculated
                new_charging_frequency,
                new_avg_time_between_charges,
                total_bookings,
                total_charging_events,
                ends_at,
                datetime.now(),
                current_patterns.get('patterns_version', 0) + 1
            ))

        logger.info(f"✓ Updated patterns for vehicle {vehicle_id}:")
        logger.info(f"  Drain rate: {new_avg_drain_rate_per_hour:.3f}%/h (±{new_std_drain_rate:.3f})")
        logger.info(f"  Charging freq: {new_charging_frequency:.2%}")
        logger.info(f"  Total bookings: {total_bookings}")

        return {
            'vehicle_id': vehicle_id,
            'avg_drain_rate_per_hour': new_avg_drain_rate_per_hour,
            'avg_drain_rate_per_km': new_avg_drain_rate_per_km,
            'std_drain_rate_per_hour': new_std_drain_rate,
            'charging_frequency': new_charging_frequency,
            'avg_time_between_charges_hours': new_avg_time_between_charges,
            'total_bookings': total_bookings,
            'total_charging_events': total_charging_events,
            'confidence_level': self._calculate_confidence(total_bookings)
        }

    def update_vehicle_patterns_on_error(
        self,
        vehicle_id: int,
        predicted_battery: float,
        actual_battery: float,
        prediction_error: float
    ) -> Dict:
        """
        Update vehicle patterns when prediction error is large (>5%)

        This is called when a booking starts and the actual battery differs
        significantly from prediction. We adjust drain rate estimates.

        Args:
            vehicle_id: Vehicle ID
            predicted_battery: What we predicted
            actual_battery: What we observed
            prediction_error: Difference (actual - predicted)

        Returns:
            Dictionary with adjustment details
        """
        if abs(prediction_error) <= 5:
            # Error is acceptable, no immediate update needed
            return {'adjusted': False, 'reason': 'Error within tolerance'}

        logger.info(f"Large prediction error for vehicle {vehicle_id}: {prediction_error:+.1f}%")

        # Get current patterns
        current_patterns = self._get_vehicle_patterns(vehicle_id)

        # Calculate implied correction factor
        # If actual is higher than predicted, vehicle drains slower than we thought
        # If actual is lower than predicted, vehicle drains faster than we thought
        correction_factor = 1.0 + (prediction_error / 100)

        # Adjust drain rate (conservative adjustment - only 20% of the error)
        current_drain_rate = current_patterns.get('avg_drain_rate_per_hour', 2.0)
        adjusted_drain_rate = current_drain_rate / (correction_factor ** 0.2)

        # Update with conservative EMA (less weight to this single observation)
        new_drain_rate = self._update_ema(
            current_value=current_drain_rate,
            new_value=adjusted_drain_rate,
            alpha=0.15  # Only 15% weight to this adjustment
        )

        # Increase uncertainty (std) when we see large errors
        current_std = current_patterns.get('std_drain_rate_per_hour', 0.5)
        new_std = min(current_std * 1.1, 1.0)  # Increase by 10%, cap at 1.0

        # Store adjustment
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE vehicle_patterns
                SET avg_drain_rate_per_hour = ?,
                    std_drain_rate_per_hour = ?,
                    updated_at = ?,
                    patterns_version = patterns_version + 1
                WHERE vehicle_id = ?
            """, (new_drain_rate, new_std, datetime.now(), vehicle_id))

        logger.info(f"  Adjusted drain rate: {current_drain_rate:.3f} → {new_drain_rate:.3f}%/h")
        logger.info(f"  Increased uncertainty: {current_std:.3f} → {new_std:.3f}")

        return {
            'adjusted': True,
            'previous_drain_rate': current_drain_rate,
            'new_drain_rate': new_drain_rate,
            'previous_std': current_std,
            'new_std': new_std,
            'correction_factor': correction_factor
        }

    def update_user_patterns(
        self,
        user_id: int,
        battery_at_end: float,
        charging_at_end: int,
        booking_duration_hours: float
    ) -> Dict:
        """
        Update user behavioral patterns

        Args:
            user_id: User ID
            battery_at_end: Battery % when user returned vehicle
            charging_at_end: 1 if charged, 0 otherwise
            booking_duration_hours: How long they had the vehicle

        Returns:
            Dictionary with updated user patterns
        """
        # Get current user patterns
        current_patterns = self._get_user_patterns(user_id)

        # Update average return battery (how much battery users typically leave)
        new_avg_return_battery = self._update_ema(
            current_value=current_patterns.get('avg_return_battery'),
            new_value=battery_at_end
        )

        # Update charging behavior
        total_bookings = current_patterns.get('total_bookings', 0) + 1
        total_charges = current_patterns.get('total_charges', 0) + charging_at_end

        returns_with_charging_pct = (total_charges / total_bookings * 100) if total_bookings > 0 else 0

        # Update average booking duration
        new_avg_duration = self._update_ema(
            current_value=current_patterns.get('avg_booking_duration_hours'),
            new_value=booking_duration_hours
        )

        # Store updated patterns
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_patterns (
                    user_id,
                    avg_return_battery,
                    returns_with_charging_pct,
                    avg_booking_duration_hours,
                    total_bookings,
                    total_charges,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                new_avg_return_battery,
                returns_with_charging_pct,
                new_avg_duration,
                total_bookings,
                total_charges,
                datetime.now()
            ))

        logger.info(f"✓ Updated patterns for user {user_id}:")
        logger.info(f"  Avg return battery: {new_avg_return_battery:.1f}%")
        logger.info(f"  Charges vehicle: {returns_with_charging_pct:.1f}% of time")

        return {
            'user_id': user_id,
            'avg_return_battery': new_avg_return_battery,
            'returns_with_charging_pct': returns_with_charging_pct,
            'avg_booking_duration_hours': new_avg_duration,
            'total_bookings': total_bookings
        }

    # Helper methods

    def _calculate_charging_rate_from_history(
        self,
        vehicle_id: int,
        charging_ended_at: datetime
    ) -> Optional[float]:
        """
        Calculate charging rate by looking at next booking's battery level

        Args:
            vehicle_id: Vehicle ID
            charging_ended_at: When this booking ended (with charging)

        Returns:
            Charging rate in % per hour, or None if can't calculate
        """
        # Get the next booking after this one from booking_outcomes
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT battery_at_start, starts_at
                FROM booking_outcomes
                WHERE vehicle_id = ?
                  AND starts_at > ?
                ORDER BY starts_at ASC
                LIMIT 1
            """, (vehicle_id, charging_ended_at))
            row = cursor.fetchone()

            if not row:
                return None

            next_battery = row['battery_at_start']
            next_starts_at = datetime.fromisoformat(row['starts_at'])

            # Get current booking's end battery
            cursor.execute("""
                SELECT battery_at_end
                FROM booking_outcomes
                WHERE vehicle_id = ?
                  AND ends_at = ?
                LIMIT 1
            """, (vehicle_id, charging_ended_at))
            current_row = cursor.fetchone()

            if not current_row:
                return None

            battery_at_end = current_row['battery_at_end']

            # Calculate charging gain and time
            battery_gained = next_battery - battery_at_end
            time_gap_hours = (next_starts_at - charging_ended_at).total_seconds() / 3600

            if time_gap_hours > 0 and battery_gained > 0:
                charging_rate = battery_gained / time_gap_hours
                # Sanity check: charging rate should be reasonable (5-50% per hour)
                if 5 <= charging_rate <= 50:
                    return charging_rate

        return None

    def _get_vehicle_patterns(self, vehicle_id: int) -> Dict:
        """Get current vehicle patterns from database"""
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vehicle_patterns WHERE vehicle_id = ?", (vehicle_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return {}

    def _get_user_patterns(self, user_id: int) -> Dict:
        """Get current user patterns from database"""
        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_patterns WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return {}

    def _update_ema(
        self,
        current_value: Optional[float],
        new_value: Optional[float],
        alpha: Optional[float] = None
    ) -> Optional[float]:
        """
        Update value using exponential moving average

        Args:
            current_value: Current average
            new_value: New observation
            alpha: Weight for new value (default: self.ema_alpha)

        Returns:
            Updated average
        """
        if new_value is None:
            return current_value

        if current_value is None:
            return new_value

        alpha = alpha or self.ema_alpha
        return alpha * new_value + (1 - alpha) * current_value

    def _update_std(
        self,
        current_std: Optional[float],
        current_avg: Optional[float],
        new_avg: float,
        new_value: float,
        sample_count: int
    ) -> float:
        """
        Update standard deviation incrementally

        Uses Welford's online algorithm for numerical stability

        Args:
            current_std: Current standard deviation
            current_avg: Current average (before update)
            new_avg: New average (after update)
            new_value: New observation
            sample_count: Number of samples before this one

        Returns:
            Updated standard deviation
        """
        if current_std is None or current_avg is None:
            # First observation, use default
            return 0.5

        if sample_count < 2:
            # Not enough samples yet
            return 0.5

        # Simplified incremental std update
        # This is an approximation that works well for EMA
        variance = current_std ** 2
        deviation = new_value - new_avg
        new_variance = (1 - self.ema_alpha) * variance + self.ema_alpha * (deviation ** 2)

        return np.sqrt(new_variance)

    def _calculate_confidence(self, sample_count: int) -> str:
        """
        Calculate confidence level based on number of samples

        Args:
            sample_count: Number of observations

        Returns:
            Confidence level: 'low', 'medium', or 'high'
        """
        if sample_count < self.min_samples_for_confidence:
            return 'low'
        elif sample_count < 20:
            return 'medium'
        else:
            return 'high'

    def get_vehicle_pattern_summary(self, vehicle_id: int) -> Dict:
        """
        Get summary of vehicle patterns for display/debugging

        Args:
            vehicle_id: Vehicle ID

        Returns:
            Summary dictionary with patterns and confidence
        """
        patterns = self._get_vehicle_patterns(vehicle_id)

        if not patterns:
            return {
                'vehicle_id': vehicle_id,
                'has_patterns': False,
                'message': 'No historical data for this vehicle'
            }

        total_bookings = patterns.get('total_bookings', 0)
        confidence = self._calculate_confidence(total_bookings)

        return {
            'vehicle_id': vehicle_id,
            'has_patterns': True,
            'confidence_level': confidence,
            'total_bookings': total_bookings,
            'avg_drain_rate_per_hour': patterns.get('avg_drain_rate_per_hour'),
            'std_drain_rate_per_hour': patterns.get('std_drain_rate_per_hour'),
            'avg_drain_rate_per_km': patterns.get('avg_drain_rate_per_km'),
            'charging_frequency': patterns.get('charging_frequency'),
            'avg_time_between_charges_hours': patterns.get('avg_time_between_charges_hours'),
            'last_updated': patterns.get('updated_at'),
            'patterns_version': patterns.get('patterns_version')
        }


# Global instance
_pattern_updater = None

def get_pattern_updater() -> PatternUpdater:
    """Get global pattern updater instance"""
    global _pattern_updater
    if _pattern_updater is None:
        _pattern_updater = PatternUpdater()
    return _pattern_updater
