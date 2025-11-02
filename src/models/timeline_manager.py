"""
Vehicle timeline manager for tracking bookings and dynamic predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ..utils.logger import logger


class VehicleTimelineManager:
    """
    Manages vehicle booking timelines and handles dynamic updates
    when new bookings are added
    """

    def __init__(self, historical_bookings: pd.DataFrame):
        """
        Initialize timeline manager

        Args:
            historical_bookings: DataFrame with historical booking data
        """
        self.historical_bookings = historical_bookings.sort_values(['vehicle_id', 'starts_at'])

        # Cache: vehicle_id -> list of bookings (sorted by time)
        self.vehicle_timelines = {}

        # Cache: vehicle_id -> last known state
        self.vehicle_states = {}

        # Pending future bookings (not yet started)
        self.future_bookings = defaultdict(list)  # vehicle_id -> list of future bookings

        self._build_timelines()

    def _build_timelines(self):
        """Build timeline for each vehicle from historical data"""

        logger.info("Building vehicle timelines...")

        for vehicle_id in self.historical_bookings['vehicle_id'].unique():
            vehicle_bookings = self.historical_bookings[
                self.historical_bookings['vehicle_id'] == vehicle_id
            ].copy()

            vehicle_bookings = vehicle_bookings.sort_values('starts_at')

            self.vehicle_timelines[vehicle_id] = vehicle_bookings.to_dict('records')

            # Store last known state
            last_booking = vehicle_bookings.iloc[-1]
            self.vehicle_states[vehicle_id] = {
                'last_booking_end': last_booking['ends_at'],
                'last_battery_end': last_booking['battery_at_end'],
                'last_mileage': last_booking['mileage_at_end'],
                'total_bookings': len(vehicle_bookings)
            }

        logger.info(f"Built timelines for {len(self.vehicle_timelines)} vehicles")

    def get_vehicle_state(self, vehicle_id: str) -> Optional[Dict]:
        """
        Get current state of a vehicle

        Args:
            vehicle_id: Vehicle ID

        Returns:
            Dictionary with vehicle state or None if vehicle not found
        """
        return self.vehicle_states.get(vehicle_id)

    def get_vehicle_timeline(self, vehicle_id: str, include_future: bool = True) -> List[Dict]:
        """
        Get complete timeline for a vehicle

        Args:
            vehicle_id: Vehicle ID
            include_future: If True, include pending future bookings

        Returns:
            List of booking dictionaries, sorted by time
        """
        timeline = self.vehicle_timelines.get(vehicle_id, []).copy()

        if include_future and vehicle_id in self.future_bookings:
            timeline.extend(self.future_bookings[vehicle_id])
            timeline = sorted(timeline, key=lambda x: x['starts_at'])

        return timeline

    def add_future_booking(
        self,
        vehicle_id: str,
        user_id: str,
        booking_id: str,
        starts_at: datetime,
        predicted_battery_at_start: float,
        predicted_battery_at_end: float = None
    ):
        """
        Add a future booking to the timeline

        Args:
            vehicle_id: Vehicle ID
            user_id: User ID
            booking_id: Booking ID
            starts_at: Booking start time
            predicted_battery_at_start: Predicted battery at start
            predicted_battery_at_end: Predicted battery at end (optional)
        """

        future_booking = {
            'booking_id': booking_id,
            'vehicle_id': vehicle_id,
            'user_id': user_id,
            'starts_at': starts_at,
            'predicted_battery_at_start': predicted_battery_at_start,
            'predicted_battery_at_end': predicted_battery_at_end,
            'is_future': True
        }

        self.future_bookings[vehicle_id].append(future_booking)

        # Keep future bookings sorted
        self.future_bookings[vehicle_id] = sorted(
            self.future_bookings[vehicle_id],
            key=lambda x: x['starts_at']
        )

        logger.info(f"Added future booking {booking_id} for {vehicle_id} at {starts_at}")

    def get_intermediate_bookings(
        self,
        vehicle_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Get all bookings for a vehicle between two times

        Args:
            vehicle_id: Vehicle ID
            start_time: Start of time window
            end_time: End of time window

        Returns:
            List of bookings in the time window
        """

        timeline = self.get_vehicle_timeline(vehicle_id, include_future=True)

        intermediate = [
            booking for booking in timeline
            if start_time < booking['starts_at'] < end_time
        ]

        return sorted(intermediate, key=lambda x: x['starts_at'])

    def get_last_booking_before(
        self,
        vehicle_id: str,
        timestamp: datetime,
        include_future: bool = True
    ) -> Optional[Dict]:
        """
        Get the last booking before a given timestamp

        Args:
            vehicle_id: Vehicle ID
            timestamp: Reference timestamp
            include_future: If True, include future bookings

        Returns:
            Last booking before timestamp, or None
        """

        timeline = self.get_vehicle_timeline(vehicle_id, include_future=include_future)

        # Filter bookings that end before timestamp
        past_bookings = [
            booking for booking in timeline
            if booking.get('ends_at', booking['starts_at']) < timestamp
        ]

        if not past_bookings:
            return None

        # Return the most recent one
        return max(past_bookings, key=lambda x: x.get('ends_at', x['starts_at']))

    def remove_future_booking(self, vehicle_id: str, booking_id: str):
        """
        Remove a future booking from the timeline

        Args:
            vehicle_id: Vehicle ID
            booking_id: Booking ID to remove
        """

        if vehicle_id in self.future_bookings:
            self.future_bookings[vehicle_id] = [
                b for b in self.future_bookings[vehicle_id]
                if b['booking_id'] != booking_id
            ]

            logger.info(f"Removed future booking {booking_id} for {vehicle_id}")

    def update_future_booking(
        self,
        vehicle_id: str,
        booking_id: str,
        **updates
    ):
        """
        Update a future booking

        Args:
            vehicle_id: Vehicle ID
            booking_id: Booking ID
            **updates: Fields to update
        """

        if vehicle_id in self.future_bookings:
            for booking in self.future_bookings[vehicle_id]:
                if booking['booking_id'] == booking_id:
                    booking.update(updates)
                    logger.info(f"Updated future booking {booking_id}")
                    break

    def get_affected_bookings(
        self,
        vehicle_id: str,
        new_booking_time: datetime
    ) -> List[Dict]:
        """
        Get all future bookings that would be affected by a new booking

        Args:
            vehicle_id: Vehicle ID
            new_booking_time: Start time of new booking

        Returns:
            List of affected future bookings (those after the new booking)
        """

        if vehicle_id not in self.future_bookings:
            return []

        # Find all future bookings that start after the new booking
        affected = [
            booking for booking in self.future_bookings[vehicle_id]
            if booking['starts_at'] > new_booking_time
        ]

        return sorted(affected, key=lambda x: x['starts_at'])

    def clear_future_bookings(self, vehicle_id: Optional[str] = None):
        """
        Clear future bookings

        Args:
            vehicle_id: If provided, clear only for this vehicle; otherwise clear all
        """

        if vehicle_id:
            self.future_bookings[vehicle_id] = []
            logger.info(f"Cleared future bookings for {vehicle_id}")
        else:
            self.future_bookings.clear()
            logger.info("Cleared all future bookings")

    def get_statistics(self) -> Dict:
        """Get timeline statistics"""

        stats = {
            'total_vehicles': len(self.vehicle_timelines),
            'total_historical_bookings': len(self.historical_bookings),
            'total_future_bookings': sum(len(bookings) for bookings in self.future_bookings.values()),
            'vehicles_with_future_bookings': len([v for v in self.future_bookings.values() if v]),
            'avg_bookings_per_vehicle': np.mean([
                state['total_bookings'] for state in self.vehicle_states.values()
            ]) if self.vehicle_states else 0
        }

        return stats


if __name__ == "__main__":
    # Test timeline manager
    from pathlib import Path
    from ..data.data_loader import BookingDataLoader

    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "bookings.csv"
    loader = BookingDataLoader(str(data_path))
    df = loader.load()

    # Create timeline manager
    manager = VehicleTimelineManager(df)

    # Get stats
    stats = manager.get_statistics()
    print("\nTimeline Manager Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test vehicle state
    vehicle_id = df['vehicle_id'].iloc[0]
    state = manager.get_vehicle_state(vehicle_id)
    print(f"\nState for {vehicle_id}:")
    print(state)

    # Test adding future booking
    manager.add_future_booking(
        vehicle_id=vehicle_id,
        user_id="U0001",
        booking_id="FUTURE_001",
        starts_at=datetime.now() + timedelta(days=7),
        predicted_battery_at_start=75.0
    )

    print(f"\nFuture bookings for {vehicle_id}: {len(manager.future_bookings[vehicle_id])}")
