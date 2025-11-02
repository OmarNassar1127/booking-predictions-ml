"""
Prediction service that handles battery predictions with dynamic updates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .battery_predictor import BatteryPredictionModel
from .timeline_manager import VehicleTimelineManager
from ..features.feature_engineer import PredictionFeatureBuilder
from ..utils.logger import logger


class BatteryPredictionService:
    """
    Main service for battery prediction with dynamic updates
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        historical_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize prediction service

        Args:
            model_path: Path to trained model (will load if provided)
            historical_data: Historical booking data for timeline management
        """

        self.model = None
        self.timeline_manager = None
        self.feature_builder = None
        self.historical_data = historical_data

        if model_path:
            self.load_model(model_path)

        if historical_data is not None:
            self._initialize_timeline_manager(historical_data)

    def load_model(self, model_path: str):
        """Load trained model from disk"""
        logger.info(f"Loading model from {model_path}")
        self.model = BatteryPredictionModel.load(model_path)

        # Initialize feature builder if we have historical data
        if self.historical_data is not None:
            self.feature_builder = PredictionFeatureBuilder(
                self.model.feature_engineer,
                self.historical_data
            )

    def _initialize_timeline_manager(self, historical_data: pd.DataFrame):
        """Initialize timeline manager with historical data"""
        self.timeline_manager = VehicleTimelineManager(historical_data)
        logger.info("Timeline manager initialized")

    def predict_battery_at_start(
        self,
        vehicle_id: str,
        user_id: str,
        booking_start_time: datetime,
        booking_id: Optional[str] = None,
        update_timeline: bool = True
    ) -> Dict:
        """
        Predict battery percentage at start of a booking

        Args:
            vehicle_id: Vehicle ID
            user_id: User ID
            booking_start_time: When the booking will start
            booking_id: Booking ID (optional, for tracking)
            update_timeline: If True, add this booking to timeline

        Returns:
            Dictionary with prediction results
        """

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.timeline_manager is None:
            raise ValueError("Timeline manager not initialized. Provide historical_data.")

        # Get last booking before this one
        last_booking = self.timeline_manager.get_last_booking_before(
            vehicle_id,
            booking_start_time,
            include_future=True
        )

        # Get intermediate bookings
        if last_booking:
            intermediate_bookings = self.timeline_manager.get_intermediate_bookings(
                vehicle_id,
                last_booking.get('ends_at', last_booking['starts_at']),
                booking_start_time
            )
        else:
            intermediate_bookings = []

        # Build features for prediction
        prediction_df = self.feature_builder.build_features_for_prediction(
            vehicle_id=vehicle_id,
            user_id=user_id,
            booking_start_time=booking_start_time,
            intermediate_bookings=intermediate_bookings
        )

        # Make prediction
        result_df = self.model.predict(prediction_df, return_confidence=True)

        prediction = result_df['predicted_battery_at_start'].iloc[0]
        lower_bound = result_df['prediction_lower_bound'].iloc[0]
        upper_bound = result_df['prediction_upper_bound'].iloc[0]
        confidence = result_df['prediction_confidence'].iloc[0]

        # Prepare result
        result = {
            'booking_id': booking_id,
            'vehicle_id': vehicle_id,
            'user_id': user_id,
            'booking_start_time': booking_start_time.isoformat(),
            'predicted_battery_percentage': float(prediction),
            'confidence_interval': {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'confidence_level': float(confidence)
            },
            'last_known_battery': float(last_booking['battery_at_end']) if last_booking else None,
            'time_since_last_booking_hours': float(
                (booking_start_time - last_booking.get('ends_at', last_booking['starts_at'])).total_seconds() / 3600
            ) if last_booking else None,
            'intermediate_bookings_count': len(intermediate_bookings),
            'timestamp': datetime.now().isoformat()
        }

        # Update timeline if requested
        if update_timeline and booking_id:
            self.timeline_manager.add_future_booking(
                vehicle_id=vehicle_id,
                user_id=user_id,
                booking_id=booking_id,
                starts_at=booking_start_time,
                predicted_battery_at_start=prediction
            )

        return result

    def predict_batch(
        self,
        bookings: List[Dict],
        update_timeline: bool = False
    ) -> List[Dict]:
        """
        Predict battery for multiple bookings

        Args:
            bookings: List of booking dictionaries with keys:
                     vehicle_id, user_id, booking_start_time, booking_id (optional)
            update_timeline: If True, add all bookings to timeline

        Returns:
            List of prediction results
        """

        results = []

        for booking in bookings:
            try:
                result = self.predict_battery_at_start(
                    vehicle_id=booking['vehicle_id'],
                    user_id=booking['user_id'],
                    booking_start_time=booking['booking_start_time'],
                    booking_id=booking.get('booking_id'),
                    update_timeline=update_timeline
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for booking {booking.get('booking_id')}: {e}")
                results.append({
                    'booking_id': booking.get('booking_id'),
                    'error': str(e),
                    'vehicle_id': booking['vehicle_id']
                })

        return results

    def handle_new_booking_created(
        self,
        vehicle_id: str,
        user_id: str,
        booking_id: str,
        booking_start_time: datetime,
        booking_end_time: datetime,
        actual_battery_at_start: Optional[float] = None
    ) -> Dict:
        """
        Handle when a new booking is created (updates affected predictions)

        Args:
            vehicle_id: Vehicle ID
            user_id: User ID
            booking_id: Booking ID
            booking_start_time: Booking start time
            booking_end_time: Booking end time
            actual_battery_at_start: Actual battery if known

        Returns:
            Dictionary with updated predictions for affected bookings
        """

        # Get affected bookings (those that come after this one)
        affected_bookings = self.timeline_manager.get_affected_bookings(
            vehicle_id,
            booking_start_time
        )

        # Predict battery for the new booking
        new_booking_prediction = self.predict_battery_at_start(
            vehicle_id=vehicle_id,
            user_id=user_id,
            booking_start_time=booking_start_time,
            booking_id=booking_id,
            update_timeline=True
        )

        # Re-predict for all affected bookings
        updated_predictions = []

        for affected in affected_bookings:
            try:
                # Remove old prediction
                self.timeline_manager.remove_future_booking(
                    vehicle_id,
                    affected['booking_id']
                )

                # Create new prediction
                new_prediction = self.predict_battery_at_start(
                    vehicle_id=vehicle_id,
                    user_id=affected['user_id'],
                    booking_start_time=affected['starts_at'],
                    booking_id=affected['booking_id'],
                    update_timeline=True
                )

                updated_predictions.append(new_prediction)

            except Exception as e:
                logger.error(f"Error updating prediction for {affected['booking_id']}: {e}")

        return {
            'new_booking': new_booking_prediction,
            'affected_bookings_count': len(affected_bookings),
            'updated_predictions': updated_predictions
        }

    def predict_from_current_state(
        self,
        vehicle_id: str,
        user_id: str,
        booking_start_time: datetime,
        current_battery_level: float,
        current_timestamp: Optional[datetime] = None,
        intermediate_bookings: Optional[List[Dict]] = None,
        booking_id: Optional[str] = None,
        last_booking_had_charging: bool = False
    ) -> Dict:
        """
        Predict battery from CURRENT real-time state (not historical)

        This method handles the realistic scenario where:
        - You know the current battery level RIGHT NOW
        - There may be scheduled bookings between now and target time
        - You need to estimate battery changes through those bookings

        Args:
            vehicle_id: Vehicle ID
            user_id: User making the target booking
            booking_start_time: Target prediction time
            current_battery_level: Current battery % (from vehicle telemetry)
            current_timestamp: Current time (defaults to now)
            intermediate_bookings: Scheduled bookings between now and target, format:
                [{"starts_at": datetime, "ends_at": datetime, "user_id": optional, "charging_at_end": optional bool}]
            booking_id: Optional booking ID for tracking
            last_booking_had_charging: Whether the last completed booking had charging_at_end = true

        Returns:
            Dictionary with prediction results including cascade steps
        """

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Validate inputs
        if not (0 <= current_battery_level <= 100):
            raise ValueError(f"current_battery_level must be between 0-100, got {current_battery_level}")

        if current_timestamp is None:
            current_timestamp = datetime.now()

        if booking_start_time <= current_timestamp:
            raise ValueError("booking_start_time must be in the future")

        # Get vehicle stats for estimation
        vehicle_data = self.historical_data[self.historical_data['vehicle_id'] == vehicle_id]

        if len(vehicle_data) == 0:
            raise ValueError(f"No historical data found for vehicle {vehicle_id}")

        # Get vehicle patterns from tracking database (includes learned charging rate)
        from ..database.tracking import get_tracker
        tracker = get_tracker()
        with tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT avg_drain_rate_per_hour, avg_charging_rate_per_hour
                FROM vehicle_patterns
                WHERE vehicle_id = ?
            """, (vehicle_id,))
            pattern_row = cursor.fetchone()

        if pattern_row and pattern_row['avg_drain_rate_per_hour']:
            avg_drain_per_hour = pattern_row['avg_drain_rate_per_hour']
            avg_charging_rate_per_hour = pattern_row['avg_charging_rate_per_hour'] or 25.0
        else:
            # Fallback: Calculate from historical data
            vehicle_data_with_drain = vehicle_data[vehicle_data['battery_at_start'] > vehicle_data['battery_at_end']].copy()
            if len(vehicle_data_with_drain) > 0:
                vehicle_data_with_drain['duration_hours'] = (
                    vehicle_data_with_drain['ends_at'] - vehicle_data_with_drain['starts_at']
                ).dt.total_seconds() / 3600
                vehicle_data_with_drain['drain_per_hour'] = (
                    (vehicle_data_with_drain['battery_at_start'] - vehicle_data_with_drain['battery_at_end']) /
                    vehicle_data_with_drain['duration_hours']
                )
                avg_drain_per_hour = vehicle_data_with_drain['drain_per_hour'].median()
            else:
                avg_drain_per_hour = 5.0  # Default: 5% per hour
            avg_charging_rate_per_hour = 25.0  # Default charging rate

        # Cascade prediction through intermediate bookings
        cascade_steps = []
        current_battery = current_battery_level
        current_time = current_timestamp
        had_charging = last_booking_had_charging  # Track if previous booking had charging

        # Sort intermediate bookings by time
        if intermediate_bookings:
            intermediate_bookings = sorted(intermediate_bookings, key=lambda x: x['starts_at'])
        else:
            intermediate_bookings = []

        # Process each intermediate booking
        for booking in intermediate_bookings:
            booking_start = booking['starts_at']
            booking_end = booking['ends_at']

            # 1. Gap period before booking starts (charging or idle)
            if booking_start > current_time:
                gap_hours = (booking_start - current_time).total_seconds() / 3600

                # Check if previous booking had charging
                if had_charging:
                    # Apply charging during gap
                    battery_gain = min(gap_hours * avg_charging_rate_per_hour, 100 - current_battery)
                    current_battery = min(100, current_battery + battery_gain)

                    cascade_steps.append({
                        'type': 'charging',
                        'from': current_time.isoformat(),
                        'to': booking_start.isoformat(),
                        'duration_hours': gap_hours,
                        'battery_change': battery_gain,
                        'battery_after': current_battery,
                        'charging_rate': avg_charging_rate_per_hour
                    })
                else:
                    # Apply idle drain (no charging)
                    idle_drain = gap_hours * 0.5  # 0.5% per hour idle drain
                    current_battery = max(0, current_battery - idle_drain)

                    cascade_steps.append({
                        'type': 'idle',
                        'from': current_time.isoformat(),
                        'to': booking_start.isoformat(),
                        'duration_hours': gap_hours,
                        'battery_change': -idle_drain,
                        'battery_after': current_battery
                    })

                current_time = booking_start
                had_charging = False  # Reset after gap

            # 2. During booking - estimate drain
            booking_duration_hours = (booking_end - booking_start).total_seconds() / 3600
            estimated_drain = booking_duration_hours * avg_drain_per_hour
            current_battery = max(0, current_battery - estimated_drain)

            cascade_steps.append({
                'type': 'booking',
                'from': booking_start.isoformat(),
                'to': booking_end.isoformat(),
                'duration_hours': booking_duration_hours,
                'battery_change': -estimated_drain,
                'battery_after': current_battery,
                'user_id': booking.get('user_id', 'unknown')
            })

            current_time = booking_end

            # Track if this booking has charging for next gap
            had_charging = booking.get('charging_at_end', False)

        # 3. Final gap to target time (charging or idle)
        if booking_start_time > current_time:
            final_gap_hours = (booking_start_time - current_time).total_seconds() / 3600

            # Apply charging or idle drain based on had_charging flag
            if had_charging:
                # Apply charging during final gap
                battery_gain = min(final_gap_hours * avg_charging_rate_per_hour, 100 - current_battery)
                final_battery = min(100, current_battery + battery_gain)

                cascade_steps.append({
                    'type': 'charging',
                    'from': current_time.isoformat(),
                    'to': booking_start_time.isoformat(),
                    'duration_hours': final_gap_hours,
                    'battery_change': battery_gain,
                    'battery_after': final_battery,
                    'charging_rate': avg_charging_rate_per_hour
                })
            else:
                # Apply idle drain (no charging)
                idle_drain = final_gap_hours * 0.5
                final_battery = max(0, current_battery - idle_drain)

                cascade_steps.append({
                    'type': 'idle',
                    'from': current_time.isoformat(),
                    'to': booking_start_time.isoformat(),
                    'duration_hours': final_gap_hours,
                    'battery_change': -idle_drain,
                    'battery_after': final_battery
                })
        else:
            final_battery = current_battery

        # Build result
        result = {
            'booking_id': booking_id,
            'vehicle_id': vehicle_id,
            'user_id': user_id,
            'booking_start_time': booking_start_time.isoformat(),
            'predicted_battery_percentage': float(final_battery),
            'confidence_interval': {
                'lower': max(0, final_battery - 10),
                'upper': min(100, final_battery + 10),
                'confidence_level': 0.95
            },
            'current_battery_level': current_battery_level,
            'current_timestamp': current_timestamp.isoformat(),
            'cascade_steps': cascade_steps,
            'total_intermediate_bookings': len(intermediate_bookings),
            'prediction_method': 'real_time_cascade',
            'timestamp': datetime.now().isoformat()
        }

        return result

    def get_vehicle_timeline_with_predictions(
        self,
        vehicle_id: str,
        include_historical: bool = True,
        include_future: bool = True
    ) -> List[Dict]:
        """
        Get complete timeline for a vehicle with predictions

        Args:
            vehicle_id: Vehicle ID
            include_historical: Include past bookings
            include_future: Include future bookings with predictions

        Returns:
            List of bookings with predictions
        """

        timeline = []

        if include_historical:
            historical = self.timeline_manager.get_vehicle_timeline(vehicle_id, include_future=False)
            timeline.extend(historical)

        if include_future:
            future = self.timeline_manager.future_bookings.get(vehicle_id, [])
            timeline.extend(future)

        # Sort by time
        timeline = sorted(timeline, key=lambda x: x['starts_at'])

        return timeline

    def get_prediction_stats(self) -> Dict:
        """Get statistics about predictions"""

        timeline_stats = self.timeline_manager.get_statistics()

        return {
            **timeline_stats,
            'model_loaded': self.model is not None,
            'feature_count': len(self.model.feature_names) if self.model else 0,
        }

    def retrain_model(
        self,
        new_data: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Dict:
        """
        Retrain model with new data

        Args:
            new_data: New booking data
            test_size: Test set fraction
            validation_size: Validation set fraction

        Returns:
            Training metrics
        """

        logger.info("Retraining model with new data...")

        # Combine with existing historical data if available
        if self.historical_data is not None:
            combined_data = pd.concat([self.historical_data, new_data]).drop_duplicates()
        else:
            combined_data = new_data

        # Update historical data
        self.historical_data = combined_data

        # Create new model
        self.model = BatteryPredictionModel()

        # Split and train
        train_df, val_df, test_df = self.model.prepare_data(
            combined_data,
            test_size=test_size,
            validation_size=validation_size
        )

        metrics = self.model.train(train_df, val_df)
        test_metrics = self.model.evaluate(test_df)

        # Re-initialize timeline and feature builder
        self._initialize_timeline_manager(combined_data)
        self.feature_builder = PredictionFeatureBuilder(
            self.model.feature_engineer,
            combined_data
        )

        logger.info("Model retrained successfully")

        return {
            **metrics,
            'test': test_metrics
        }


if __name__ == "__main__":
    # Test prediction service
    from ..data.data_loader import BookingDataLoader

    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "bookings.csv"
    loader = BookingDataLoader(str(data_path))
    df = loader.load()

    # Create service
    service = BatteryPredictionService(historical_data=df)

    # Train model
    logger.info("Training model...")
    model = BatteryPredictionModel()
    train_df, val_df, test_df = model.prepare_data(df)
    model.train(train_df, val_df)

    # Save and load model
    model_path = Path(__file__).parent.parent.parent / "models" / "battery_predictor.pkl"
    model.save(str(model_path))

    service.load_model(str(model_path))

    # Make a prediction
    test_vehicle = df['vehicle_id'].iloc[0]
    test_user = df['user_id'].iloc[0]
    future_time = df['starts_at'].max() + timedelta(days=7)

    result = service.predict_battery_at_start(
        vehicle_id=test_vehicle,
        user_id=test_user,
        booking_start_time=future_time,
        booking_id="TEST_001"
    )

    print("\nPrediction Result:")
    print(result)

    # Get stats
    stats = service.get_prediction_stats()
    print("\nService Statistics:")
    print(stats)
