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
