"""
Prediction tracking database for monitoring accuracy and learning

This module stores:
- Predictions made by the model
- Actual outcomes when bookings complete
- Prediction errors and accuracy metrics
- Pattern learning data (drain rates, charging frequencies)
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from contextlib import contextmanager

from ..utils.logger import logger


class PredictionTracker:
    """Tracks predictions vs actuals and calculates accuracy metrics"""

    def __init__(self, db_path: str = "data/tracking/predictions.db"):
        """
        Initialize prediction tracker

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    booking_id TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,

                    -- Prediction details
                    predicted_at TIMESTAMP NOT NULL,
                    booking_start_time TIMESTAMP NOT NULL,
                    predicted_battery REAL NOT NULL,
                    confidence_lower REAL,
                    confidence_upper REAL,

                    -- Actual outcome (filled when booking starts)
                    actual_battery REAL,
                    actual_started_at TIMESTAMP,
                    prediction_error REAL,

                    -- Metadata
                    model_version TEXT,
                    prediction_method TEXT,
                    current_battery_level REAL,

                    -- Status
                    status TEXT DEFAULT 'pending',  -- pending, completed, cancelled
                    updated_at TIMESTAMP,

                    UNIQUE(booking_id)
                )
            """)

            # Booking outcomes table (for training data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS booking_outcomes (
                    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booking_id TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,

                    -- Timing
                    starts_at TIMESTAMP NOT NULL,
                    ends_at TIMESTAMP NOT NULL,
                    duration_hours REAL,

                    -- Battery data
                    battery_at_start REAL NOT NULL,
                    battery_at_end REAL NOT NULL,
                    battery_drain REAL,
                    charging_at_end INTEGER DEFAULT 0,

                    -- Distance
                    mileage_at_start REAL,
                    mileage_at_end REAL,
                    distance_km REAL,

                    -- Calculated metrics
                    drain_rate_per_hour REAL,
                    drain_rate_per_km REAL,

                    -- Metadata
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    used_for_training INTEGER DEFAULT 0,

                    UNIQUE(booking_id)
                )
            """)

            # Vehicle patterns table (for fast pattern lookups)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_patterns (
                    vehicle_id INTEGER PRIMARY KEY,

                    -- Drain patterns
                    avg_drain_rate_per_hour REAL,
                    avg_drain_rate_per_km REAL,
                    std_drain_rate_per_hour REAL,

                    -- Charging patterns
                    avg_charging_rate_per_hour REAL,
                    charging_frequency REAL,
                    avg_time_between_charges_hours REAL,

                    -- Statistics
                    total_bookings INTEGER DEFAULT 0,
                    total_charging_events INTEGER DEFAULT 0,
                    last_booking_at TIMESTAMP,

                    -- Update tracking
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    patterns_version INTEGER DEFAULT 1
                )
            """)

            # User patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_patterns (
                    user_id INTEGER PRIMARY KEY,

                    -- Behavior patterns
                    avg_return_battery REAL,
                    returns_with_charging_pct REAL,
                    avg_booking_duration_hours REAL,

                    -- Statistics
                    total_bookings INTEGER DEFAULT 0,
                    total_charges INTEGER DEFAULT 0,

                    -- Update tracking
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Accuracy metrics table (for monitoring)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- Time period
                    date DATE NOT NULL,
                    period_type TEXT,  -- daily, weekly, monthly

                    -- Overall metrics
                    total_predictions INTEGER,
                    mae REAL,
                    rmse REAL,
                    within_5pct REAL,
                    within_10pct REAL,
                    within_15pct REAL,

                    -- Per vehicle
                    vehicle_id INTEGER,

                    -- Model version
                    model_version TEXT,

                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(date, period_type, vehicle_id)
                )
            """)

            # Create indices for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_booking ON predictions(booking_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_vehicle ON predictions(vehicle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_vehicle ON booking_outcomes(vehicle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_time ON booking_outcomes(starts_at)")

            logger.info(f"Prediction tracking database initialized at {self.db_path}")

    def store_prediction(
        self,
        booking_id: str,
        vehicle_id: int,
        user_id: int,
        booking_start_time: datetime,
        predicted_battery: float,
        confidence_lower: float = None,
        confidence_upper: float = None,
        model_version: str = None,
        prediction_method: str = 'historical',
        current_battery_level: float = None
    ) -> str:
        """
        Store a prediction

        Args:
            booking_id: Laravel booking ID
            vehicle_id: Vehicle ID
            user_id: User ID
            booking_start_time: When booking starts
            predicted_battery: Predicted battery percentage
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            model_version: Model version used
            prediction_method: Method used (historical or real_time_cascade)
            current_battery_level: Current battery if using real-time prediction

        Returns:
            prediction_id: Unique prediction identifier
        """
        prediction_id = f"PRED_{booking_id}_{datetime.now().timestamp()}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    prediction_id, booking_id, vehicle_id, user_id,
                    predicted_at, booking_start_time, predicted_battery,
                    confidence_lower, confidence_upper, model_version,
                    prediction_method, current_battery_level, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """, (
                prediction_id, booking_id, vehicle_id, user_id,
                datetime.now(), booking_start_time, predicted_battery,
                confidence_lower, confidence_upper, model_version,
                prediction_method, current_battery_level
            ))

        logger.info(f"Stored prediction {prediction_id} for booking {booking_id}: {predicted_battery:.1f}%")
        return prediction_id

    def record_actual_start(
        self,
        booking_id: str,
        actual_battery: float,
        actual_started_at: datetime = None
    ):
        """
        Record actual battery when booking starts

        Args:
            booking_id: Laravel booking ID
            actual_battery: Actual battery percentage at start
            actual_started_at: Actual start time
        """
        if actual_started_at is None:
            actual_started_at = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get predicted battery
            cursor.execute("""
                SELECT predicted_battery FROM predictions WHERE booking_id = ?
            """, (booking_id,))
            row = cursor.fetchone()

            if row:
                predicted_battery = row['predicted_battery']
                error = actual_battery - predicted_battery

                # Update prediction with actual
                cursor.execute("""
                    UPDATE predictions
                    SET actual_battery = ?,
                        actual_started_at = ?,
                        prediction_error = ?,
                        status = 'completed',
                        updated_at = ?
                    WHERE booking_id = ?
                """, (actual_battery, actual_started_at, error, datetime.now(), booking_id))

                logger.info(f"Recorded actual for {booking_id}: {actual_battery:.1f}% (error: {error:+.1f}%)")
            else:
                logger.warning(f"No prediction found for booking {booking_id}")

    def store_booking_outcome(
        self,
        booking_id: str,
        vehicle_id: int,
        user_id: int,
        starts_at: datetime,
        ends_at: datetime,
        battery_at_start: float,
        battery_at_end: float,
        charging_at_end: int = 0,
        mileage_at_start: float = None,
        mileage_at_end: float = None
    ):
        """
        Store complete booking outcome for training

        Args:
            booking_id: Laravel booking ID
            vehicle_id: Vehicle ID
            user_id: User ID
            starts_at: Booking start time
            ends_at: Booking end time
            battery_at_start: Battery at start
            battery_at_end: Battery at end
            charging_at_end: 1 if charged, 0 otherwise
            mileage_at_start: Mileage at start
            mileage_at_end: Mileage at end
        """
        # Calculate metrics
        duration_hours = (ends_at - starts_at).total_seconds() / 3600
        battery_drain = battery_at_start - battery_at_end
        drain_rate_per_hour = battery_drain / duration_hours if duration_hours > 0 else 0

        distance_km = None
        drain_rate_per_km = None
        if mileage_at_start and mileage_at_end:
            distance_km = mileage_at_end - mileage_at_start
            if distance_km > 0:
                drain_rate_per_km = battery_drain / distance_km

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO booking_outcomes (
                    booking_id, vehicle_id, user_id,
                    starts_at, ends_at, duration_hours,
                    battery_at_start, battery_at_end, battery_drain, charging_at_end,
                    mileage_at_start, mileage_at_end, distance_km,
                    drain_rate_per_hour, drain_rate_per_km
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                booking_id, vehicle_id, user_id,
                starts_at, ends_at, duration_hours,
                battery_at_start, battery_at_end, battery_drain, charging_at_end,
                mileage_at_start, mileage_at_end, distance_km,
                drain_rate_per_hour, drain_rate_per_km
            ))

        logger.info(f"Stored outcome for {booking_id}: drain {battery_drain:.1f}% over {duration_hours:.1f}h")

    def get_prediction(self, booking_id: str) -> Optional[Dict]:
        """Get prediction for a booking"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE booking_id = ?", (booking_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def delete_prediction(self, booking_id: str):
        """Delete prediction (for cancelled bookings)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE predictions SET status = 'cancelled' WHERE booking_id = ?", (booking_id,))
        logger.info(f"Marked prediction as cancelled for booking {booking_id}")

    def get_vehicle_accuracy(self, vehicle_id: int, days: int = 30) -> Dict:
        """Get prediction accuracy for a specific vehicle"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(ABS(prediction_error)) as mae,
                    AVG(CASE WHEN ABS(prediction_error) <= 5 THEN 1 ELSE 0 END) * 100 as within_5pct,
                    AVG(CASE WHEN ABS(prediction_error) <= 10 THEN 1 ELSE 0 END) * 100 as within_10pct
                FROM predictions
                WHERE vehicle_id = ?
                  AND status = 'completed'
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
            """, (vehicle_id, days))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return {}

    def get_recent_outcomes(self, vehicle_id: int = None, limit: int = 100) -> pd.DataFrame:
        """Get recent booking outcomes for a vehicle"""
        with self._get_connection() as conn:
            query = "SELECT * FROM booking_outcomes"
            params = []

            if vehicle_id:
                query += " WHERE vehicle_id = ?"
                params.append(vehicle_id)

            query += " ORDER BY starts_at DESC LIMIT ?"
            params.append(limit)

            return pd.read_sql_query(query, conn, params=params)

    def get_accuracy_summary(self) -> Dict:
        """Get overall accuracy summary"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(ABS(prediction_error)) as mae,
                    AVG(CASE WHEN ABS(prediction_error) <= 5 THEN 1 ELSE 0 END) * 100 as within_5pct,
                    AVG(CASE WHEN ABS(prediction_error) <= 10 THEN 1 ELSE 0 END) * 100 as within_10pct,
                    AVG(CASE WHEN ABS(prediction_error) <= 15 THEN 1 ELSE 0 END) * 100 as within_15pct
                FROM predictions
                WHERE status = 'completed'
                  AND predicted_at >= datetime('now', '-7 days')
            """)
            row = cursor.fetchone()
            if row:
                return dict(row)
        return {}


# Global instance
_tracker = None

def get_tracker() -> PredictionTracker:
    """Get global prediction tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = PredictionTracker()
    return _tracker
