"""
Feature engineering for battery prediction model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from ..utils.logger import logger


class BatteryFeatureEngineer:
    """Feature engineering for electric car battery prediction"""

    def __init__(self):
        self.feature_names = []
        self.vehicle_stats = {}
        self.user_stats = {}

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create all features for the model

        Args:
            df: DataFrame with booking data
            is_training: If True, compute statistics from this data; if False, use cached stats

        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features...")

        # Make a copy to avoid modifying original
        features_df = df.copy()

        # Sort by vehicle and time
        features_df = features_df.sort_values(['vehicle_id', 'starts_at']).reset_index(drop=True)

        # Basic derived features
        features_df = self._add_basic_features(features_df)

        # Temporal features
        features_df = self._add_temporal_features(features_df)

        # Vehicle-level historical features
        features_df = self._add_vehicle_features(features_df, is_training)

        # User-level historical features
        features_df = self._add_user_features(features_df, is_training)

        # Gap-based features (time since last booking)
        features_df = self._add_gap_features(features_df)

        # Rolling window features
        features_df = self._add_rolling_features(features_df)

        # Interaction features
        features_df = self._add_interaction_features(features_df)

        logger.info(f"Created {len(features_df.columns)} features")

        return features_df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""

        # Duration and distance (if not already present)
        if 'duration_hours' not in df.columns:
            df['duration_hours'] = (df['ends_at'] - df['starts_at']).dt.total_seconds() / 3600

        if 'distance_km' not in df.columns:
            df['distance_km'] = df['mileage_at_end'] - df['mileage_at_start']

        # Battery drain
        if 'battery_drain' not in df.columns:
            df['battery_drain'] = df['battery_at_start'] - df['battery_at_end']

        # Drain rate
        df['battery_drain_per_km'] = np.where(
            df['distance_km'] > 0,
            df['battery_drain'] / df['distance_km'],
            0
        )

        df['battery_drain_per_hour'] = np.where(
            df['duration_hours'] > 0,
            df['battery_drain'] / df['duration_hours'],
            0
        )

        # Speed
        df['avg_speed_kmh'] = np.where(
            df['duration_hours'] > 0,
            df['distance_km'] / df['duration_hours'],
            0
        )

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""

        df['hour_of_day'] = df['starts_at'].dt.hour
        df['day_of_week'] = df['starts_at'].dt.dayofweek
        df['day_of_month'] = df['starts_at'].dt.day
        df['month'] = df['starts_at'].dt.month
        df['quarter'] = df['starts_at'].dt.quarter
        df['year'] = df['starts_at'].dt.year

        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Time of day categories
        df['is_morning'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour_of_day'] >= 12) & (df['hour_of_day'] < 18)).astype(int)
        df['is_evening'] = ((df['hour_of_day'] >= 18) & (df['hour_of_day'] < 22)).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)

        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # NEW: Rush hour feature (7-9am, 5-7pm)
        df['is_rush_hour'] = (((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 8)) |
                               ((df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 18))).astype(int)

        # NEW: Explicit season feature
        # Winter (Dec, Jan, Feb): 0, Spring (Mar, Apr, May): 1, Summer (Jun, Jul, Aug): 2, Fall (Sep, Oct, Nov): 3
        df['season'] = ((df['month'] % 12 + 3) // 3) % 4

        return df

    def _add_vehicle_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Add vehicle-level historical features"""

        if is_training:
            # Compute vehicle statistics from training data
            vehicle_agg = df.groupby('vehicle_id').agg({
                'battery_drain': ['mean', 'std', 'median'],
                'battery_drain_per_km': ['mean', 'median'],
                'battery_drain_per_hour': ['mean', 'median'],
                'distance_km': ['mean', 'std'],
                'duration_hours': ['mean', 'std'],
                'avg_speed_kmh': ['mean', 'std'],
            })

            # Flatten column names
            vehicle_agg.columns = ['_'.join(col).strip() for col in vehicle_agg.columns.values]
            self.vehicle_stats = vehicle_agg.to_dict('index')

        # Apply vehicle statistics
        for stat_name, stat_dict in self.vehicle_stats.items():
            feature_name = f'vehicle_{stat_name}'
            df[feature_name] = df['vehicle_id'].map(stat_dict).fillna(df['battery_drain'].mean())

        # Vehicle booking count (up to this point in time)
        df['vehicle_booking_count'] = df.groupby('vehicle_id').cumcount()

        # NEW: Vehicle usage intensity (bookings per week)
        # Calculate time span for each vehicle up to current booking
        df['vehicle_days_active'] = (
            df.groupby('vehicle_id')['starts_at']
            .transform(lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600))
        )
        df['vehicle_usage_intensity'] = np.where(
            df['vehicle_days_active'] > 0,
            (df['vehicle_booking_count'] + 1) / (df['vehicle_days_active'] / 7),
            0
        )

        # Battery at end of last booking (for this vehicle)
        df['prev_battery_end'] = df.groupby('vehicle_id')['battery_at_end'].shift(1)
        df['prev_battery_end'] = df['prev_battery_end'].fillna(80)  # Default for first booking

        # NEW: Vehicle charging efficiency (% gained per hour when charging)
        if is_training and 'charging_at_end' in df.columns:
            # Calculate charging rate for bookings that had charging
            charging_bookings = df[df['charging_at_end'] == 1].copy()
            charging_bookings['next_battery_start'] = charging_bookings.groupby('vehicle_id')['battery_at_start'].shift(-1)
            charging_bookings['next_starts_at'] = charging_bookings.groupby('vehicle_id')['starts_at'].shift(-1)
            charging_bookings['charging_gap_hours'] = (
                (charging_bookings['next_starts_at'] - charging_bookings['ends_at']).dt.total_seconds() / 3600
            )
            charging_bookings['battery_gained'] = charging_bookings['next_battery_start'] - charging_bookings['battery_at_end']
            charging_bookings['charging_rate'] = np.where(
                charging_bookings['charging_gap_hours'] > 0,
                charging_bookings['battery_gained'] / charging_bookings['charging_gap_hours'],
                0
            )
            # Filter reasonable rates (5-50% per hour)
            valid_rates = charging_bookings[
                (charging_bookings['charging_rate'] >= 5) &
                (charging_bookings['charging_rate'] <= 50)
            ]
            vehicle_charging_efficiency = valid_rates.groupby('vehicle_id')['charging_rate'].mean().to_dict()
            self.vehicle_stats['charging_efficiency'] = vehicle_charging_efficiency

        if 'charging_efficiency' in self.vehicle_stats:
            df['vehicle_charging_efficiency'] = df['vehicle_id'].map(
                self.vehicle_stats['charging_efficiency']
            ).fillna(25.0)  # Default 25%/hour
        else:
            df['vehicle_charging_efficiency'] = 25.0

        # NEW: Vehicle average idle drain rate (battery loss when parked between bookings)
        if is_training:
            # Calculate idle drain: difference between battery_at_end and next battery_at_start (when not charging)
            no_charge_bookings = df[df.get('charging_at_end', 0) == 0].copy()
            no_charge_bookings['next_battery_start'] = no_charge_bookings.groupby('vehicle_id')['battery_at_start'].shift(-1)
            no_charge_bookings['next_starts_at'] = no_charge_bookings.groupby('vehicle_id')['starts_at'].shift(-1)
            no_charge_bookings['idle_gap_hours'] = (
                (no_charge_bookings['next_starts_at'] - no_charge_bookings['ends_at']).dt.total_seconds() / 3600
            )
            no_charge_bookings['battery_lost'] = no_charge_bookings['battery_at_end'] - no_charge_bookings['next_battery_start']
            no_charge_bookings['idle_drain_rate'] = np.where(
                no_charge_bookings['idle_gap_hours'] > 0,
                no_charge_bookings['battery_lost'] / no_charge_bookings['idle_gap_hours'],
                0
            )
            # Filter reasonable rates (0-2% per hour)
            valid_idle_rates = no_charge_bookings[
                (no_charge_bookings['idle_drain_rate'] >= 0) &
                (no_charge_bookings['idle_drain_rate'] <= 2)
            ]
            vehicle_idle_drain = valid_idle_rates.groupby('vehicle_id')['idle_drain_rate'].mean().to_dict()
            self.vehicle_stats['idle_drain_rate'] = vehicle_idle_drain

        if 'idle_drain_rate' in self.vehicle_stats:
            df['vehicle_avg_idle_drain_rate'] = df['vehicle_id'].map(
                self.vehicle_stats['idle_drain_rate']
            ).fillna(0.5)  # Default 0.5%/hour
        else:
            df['vehicle_avg_idle_drain_rate'] = 0.5

        return df

    def _add_user_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Add user-level historical features"""

        if is_training:
            # Compute user statistics from training data
            user_agg = df.groupby('user_id').agg({
                'battery_drain': ['mean', 'std', 'median'],
                'battery_drain_per_km': ['mean'],
                'distance_km': ['mean', 'std'],
                'duration_hours': ['mean'],
                'avg_speed_kmh': ['mean'],
            })

            # Flatten column names
            user_agg.columns = ['_'.join(col).strip() for col in user_agg.columns.values]
            self.user_stats = user_agg.to_dict('index')

        # Apply user statistics
        for stat_name, stat_dict in self.user_stats.items():
            feature_name = f'user_{stat_name}'
            df[feature_name] = df['user_id'].map(stat_dict).fillna(df['battery_drain'].mean())

        # User booking count
        df['user_booking_count'] = df.groupby('user_id').cumcount()

        # NEW: User charging frequency (% of bookings where they charge)
        if is_training and 'charging_at_end' in df.columns:
            user_charging_freq = df.groupby('user_id')['charging_at_end'].mean().to_dict()
            self.user_stats['charging_frequency'] = user_charging_freq

        if 'charging_frequency' in self.user_stats:
            df['user_charging_frequency'] = df['user_id'].map(
                self.user_stats['charging_frequency']
            ).fillna(0.25)  # Default 25% charge rate
        else:
            df['user_charging_frequency'] = 0.25

        # NEW: User preferred booking hours (mode of booking start hour)
        if is_training:
            user_pref_hours = df.groupby('user_id')['hour_of_day'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean()).to_dict()
            self.user_stats['preferred_hour'] = user_pref_hours

        if 'preferred_hour' in self.user_stats:
            df['user_preferred_hour'] = df['user_id'].map(
                self.user_stats['preferred_hour']
            ).fillna(12)  # Default noon
        else:
            df['user_preferred_hour'] = 12

        # Deviation from preferred hour
        df['hour_deviation_from_preferred'] = np.abs(df['hour_of_day'] - df['user_preferred_hour'])

        return df

    def _add_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to time gap since last booking"""

        # Time since last booking for this vehicle
        df['prev_booking_end'] = df.groupby('vehicle_id')['ends_at'].shift(1)

        df['time_since_last_booking_hours'] = (
            (df['starts_at'] - df['prev_booking_end']).dt.total_seconds() / 3600
        )

        # Fill NaN for first booking (use median gap)
        median_gap = df['time_since_last_booking_hours'].median()
        df['time_since_last_booking_hours'] = df['time_since_last_booking_hours'].fillna(median_gap)

        # Gap categories
        df['gap_very_short'] = (df['time_since_last_booking_hours'] < 1).astype(int)
        df['gap_short'] = ((df['time_since_last_booking_hours'] >= 1) &
                          (df['time_since_last_booking_hours'] < 4)).astype(int)
        df['gap_medium'] = ((df['time_since_last_booking_hours'] >= 4) &
                           (df['time_since_last_booking_hours'] < 12)).astype(int)
        df['gap_long'] = (df['time_since_last_booking_hours'] >= 12).astype(int)

        # Expected charging based on gap
        df['expected_charging_potential'] = np.minimum(
            df['time_since_last_booking_hours'] * 10,  # Assume ~10% per hour charging
            100 - df['prev_battery_end']  # Can't charge beyond 100%
        )

        # NEW: Days since last charging event for this vehicle
        if 'charging_at_end' in df.columns:
            # Mark each booking's end time when charging happened
            df['charge_end_time'] = df['ends_at'].where(df['charging_at_end'] == 1, pd.NaT)

            # Forward-fill to get the last charge time for each subsequent booking
            df['last_charge_time'] = df.groupby('vehicle_id')['charge_end_time'].fillna(method='ffill')

            # Calculate days since last charge
            df['days_since_last_charge'] = (
                (df['starts_at'] - df['last_charge_time']).dt.total_seconds() / (24 * 3600)
            )

            # Fill NaN with large number (vehicle never charged before)
            df['days_since_last_charge'] = df['days_since_last_charge'].fillna(30)  # Default 30 days

            # Drop temporary columns
            df = df.drop(['charge_end_time', 'last_charge_time'], axis=1)
        else:
            df['days_since_last_charge'] = 30  # Default if no charging data

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features for vehicle history"""

        windows = [3, 7, 14]  # Last N bookings

        for window in windows:
            # Rolling averages per vehicle
            df[f'battery_drain_rolling_{window}'] = (
                df.groupby('vehicle_id')['battery_drain']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

            df[f'distance_rolling_{window}'] = (
                df.groupby('vehicle_id')['distance_km']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

            df[f'drain_per_km_rolling_{window}'] = (
                df.groupby('vehicle_id')['battery_drain_per_km']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different variables"""

        # Time-battery interactions
        df['hour_x_prev_battery'] = df['hour_of_day'] * df['prev_battery_end']
        df['weekend_x_gap'] = df['is_weekend'] * df['time_since_last_booking_hours']

        # Vehicle-user interactions
        df['vehicle_user_combo'] = df['vehicle_id'].astype(str) + '_' + df['user_id'].astype(str)

        # NEW: User-vehicle familiarity (how many times this user has used this vehicle before)
        df['user_vehicle_familiarity_count'] = df.groupby(['vehicle_id', 'user_id']).cumcount()

        # Binary flag for whether user is familiar with this vehicle (>2 bookings)
        df['is_familiar_with_vehicle'] = (df['user_vehicle_familiarity_count'] > 2).astype(int)

        # NEW: Community-vehicle pairing strength (how common this community-vehicle pairing is)
        if 'account_community_id' in df.columns:
            # Calculate total bookings per community-vehicle pair
            community_vehicle_counts = df.groupby(['account_community_id', 'vehicle_id']).size()

            # Calculate total bookings per community
            community_total_counts = df.groupby('account_community_id').size()

            # Pairing strength = proportion of community's bookings that use this vehicle
            pairing_strength = (community_vehicle_counts / community_total_counts).to_dict()

            # Map to dataframe
            df['community_vehicle_pairing'] = df.apply(
                lambda row: pairing_strength.get((row['account_community_id'], row['vehicle_id']), 0),
                axis=1
            )
        else:
            df['community_vehicle_pairing'] = 0

        # Expected battery based on simple heuristics
        df['expected_battery_simple'] = np.clip(
            df['prev_battery_end'] + (df['time_since_last_booking_hours'] * 5),  # 5% per hour
            0, 100
        )

        return df

    def get_feature_names(self, target_col: str = 'battery_at_start') -> List[str]:
        """
        Get list of feature column names (excluding target and metadata)

        Args:
            target_col: Name of target column to exclude

        Returns:
            List of feature column names
        """
        exclude_cols = [
            target_col,
            'booking_id', 'vehicle_id', 'user_id', 'vehicle_user_combo',
            'starts_at', 'ends_at', 'started_at', 'ended_at',
            'battery_at_end', 'battery_drain',
            'mileage_at_start', 'mileage_at_end',
            'prev_booking_end'
        ]

        return [col for col in self.feature_names if col not in exclude_cols]


class PredictionFeatureBuilder:
    """Build features for making predictions on new bookings"""

    def __init__(self, feature_engineer: BatteryFeatureEngineer, historical_data: pd.DataFrame):
        """
        Initialize prediction feature builder

        Args:
            feature_engineer: Trained feature engineer
            historical_data: Historical booking data to extract last known states
        """
        self.feature_engineer = feature_engineer
        self.historical_data = historical_data.sort_values(['vehicle_id', 'starts_at'])

        # Cache last booking per vehicle
        self.last_booking_per_vehicle = {}
        for vehicle_id in historical_data['vehicle_id'].unique():
            vehicle_bookings = historical_data[historical_data['vehicle_id'] == vehicle_id]
            last_booking = vehicle_bookings.iloc[-1]
            self.last_booking_per_vehicle[vehicle_id] = last_booking

    def build_features_for_prediction(
        self,
        vehicle_id: str,
        user_id: str,
        booking_start_time: datetime,
        intermediate_bookings: List[Dict] = None
    ) -> pd.DataFrame:
        """
        Build features for a future booking prediction

        Args:
            vehicle_id: Vehicle ID
            user_id: User ID
            booking_start_time: When the booking will start
            intermediate_bookings: List of bookings between last known and this one

        Returns:
            DataFrame with features (single row)
        """

        # Get last known state for this vehicle
        if vehicle_id in self.last_booking_per_vehicle:
            last_booking = self.last_booking_per_vehicle[vehicle_id]
            prev_battery_end = last_booking['battery_at_end']
            prev_booking_end_time = last_booking['ends_at']
        else:
            # New vehicle - use fleet averages
            prev_battery_end = 80.0  # Default
            prev_booking_end_time = booking_start_time - timedelta(hours=12)

        # If there are intermediate bookings, use the last one
        if intermediate_bookings:
            last_intermediate = intermediate_bookings[-1]
            prev_battery_end = last_intermediate.get('predicted_battery_end', prev_battery_end)
            prev_booking_end_time = last_intermediate['ends_at']

        # Calculate gap
        time_gap_hours = (booking_start_time - prev_booking_end_time).total_seconds() / 3600

        # Create a dummy row with known information
        prediction_row = {
            'booking_id': -1,
            'vehicle_id': vehicle_id,
            'user_id': user_id,
            'starts_at': booking_start_time,
            'ends_at': booking_start_time + timedelta(hours=2),  # Dummy duration
            'battery_at_start': 0,  # This is what we're predicting
            'battery_at_end': 0,  # Dummy
            'mileage_at_start': 0,  # Dummy
            'mileage_at_end': 0,  # Dummy
        }

        prediction_df = pd.DataFrame([prediction_row])

        # Engineer features (use training=False to use cached stats)
        prediction_df = self.feature_engineer.create_features(prediction_df, is_training=False)

        # Override some features with actual values
        prediction_df['prev_battery_end'] = prev_battery_end
        prediction_df['time_since_last_booking_hours'] = time_gap_hours

        return prediction_df


if __name__ == "__main__":
    # Test feature engineering
    from pathlib import Path
    from ..data.data_loader import BookingDataLoader

    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "bookings.csv"
    loader = BookingDataLoader(str(data_path))
    df = loader.load()

    # Create features
    engineer = BatteryFeatureEngineer()
    features_df = engineer.create_features(df, is_training=True)

    print("\nFeature Engineering Complete")
    print(f"Original columns: {len(df.columns)}")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"\nSample features:")
    print(features_df.head())
