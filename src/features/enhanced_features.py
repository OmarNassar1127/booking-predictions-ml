"""
Enhanced feature engineering specifically for charging patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..utils.logger import logger


class EnhancedBatteryFeatures:
    """Enhanced features focusing on charging patterns"""

    def __init__(self):
        self.vehicle_charging_stats = {}
        self.user_charging_stats = {}

    def create_charging_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create enhanced charging-specific features

        Args:
            df: DataFrame with booking data
            is_training: If True, compute statistics from this data

        Returns:
            DataFrame with enhanced charging features
        """
        logger.info("Creating enhanced charging features...")

        df = df.copy()

        # Sort by vehicle and time
        df = df.sort_values(['vehicle_id', 'starts_at']).reset_index(drop=True)

        # 1. Previous booking charging indicator
        df['prev_had_charging'] = df.groupby('vehicle_id')['charging_at_end'].shift(1).fillna(0)
        df['prev_battery_end'] = df.groupby('vehicle_id')['battery_at_end'].shift(1)
        df['prev_ends_at'] = df.groupby('vehicle_id')['ends_at'].shift(1)

        # 2. Time gap
        df['time_gap_hours'] = (df['starts_at'] - df['prev_ends_at']).dt.total_seconds() / 3600
        median_gap = df['time_gap_hours'].median()
        df['time_gap_hours'] = df['time_gap_hours'].fillna(median_gap)

        # 3. Charging probability based on time gap
        df['charging_likely_by_time'] = np.where(
            df['time_gap_hours'] < 1, 0.1,
            np.where(df['time_gap_hours'] < 4, 0.3,
            np.where(df['time_gap_hours'] < 12, 0.6, 0.9))
        )

        # 4. Expected charging if it occurred
        # Based on time gap and typical charging rate
        avg_charging_rate = 6.0  # % per hour (from analysis)
        df['potential_charging_gain'] = np.minimum(
            df['time_gap_hours'] * avg_charging_rate,
            100 - df['prev_battery_end'].fillna(70)  # Can't charge beyond 100%
        )

        # 5. Combine: if previous had charging flag, expect charging
        df['expected_charging_gain'] = np.where(
            df['prev_had_charging'] == 1,
            df['potential_charging_gain'],
            0  # No charging expected
        )

        # 6. Vehicle-specific charging patterns
        if is_training:
            # Calculate per-vehicle charging statistics
            self._calculate_vehicle_charging_stats(df)
            self._calculate_user_charging_stats(df)

        # Apply vehicle-specific charging rates
        df['vehicle_avg_charging_rate'] = df['vehicle_id'].map(
            {k: v.get('avg_charging_rate', 6.0) for k, v in self.vehicle_charging_stats.items()}
        ).fillna(6.0)

        df['vehicle_charging_frequency'] = df['vehicle_id'].map(
            {k: v.get('charging_frequency', 0.5) for k, v in self.vehicle_charging_stats.items()}
        ).fillna(0.5)

        # 7. User-specific patterns
        df['user_returns_with_charging'] = df['user_id'].map(
            {k: v.get('returns_with_charging', 0.25) for k, v in self.user_charging_stats.items()}
        ).fillna(0.25)

        # 8. Predicted battery (simple rule-based baseline)
        df['predicted_battery_simple'] = np.where(
            df['prev_had_charging'] == 1,
            # With charging: prev battery + charging gain
            np.clip(df['prev_battery_end'].fillna(70) + df['expected_charging_gain'], 0, 100),
            # Without charging: prev battery + small natural drain/gain
            np.clip(df['prev_battery_end'].fillna(70) + df['time_gap_hours'] * 0.5, 0, 100)
        )

        # 9. Historical charging sequence features
        # How many of last N bookings had charging
        for window in [3, 5, 10]:
            df[f'charging_ratio_last_{window}'] = (
                df.groupby('vehicle_id')['charging_at_end']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )

        # 10. Time since last charging event
        df['last_charging_booking'] = np.where(
            df['charging_at_end'] == 1,
            df.index,
            np.nan
        )
        df['last_charging_booking'] = df.groupby('vehicle_id')['last_charging_booking'].ffill()
        df['bookings_since_last_charge'] = df.index - df['last_charging_booking'].fillna(0)

        # 11. Battery level categories
        df['prev_battery_low'] = (df['prev_battery_end'] < 30).astype(int)
        df['prev_battery_medium'] = ((df['prev_battery_end'] >= 30) & (df['prev_battery_end'] < 70)).astype(int)
        df['prev_battery_high'] = (df['prev_battery_end'] >= 70).astype(int)

        # 12. Charging urgency (low battery + long gap = likely charging)
        df['charging_urgency'] = np.where(
            (df['prev_battery_end'] < 40) & (df['time_gap_hours'] > 2),
            1,
            0
        )

        # 13. Hour of day patterns for charging
        df['hour_of_day'] = df['starts_at'].dt.hour
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)
        df['is_morning'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12)).astype(int)

        # Night bookings after charging more likely to have high battery
        df['night_after_charging'] = (df['is_night'] * df['prev_had_charging']).astype(int)

        logger.info(f"Created enhanced charging features. Total columns: {len(df.columns)}")

        return df

    def _calculate_vehicle_charging_stats(self, df: pd.DataFrame):
        """Calculate per-vehicle charging statistics"""

        for vehicle_id in df['vehicle_id'].unique():
            vehicle_df = df[df['vehicle_id'] == vehicle_id].copy()

            # Calculate battery change for consecutive bookings
            vehicle_df['battery_change'] = vehicle_df['battery_at_start'] - vehicle_df['prev_battery_end']
            vehicle_df['had_charging'] = vehicle_df['prev_had_charging']

            # Charging sessions only (where battery increased and had charging flag)
            charging_sessions = vehicle_df[
                (vehicle_df['had_charging'] == 1) &
                (vehicle_df['battery_change'] > 1) &
                (vehicle_df['time_gap_hours'] > 0)
            ]

            if len(charging_sessions) > 0:
                avg_charging_rate = (charging_sessions['battery_change'] / charging_sessions['time_gap_hours']).mean()
                max_charging_rate = (charging_sessions['battery_change'] / charging_sessions['time_gap_hours']).max()
            else:
                avg_charging_rate = 6.0
                max_charging_rate = 15.0

            # Charging frequency
            charging_frequency = (vehicle_df['charging_at_end'] == 1).sum() / len(vehicle_df) if len(vehicle_df) > 0 else 0.25

            self.vehicle_charging_stats[vehicle_id] = {
                'avg_charging_rate': avg_charging_rate,
                'max_charging_rate': max_charging_rate,
                'charging_frequency': charging_frequency,
                'total_charging_sessions': len(charging_sessions)
            }

    def _calculate_user_charging_stats(self, df: pd.DataFrame):
        """Calculate per-user charging statistics"""

        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]

            # How often does user return vehicle with charging flag
            returns_with_charging = (user_df['charging_at_end'] == 1).sum() / len(user_df) if len(user_df) > 0 else 0.25

            # Average battery when user returns
            avg_return_battery = user_df['battery_at_end'].mean() if len(user_df) > 0 else 65

            self.user_charging_stats[user_id] = {
                'returns_with_charging': returns_with_charging,
                'avg_return_battery': avg_return_battery
            }


if __name__ == "__main__":
    # Test enhanced features
    from pathlib import Path
    from ..data.data_loader import BookingDataLoader

    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cleaned_bookings.csv"
    loader = BookingDataLoader(str(data_path))
    df = loader.load()

    enhancer = EnhancedBatteryFeatures()
    enhanced_df = enhancer.create_charging_features(df, is_training=True)

    print("\nEnhanced Features Created:")
    print(f"Total columns: {len(enhanced_df.columns)}")
    print("\nNew charging features:")
    charging_cols = [col for col in enhanced_df.columns if 'charg' in col.lower()]
    print(charging_cols)
