"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta

from ..utils.logger import logger


class BookingDataLoader:
    """Load and preprocess booking data"""

    def __init__(self, data_path: str):
        """
        Initialize data loader

        Args:
            data_path: Path to CSV file with booking data
        """
        self.data_path = Path(data_path)
        self.df = None

    def load(self) -> pd.DataFrame:
        """Load data from CSV"""
        logger.info(f"Loading data from {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Handle different column name formats
        # Rename started_at/ended_at to starts_at/ends_at if needed
        if 'started_at' in self.df.columns and 'starts_at' not in self.df.columns:
            self.df.rename(columns={'started_at': 'starts_at'}, inplace=True)
        if 'ended_at' in self.df.columns and 'ends_at' not in self.df.columns:
            self.df.rename(columns={'ended_at': 'ends_at'}, inplace=True)

        # Convert date columns to datetime
        date_columns = ['starts_at', 'ends_at']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])

        logger.info(f"Loaded {len(self.df)} bookings")
        logger.info(f"Date range: {self.df['starts_at'].min()} to {self.df['starts_at'].max()}")
        logger.info(f"Vehicles: {self.df['vehicle_id'].nunique()}, Users: {self.df['user_id'].nunique()}")

        return self.df

    def validate_data(self) -> Dict[str, any]:
        """
        Validate data quality and return statistics

        Returns:
            Dictionary with validation results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        issues = []
        stats = {}

        # Check required columns
        required_cols = [
            'booking_id', 'vehicle_id', 'user_id',
            'starts_at', 'ends_at',
            'battery_at_start', 'battery_at_end',
            'mileage_at_start', 'mileage_at_end'
        ]

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for missing values
        missing_values = self.df[required_cols].isnull().sum()
        if missing_values.any():
            issues.append(f"Missing values found:\n{missing_values[missing_values > 0]}")

        # Check battery levels are in valid range
        battery_cols = ['battery_at_start', 'battery_at_end']
        for col in battery_cols:
            invalid_battery = (
                (self.df[col] < 0) | (self.df[col] > 100)
            ).sum()
            if invalid_battery > 0:
                issues.append(f"{col}: {invalid_battery} values outside 0-100% range")

        # Check that end time is after start time
        invalid_times = (self.df['ends_at'] <= self.df['starts_at']).sum()
        if invalid_times > 0:
            issues.append(f"{invalid_times} bookings with end time before start time")

        # Check that mileage increases
        invalid_mileage = (self.df['mileage_at_end'] < self.df['mileage_at_start']).sum()
        if invalid_mileage > 0:
            issues.append(f"{invalid_mileage} bookings with decreasing mileage")

        # Check battery drain is reasonable (should decrease during booking)
        battery_increase = (self.df['battery_at_end'] > self.df['battery_at_start']).sum()
        if battery_increase > 0:
            logger.warning(f"{battery_increase} bookings show battery increase (unusual)")

        # Statistics
        stats['total_bookings'] = len(self.df)
        stats['unique_vehicles'] = self.df['vehicle_id'].nunique()
        stats['unique_users'] = self.df['user_id'].nunique()
        stats['date_range'] = (
            self.df['starts_at'].min(),
            self.df['starts_at'].max()
        )
        stats['avg_battery_start'] = self.df['battery_at_start'].mean()
        stats['avg_battery_end'] = self.df['battery_at_end'].mean()
        stats['avg_battery_drain'] = (self.df['battery_at_start'] - self.df['battery_at_end']).mean()

        # Bookings per vehicle
        bookings_per_vehicle = self.df.groupby('vehicle_id').size()
        stats['avg_bookings_per_vehicle'] = bookings_per_vehicle.mean()
        stats['min_bookings_per_vehicle'] = bookings_per_vehicle.min()
        stats['max_bookings_per_vehicle'] = bookings_per_vehicle.max()

        # Bookings per user
        bookings_per_user = self.df.groupby('user_id').size()
        stats['avg_bookings_per_user'] = bookings_per_user.mean()

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }

    def get_vehicle_timeline(self, vehicle_id: str) -> pd.DataFrame:
        """
        Get chronological booking history for a specific vehicle

        Args:
            vehicle_id: Vehicle ID

        Returns:
            DataFrame with bookings for this vehicle, sorted by time
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        vehicle_bookings = self.df[self.df['vehicle_id'] == vehicle_id].copy()
        vehicle_bookings = vehicle_bookings.sort_values('starts_at')

        return vehicle_bookings

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """
        Get booking history for a specific user

        Args:
            user_id: User ID

        Returns:
            DataFrame with bookings for this user
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        user_bookings = self.df[self.df['user_id'] == user_id].copy()
        user_bookings = user_bookings.sort_values('starts_at')

        return user_bookings

    def split_data(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        time_based: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets

        Args:
            test_size: Fraction of data for test set
            validation_size: Fraction of data for validation set
            time_based: If True, use chronological split (important for time series)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        df_sorted = self.df.sort_values('starts_at').reset_index(drop=True)

        n = len(df_sorted)
        n_test = int(n * test_size)
        n_val = int(n * validation_size)
        n_train = n - n_test - n_val

        if time_based:
            # Chronological split
            train_df = df_sorted.iloc[:n_train]
            val_df = df_sorted.iloc[n_train:n_train + n_val]
            test_df = df_sorted.iloc[n_train + n_val:]

            logger.info(f"Time-based split:")
            logger.info(f"  Train: {train_df['starts_at'].min()} to {train_df['starts_at'].max()}")
            logger.info(f"  Val:   {val_df['starts_at'].min()} to {val_df['starts_at'].max()}")
            logger.info(f"  Test:  {test_df['starts_at'].min()} to {test_df['starts_at'].max()}")
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            train_val_df, test_df = train_test_split(
                df_sorted,
                test_size=test_size,
                random_state=42
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=validation_size / (1 - test_size),
                random_state=42
            )

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def get_charging_events(self) -> pd.DataFrame:
        """
        Infer charging events from battery level changes between bookings

        Returns:
            DataFrame with inferred charging events
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        charging_events = []

        for vehicle_id in self.df['vehicle_id'].unique():
            vehicle_bookings = self.get_vehicle_timeline(vehicle_id)

            for i in range(1, len(vehicle_bookings)):
                prev_booking = vehicle_bookings.iloc[i - 1]
                curr_booking = vehicle_bookings.iloc[i]

                # Calculate gap and battery change
                time_gap = (curr_booking['starts_at'] - prev_booking['ends_at'])
                time_gap_hours = time_gap.total_seconds() / 3600

                battery_change = curr_booking['battery_at_start'] - prev_booking['battery_at_end']

                # Consider it a charging event if battery increased
                if battery_change > 1 and time_gap_hours > 0:  # At least 1% increase
                    charging_events.append({
                        'vehicle_id': vehicle_id,
                        'charging_start': prev_booking['ends_at'],
                        'charging_end': curr_booking['starts_at'],
                        'duration_hours': time_gap_hours,
                        'battery_before': prev_booking['battery_at_end'],
                        'battery_after': curr_booking['battery_at_start'],
                        'battery_gain': battery_change,
                        'charging_rate_percent_per_hour': battery_change / time_gap_hours if time_gap_hours > 0 else 0,
                        'prev_booking_id': prev_booking['booking_id'],
                        'next_booking_id': curr_booking['booking_id'],
                    })

        charging_df = pd.DataFrame(charging_events)

        if len(charging_df) > 0:
            logger.info(f"Identified {len(charging_df)} charging events")
            logger.info(f"  Avg charging gain: {charging_df['battery_gain'].mean():.1f}%")
            logger.info(f"  Avg charging duration: {charging_df['duration_hours'].mean():.1f} hours")

        return charging_df


if __name__ == "__main__":
    # Example usage
    loader = BookingDataLoader("../../data/raw/bookings.csv")
    df = loader.load()

    # Validate
    validation = loader.validate_data()
    print("\nValidation Results:")
    print(f"Valid: {validation['valid']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    print("\nStatistics:")
    for key, value in validation['stats'].items():
        print(f"  {key}: {value}")

    # Get charging events
    charging_df = loader.get_charging_events()
    print("\nSample charging events:")
    print(charging_df.head())
