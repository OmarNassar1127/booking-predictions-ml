"""
Clean and prepare booking data for training

This script:
- Removes invalid bookings (end time before start time)
- Handles battery increase cases
- Validates data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_booking_data(input_path: str, output_path: str):
    """Clean booking data"""

    print("=" * 80)
    print("DATA CLEANING")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {input_path}")
    df = pd.read_csv(input_path)

    print(f"Original dataset: {len(df)} bookings")

    # Rename columns if needed
    if 'started_at' in df.columns:
        df.rename(columns={'started_at': 'starts_at'}, inplace=True)
    if 'ended_at' in df.columns:
        df.rename(columns={'ended_at': 'ends_at'}, inplace=True)

    # Convert to datetime
    df['starts_at'] = pd.to_datetime(df['starts_at'])
    df['ends_at'] = pd.to_datetime(df['ends_at'])

    # 1. Remove bookings with end time before start time
    invalid_times = df['ends_at'] <= df['starts_at']
    print(f"\nRemoving {invalid_times.sum()} bookings with invalid times")
    df = df[~invalid_times]

    # 2. Handle battery increases
    # Battery can increase if charging during booking (charging_at_end field)
    battery_increase = df['battery_at_end'] > df['battery_at_start']
    print(f"\nBookings with battery increase: {battery_increase.sum()}")

    if 'charging_at_end' in df.columns:
        charging_bookings = df['charging_at_end'] == 1
        print(f"  - Charging at end: {charging_bookings.sum()}")
        print(f"  - Battery increased without charging: {(battery_increase & ~charging_bookings).sum()}")

    # For now, keep these bookings as they may represent legitimate charging
    # The model will learn this pattern

    # 3. Remove bookings with invalid battery levels
    invalid_battery = (
        (df['battery_at_start'] < 0) | (df['battery_at_start'] > 100) |
        (df['battery_at_end'] < 0) | (df['battery_at_end'] > 100)
    )
    print(f"\nRemoving {invalid_battery.sum()} bookings with invalid battery levels")
    df = df[~invalid_battery]

    # 4. Remove bookings with decreasing mileage
    invalid_mileage = df['mileage_at_end'] < df['mileage_at_start']
    print(f"\nRemoving {invalid_mileage.sum()} bookings with decreasing mileage")
    df = df[~invalid_mileage]

    # 5. Calculate derived features for analysis
    df['duration_hours'] = (df['ends_at'] - df['starts_at']).dt.total_seconds() / 3600
    df['distance_km'] = df['mileage_at_end'] - df['mileage_at_start']
    df['battery_change'] = df['battery_at_end'] - df['battery_at_start']

    # 6. Remove extreme outliers
    # Duration > 7 days
    extreme_duration = df['duration_hours'] > 168
    print(f"\nRemoving {extreme_duration.sum()} bookings with extreme duration (>7 days)")
    df = df[~extreme_duration]

    # Distance > 1000 km (unlikely for car sharing)
    extreme_distance = df['distance_km'] > 1000
    print(f"Removing {extreme_distance.sum()} bookings with extreme distance (>1000km)")
    df = df[~extreme_distance]

    # Battery drain > 95%
    extreme_drain = (df['battery_at_start'] - df['battery_at_end']) > 95
    print(f"Removing {extreme_drain.sum()} bookings with extreme battery drain (>95%)")
    df = df[~extreme_drain]

    # Sort by time
    df = df.sort_values(['vehicle_id', 'starts_at'])

    # Final statistics
    print("\n" + "=" * 80)
    print("CLEANED DATA STATISTICS")
    print("=" * 80)
    print(f"\nFinal dataset: {len(df)} bookings")
    print(f"Vehicles: {df['vehicle_id'].nunique()}")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Date range: {df['starts_at'].min()} to {df['ends_at'].max()}")
    print(f"\nAverage battery at start: {df['battery_at_start'].mean():.1f}%")
    print(f"Average battery at end: {df['battery_at_end'].mean():.1f}%")
    print(f"Average battery change: {df['battery_change'].mean():.1f}%")
    print(f"Average duration: {df['duration_hours'].mean():.1f} hours")
    print(f"Average distance: {df['distance_km'].mean():.1f} km")

    # Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop temporary columns before saving
    df = df.drop(columns=['duration_hours', 'distance_km', 'battery_change'])

    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to {output_path}")

    return df


if __name__ == "__main__":
    clean_booking_data(
        input_path="data/raw/prepared_bookings.csv",
        output_path="data/processed/cleaned_bookings.csv"
    )
