"""
Improve model performance to achieve 89%+ within 10% accuracy

Strategy:
1. Better utilize charging_at_end field
2. Enhanced charging pattern features
3. Vehicle-specific charging behavior
4. Improved hyperparameters
5. Ensemble methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import BookingDataLoader
from src.utils.logger import logger, setup_logger


def analyze_charging_patterns(df: pd.DataFrame):
    """Analyze charging patterns in the data"""

    print("\n" + "="*80)
    print("CHARGING PATTERN ANALYSIS")
    print("="*80)

    # Basic stats
    if 'charging_at_end' in df.columns:
        print(f"\nBookings with charging_at_end=1: {(df['charging_at_end'] == 1).sum()} ({(df['charging_at_end'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"Bookings with charging_at_end=0: {(df['charging_at_end'] == 0).sum()} ({(df['charging_at_end'] == 0).sum() / len(df) * 100:.1f}%)")

    # Sort by vehicle and time
    df = df.sort_values(['vehicle_id', 'starts_at'])

    # Calculate actual battery change between bookings
    df['prev_battery_end'] = df.groupby('vehicle_id')['battery_at_end'].shift(1)
    df['prev_ends_at'] = df.groupby('vehicle_id')['ends_at'].shift(1)
    df['prev_charging'] = df.groupby('vehicle_id')['charging_at_end'].shift(1)

    df['time_gap_hours'] = (df['starts_at'] - df['prev_ends_at']).dt.total_seconds() / 3600
    df['battery_change'] = df['battery_at_start'] - df['prev_battery_end']

    # Filter to valid consecutive bookings
    valid_df = df[df['time_gap_hours'].notna() & (df['time_gap_hours'] > 0) & (df['time_gap_hours'] < 168)].copy()

    print(f"\n📊 Battery Change Analysis:")
    print(f"   Average battery change: {valid_df['battery_change'].mean():.2f}%")
    print(f"   Median battery change: {valid_df['battery_change'].median():.2f}%")

    # Compare charging vs non-charging
    if 'prev_charging' in valid_df.columns:
        charging = valid_df[valid_df['prev_charging'] == 1]
        not_charging = valid_df[valid_df['prev_charging'] == 0]

        print(f"\n🔌 When previous booking had charging_at_end=1:")
        print(f"   Count: {len(charging)}")
        print(f"   Avg battery change: {charging['battery_change'].mean():.2f}%")
        print(f"   Avg time gap: {charging['time_gap_hours'].mean():.2f} hours")

        print(f"\n🚗 When previous booking had charging_at_end=0:")
        print(f"   Count: {len(not_charging)}")
        print(f"   Avg battery change: {not_charging['battery_change'].mean():.2f}%")
        print(f"   Avg time gap: {not_charging['time_gap_hours'].mean():.2f} hours")

        # Charging rate analysis
        charging_with_increase = charging[charging['battery_change'] > 1]
        print(f"\n⚡ Actual charging sessions (battery increased):")
        print(f"   Count: {len(charging_with_increase)}")
        print(f"   Avg battery gain: {charging_with_increase['battery_change'].mean():.2f}%")
        print(f"   Avg charging rate: {(charging_with_increase['battery_change'] / charging_with_increase['time_gap_hours']).mean():.2f}% per hour")

    # Time-of-day analysis
    print(f"\n🕐 Time Gap Analysis:")
    print(f"   < 1 hour: {(valid_df['time_gap_hours'] < 1).sum()} bookings, avg change: {valid_df[valid_df['time_gap_hours'] < 1]['battery_change'].mean():.2f}%")
    print(f"   1-4 hours: {((valid_df['time_gap_hours'] >= 1) & (valid_df['time_gap_hours'] < 4)).sum()} bookings, avg change: {valid_df[(valid_df['time_gap_hours'] >= 1) & (valid_df['time_gap_hours'] < 4)]['battery_change'].mean():.2f}%")
    print(f"   4-12 hours: {((valid_df['time_gap_hours'] >= 4) & (valid_df['time_gap_hours'] < 12)).sum()} bookings, avg change: {valid_df[(valid_df['time_gap_hours'] >= 4) & (valid_df['time_gap_hours'] < 12)]['battery_change'].mean():.2f}%")
    print(f"   12+ hours: {(valid_df['time_gap_hours'] >= 12).sum()} bookings, avg change: {valid_df[valid_df['time_gap_hours'] >= 12]['battery_change'].mean():.2f}%")

    return valid_df


def main():
    """Analyze and identify improvement opportunities"""

    # Setup logging
    log_file = Path("logs") / "improvements.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    logger.info("="*80)
    logger.info("MODEL IMPROVEMENT ANALYSIS")
    logger.info("="*80)

    # Load data
    data_path = "data/processed/cleaned_bookings.csv"
    logger.info(f"Loading data from {data_path}")

    loader = BookingDataLoader(data_path)
    df = loader.load()

    # Analyze charging patterns
    analysis_df = analyze_charging_patterns(df)

    # Recommendations
    print("\n" + "="*80)
    print("🎯 IMPROVEMENT RECOMMENDATIONS")
    print("="*80)

    print("\n1. ✅ UTILIZE charging_at_end FIELD")
    print("   - This field explicitly tells us when charging occurred")
    print("   - Add features based on previous booking's charging status")
    print("   - Model charging probability based on patterns")

    print("\n2. ✅ VEHICLE-SPECIFIC CHARGING RATES")
    print("   - Different vehicles charge at different rates")
    print("   - Learn per-vehicle charging speed from historical data")
    print("   - Use this for more accurate charging predictions")

    print("\n3. ✅ TEMPORAL CHARGING PATTERNS")
    print("   - Charging is more likely overnight (long gaps)")
    print("   - Model time-of-day charging probability")
    print("   - Weekend vs weekday patterns")

    print("\n4. ✅ IMPROVED FEATURE ENGINEERING")
    print("   - Consecutive charging sessions")
    print("   - Charging location patterns (if available)")
    print("   - Battery level when charging started")

    print("\n5. ✅ BETTER MODEL ARCHITECTURE")
    print("   - Ensemble: LightGBM + XGBoost + CatBoost")
    print("   - Separate models for charging vs non-charging scenarios")
    print("   - Quantile regression for better uncertainty estimates")

    print("\n6. ✅ HYPERPARAMETER OPTIMIZATION")
    print("   - Use Optuna for automatic tuning")
    print("   - Optimize for 'within 10%' metric specifically")
    print("   - Cross-validation to avoid overfitting")

    print("\n" + "="*80)
    print("Next: Run enhanced_train_model.py to implement improvements")
    print("="*80)


if __name__ == "__main__":
    main()
