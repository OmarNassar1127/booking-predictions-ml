"""
Train the battery prediction model on REAL PRODUCTION DATA

This script trains on cleaned_bookings.csv (32,904 real bookings with actual charging_at_end data)
instead of the synthetic data used previously.

Usage:
    python train_on_production_data.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import BookingDataLoader
from src.models.battery_predictor import BatteryPredictionModel
from src.utils.logger import logger, setup_logger
from src.utils.config_loader import config


def main():
    """Train and save the model on real production data"""

    # Setup logging
    log_file = Path("logs") / "training_production.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    logger.info("=" * 80)
    logger.info("BATTERY PREDICTION MODEL TRAINING - REAL PRODUCTION DATA")
    logger.info("=" * 80)

    # Load REAL production data
    data_path = "data/processed/cleaned_bookings.csv"
    logger.info(f"Loading REAL production data from {data_path}")

    # Load directly with pandas since it's already processed
    df = pd.read_csv(data_path)

    # Convert datetime columns
    df['starts_at'] = pd.to_datetime(df['starts_at'])
    df['ends_at'] = pd.to_datetime(df['ends_at'])

    logger.info(f"✓ Loaded {len(df):,} real production bookings")
    logger.info(f"  - Unique vehicles: {df['vehicle_id'].nunique()}")
    logger.info(f"  - Unique users: {df['user_id'].nunique()}")
    logger.info(f"  - With charging: {(df['charging_at_end']==1).sum():,} ({(df['charging_at_end']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"  - Without charging: {(df['charging_at_end']==0).sum():,} ({(df['charging_at_end']==0).sum()/len(df)*100:.1f}%)")
    logger.info(f"  - Date range: {df['starts_at'].min()} to {df['starts_at'].max()}")

    # Basic validation
    logger.info("\nValidating production data...")

    issues = []

    # Check for required columns
    required_cols = ['booking_id', 'vehicle_id', 'user_id', 'starts_at', 'ends_at',
                     'battery_at_start', 'battery_at_end', 'charging_at_end']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check for nulls
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Check battery ranges
    if (df['battery_at_start'] < 0).any() or (df['battery_at_start'] > 100).any():
        issues.append("battery_at_start has values outside 0-100 range")
    if (df['battery_at_end'] < 0).any() or (df['battery_at_end'] > 100).any():
        issues.append("battery_at_end has values outside 0-100 range")

    # Check charging values
    if not df['charging_at_end'].isin([0, 1]).all():
        issues.append("charging_at_end should only contain 0 or 1")

    if issues:
        logger.error("Data validation failed!")
        for issue in issues:
            logger.error(f"  - {issue}")
        return

    logger.info("Data validation passed ✓")

    # Create model
    logger.info("\nInitializing model...")
    model = BatteryPredictionModel(config=config.all.get('model', {}))

    # Split data (chronologically for time-series)
    logger.info("\nSplitting data chronologically...")
    df = df.sort_values('starts_at').reset_index(drop=True)

    train_df, val_df, test_df = model.prepare_data(
        df,
        test_size=config.get('training.test_size', 0.2),
        validation_size=config.get('training.validation_size', 0.1)
    )

    logger.info(f"  Train set: {len(train_df):,} bookings")
    logger.info(f"  Validation set: {len(val_df):,} bookings")
    logger.info(f"  Test set: {len(test_df):,} bookings")

    # Train model
    logger.info("\nTraining model on REAL production data...")
    logger.info("This may take a few minutes...")

    metrics = model.train(train_df, val_df)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\nTraining Metrics:")
    logger.info(f"  MAE: {metrics['train']['mae']:.2f}%")
    logger.info(f"  RMSE: {metrics['train']['rmse']:.2f}%")
    logger.info(f"  R²: {metrics['train']['r2']:.4f}")
    logger.info(f"  Within 5%: {metrics['train']['within_5pct']:.1f}%")
    logger.info(f"  Within 10%: {metrics['train']['within_10pct']:.1f}%")

    if metrics['validation']:
        logger.info(f"\nValidation Metrics:")
        logger.info(f"  MAE: {metrics['validation']['mae']:.2f}%")
        logger.info(f"  RMSE: {metrics['validation']['rmse']:.2f}%")
        logger.info(f"  R²: {metrics['validation']['r2']:.4f}")
        logger.info(f"  Within 5%: {metrics['validation']['within_5pct']:.1f}%")
        logger.info(f"  Within 10%: {metrics['validation']['within_10pct']:.1f}%")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = model.evaluate(test_df)

    logger.info(f"\nTest Metrics:")
    logger.info(f"  MAE: {test_metrics['mae']:.2f}%")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}%")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  Within 5%: {test_metrics['within_5pct']:.1f}%")
    logger.info(f"  Within 10%: {test_metrics['within_10pct']:.1f}%")

    # Feature importance
    logger.info("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance(top_n=10)
    for idx, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    # Save model
    save_path = config.get('model.save_path', 'models/battery_predictor.pkl')
    logger.info(f"\nSaving model to {save_path}...")

    model.save(save_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ MODEL TRAINING AND SAVING COMPLETE")
    logger.info("=" * 80)

    # Performance assessment
    logger.info("\nPerformance Assessment:")
    accuracy = test_metrics['within_10pct']

    if accuracy >= 80:
        logger.info(f"  🎉 EXCELLENT: {accuracy:.1f}% within 10% (A grade)")
    elif accuracy >= 70:
        logger.info(f"  ✅ GOOD: {accuracy:.1f}% within 10% (B grade)")
    elif accuracy >= 60:
        logger.info(f"  ⚠️  ACCEPTABLE: {accuracy:.1f}% within 10% (C grade)")
    else:
        logger.info(f"  ❌ NEEDS IMPROVEMENT: {accuracy:.1f}% within 10%")

    logger.info(f"\nModel trained on:")
    logger.info(f"  • {len(df):,} real production bookings")
    logger.info(f"  • {df['vehicle_id'].nunique()} vehicles")
    logger.info(f"  • {(df['charging_at_end']==1).sum():,} charging events")
    logger.info(f"  • Date range: {df['starts_at'].min().date()} to {df['starts_at'].max().date()}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run test_real_accuracy.py to see real-world performance")
    logger.info(f"  2. Start the API server: python run_api.py")
    logger.info(f"  3. Start the dashboard: python run_dashboard.py")


if __name__ == "__main__":
    main()
