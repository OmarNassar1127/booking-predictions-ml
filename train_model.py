"""
Train the battery prediction model

Usage:
    python train_model.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import BookingDataLoader
from src.models.battery_predictor import BatteryPredictionModel
from src.utils.logger import logger, setup_logger
from src.utils.config_loader import config


def main():
    """Train and save the model"""

    # Setup logging
    log_file = Path("logs") / "training.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    logger.info("=" * 80)
    logger.info("BATTERY PREDICTION MODEL TRAINING")
    logger.info("=" * 80)

    # Load data
    data_path = config.get('data.raw_data_path', 'data/raw/bookings.csv')
    logger.info(f"Loading data from {data_path}")

    loader = BookingDataLoader(data_path)
    df = loader.load()

    # Validate data
    validation = loader.validate_data()

    if not validation['valid']:
        logger.error("Data validation failed!")
        for issue in validation['issues']:
            logger.error(f"  - {issue}")
        return

    logger.info("Data validation passed ✓")

    # Display statistics
    logger.info("\nData Statistics:")
    for key, value in validation['stats'].items():
        logger.info(f"  {key}: {value}")

    # Create model
    logger.info("\nInitializing model...")
    model = BatteryPredictionModel(config=config.all.get('model', {}))

    # Split data
    logger.info("\nSplitting data...")
    train_df, val_df, test_df = model.prepare_data(
        df,
        test_size=config.get('training.test_size', 0.2),
        validation_size=config.get('training.validation_size', 0.1)
    )

    # Train model
    logger.info("\nTraining model...")
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

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Start the API server: python run_api.py")
    logger.info(f"  2. Start the dashboard: python run_dashboard.py")
    logger.info(f"  3. Integrate with Laravel using the API endpoints")


if __name__ == "__main__":
    main()
