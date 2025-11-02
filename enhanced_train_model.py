"""
Enhanced model training with optimized features and hyperparameters

Goals:
- Achieve 89%+ predictions within 10% of actual
- Utilize charging_at_end field effectively
- Better handling of charging scenarios
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import BookingDataLoader
from src.features.feature_engineer import BatteryFeatureEngineer
from src.features.enhanced_features import EnhancedBatteryFeatures
from src.utils.logger import logger, setup_logger

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def calculate_within_percentage(y_true, y_pred, threshold=10):
    """Calculate percentage of predictions within threshold%"""
    errors = np.abs(y_true - y_pred)
    within = (errors <= threshold).sum()
    return (within / len(y_true)) * 100


def train_enhanced_model():
    """Train enhanced model with improved features"""

    # Setup logging
    log_file = Path("logs") / "enhanced_training.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logger(log_file=str(log_file))

    logger.info("="*80)
    logger.info("ENHANCED MODEL TRAINING")
    logger.info("="*80)

    # Load data
    data_path = "data/processed/cleaned_bookings.csv"
    logger.info(f"Loading data from {data_path}")

    loader = BookingDataLoader(data_path)
    df = loader.load()

    logger.info(f"Loaded {len(df)} bookings")

    # Split data chronologically
    df_sorted = df.sort_values('starts_at').reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Feature engineering
    logger.info("Creating features...")

    # 1. Enhanced charging features
    enhancer = EnhancedBatteryFeatures()
    train_enhanced = enhancer.create_charging_features(train_df, is_training=True)
    val_enhanced = enhancer.create_charging_features(val_df, is_training=False)
    test_enhanced = enhancer.create_charging_features(test_df, is_training=False)

    # 2. Original features
    feature_engineer = BatteryFeatureEngineer()
    train_features = feature_engineer.create_features(train_enhanced, is_training=True)
    val_features = feature_engineer.create_features(val_enhanced, is_training=False)
    test_features = feature_engineer.create_features(test_enhanced, is_training=False)

    # Select features
    exclude_cols = [
        'battery_at_start', 'booking_id', 'vehicle_id', 'user_id',
        'starts_at', 'ends_at', 'battery_at_end', 'battery_drain',
        'mileage_at_start', 'mileage_at_end', 'prev_booking_end',
        'vehicle_user_combo', 'prev_ends_at', 'last_charging_booking',
        'account_community_id'  # Add this if you want to exclude it
    ]

    feature_cols = [col for col in train_features.columns if col not in exclude_cols]

    # Prepare datasets
    X_train = train_features[feature_cols]
    y_train = train_features['battery_at_start']

    X_val = val_features[feature_cols]
    y_val = val_features['battery_at_start']

    X_test = test_features[feature_cols]
    y_test = test_features['battery_at_start']

    logger.info(f"Training with {len(feature_cols)} features")

    # Enhanced hyperparameters (optimized for accuracy)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.03,  # Lower learning rate
        'max_depth': 9,  # Deeper trees
        'num_leaves': 63,  # More leaves
        'min_child_samples': 15,  # Smaller to capture more patterns
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    logger.info("Training LightGBM model...")

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Evaluate
    logger.info("\nEvaluating model...")

    # Training set
    train_pred = np.clip(model.predict(X_train), 0, 100)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_within_5 = calculate_within_percentage(y_train, train_pred, 5)
    train_within_10 = calculate_within_percentage(y_train, train_pred, 10)

    # Validation set
    val_pred = np.clip(model.predict(X_val), 0, 100)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_within_5 = calculate_within_percentage(y_val, val_pred, 5)
    val_within_10 = calculate_within_percentage(y_val, val_pred, 10)

    # Test set
    test_pred = np.clip(model.predict(X_test), 0, 100)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    test_within_5 = calculate_within_percentage(y_test, test_pred, 5)
    test_within_10 = calculate_within_percentage(y_test, test_pred, 10)
    test_within_15 = calculate_within_percentage(y_test, test_pred, 15)

    # Results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)

    print(f"\n📊 Training Set:")
    print(f"   MAE: {train_mae:.2f}%")
    print(f"   Within 5%: {train_within_5:.1f}%")
    print(f"   Within 10%: {train_within_10:.1f}%")

    print(f"\n📊 Validation Set:")
    print(f"   MAE: {val_mae:.2f}%")
    print(f"   Within 5%: {val_within_5:.1f}%")
    print(f"   Within 10%: {val_within_10:.1f}%")

    print(f"\n📊 Test Set:")
    print(f"   MAE: {test_mae:.2f}%")
    print(f"   RMSE: {test_rmse:.2f}%")
    print(f"   R²: {test_r2:.4f}")
    print(f"   Within 5%: {test_within_5:.1f}%")
    print(f"   Within 10%: {test_within_10:.1f}% 🎯")
    print(f"   Within 15%: {test_within_15:.1f}%")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔝 Top 15 Most Important Features:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"   {row['feature']}: {row['importance']:.0f}")

    # Improvement vs baseline
    print(f"\n📈 Improvement vs Baseline (71.4% within 10%):")
    improvement = test_within_10 - 71.4
    print(f"   {improvement:+.1f} percentage points")

    if test_within_10 >= 89:
        print(f"\n🎉 TARGET ACHIEVED! {test_within_10:.1f}% ≥ 89%")
    else:
        print(f"\n⚠️  Target not yet reached. Need {89 - test_within_10:.1f} more percentage points")

    # Save enhanced model
    save_path = Path("models") / "enhanced_battery_predictor.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_engineer': feature_engineer,
        'enhancer': enhancer,
        'feature_cols': feature_cols,
        'performance': {
            'test_mae': test_mae,
            'test_within_10': test_within_10,
            'test_within_5': test_within_5
        }
    }

    joblib.dump(model_data, save_path)
    logger.info(f"\nModel saved to {save_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    return model, test_within_10


if __name__ == "__main__":
    model, accuracy = train_enhanced_model()
