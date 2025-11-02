"""
Hyperparameter optimization using Optuna to reach 89%+ within 10%
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import BookingDataLoader
from src.features.feature_engineer import BatteryFeatureEngineer
from src.features.enhanced_features import EnhancedBatteryFeatures
from src.utils.logger import logger

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib
import optuna


def calculate_within_percentage(y_true, y_pred, threshold=10):
    """Calculate percentage of predictions within threshold%"""
    errors = np.abs(y_true - y_pred)
    within = (errors <= threshold).sum()
    return (within / len(y_true)) * 100


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function - optimize for within 10% accuracy"""

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,

        # Hyperparameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Predict and clip
    val_pred = np.clip(model.predict(X_val), 0, 100)

    # Optimize for within 10% metric
    within_10 = calculate_within_percentage(y_val, val_pred, 10)

    # Return negative because Optuna minimizes
    return -within_10


def optimize_hyperparameters(n_trials=50):
    """Run hyperparameter optimization"""

    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)

    # Load data
    data_path = "data/processed/cleaned_bookings.csv"
    loader = BookingDataLoader(data_path)
    df = loader.load()

    # Split
    df_sorted = df.sort_values('starts_at').reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    # Features
    enhancer = EnhancedBatteryFeatures()
    train_enhanced = enhancer.create_charging_features(train_df, is_training=True)
    val_enhanced = enhancer.create_charging_features(val_df, is_training=False)
    test_enhanced = enhancer.create_charging_features(test_df, is_training=False)

    feature_engineer = BatteryFeatureEngineer()
    train_features = feature_engineer.create_features(train_enhanced, is_training=True)
    val_features = feature_engineer.create_features(val_enhanced, is_training=False)
    test_features = feature_engineer.create_features(test_enhanced, is_training=False)

    exclude_cols = [
        'battery_at_start', 'booking_id', 'vehicle_id', 'user_id',
        'starts_at', 'ends_at', 'battery_at_end', 'battery_drain',
        'mileage_at_start', 'mileage_at_end', 'prev_booking_end',
        'vehicle_user_combo', 'prev_ends_at', 'last_charging_booking',
        'account_community_id'
    ]

    feature_cols = [col for col in train_features.columns if col not in exclude_cols]

    X_train = train_features[feature_cols]
    y_train = train_features['battery_at_start']
    X_val = val_features[feature_cols]
    y_val = val_features['battery_at_start']
    X_test = test_features[feature_cols]
    y_test = test_features['battery_at_start']

    print(f"\nOptimizing hyperparameters ({n_trials} trials)...")
    print("This may take a few minutes...\n")

    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        study_name='battery_prediction_optimization'
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Best parameters
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)

    print(f"\n🎯 Best validation within 10%: {-study.best_value:.2f}%")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")

    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42
    })

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Final evaluation
    test_pred = np.clip(final_model.predict(X_test), 0, 100)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_within_5 = calculate_within_percentage(y_test, test_pred, 5)
    test_within_10 = calculate_within_percentage(y_test, test_pred, 10)
    test_within_15 = calculate_within_percentage(y_test, test_pred, 15)

    print("\n" + "="*80)
    print("FINAL TEST PERFORMANCE")
    print("="*80)

    print(f"\n📊 Test Set:")
    print(f"   MAE: {test_mae:.2f}%")
    print(f"   Within 5%: {test_within_5:.1f}%")
    print(f"   Within 10%: {test_within_10:.1f}% 🎯")
    print(f"   Within 15%: {test_within_15:.1f}%")

    improvement = test_within_10 - 71.4
    print(f"\n📈 Improvement vs Original (71.4%):")
    print(f"   +{improvement:.1f} percentage points")

    if test_within_10 >= 89:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! {test_within_10:.1f}% ≥ 89% 🎉🎉🎉")
    else:
        gap = 89 - test_within_10
        print(f"\n⚠️  Almost there! Need {gap:.1f} more percentage points")

    # Save optimized model
    save_path = Path("models") / "optimized_battery_predictor.pkl"
    model_data = {
        'model': final_model,
        'feature_engineer': feature_engineer,
        'enhancer': enhancer,
        'feature_cols': feature_cols,
        'best_params': best_params,
        'performance': {
            'test_mae': test_mae,
            'test_within_10': test_within_10,
            'test_within_5': test_within_5
        }
    }

    joblib.dump(model_data, save_path)
    print(f"\n✅ Optimized model saved to {save_path}")

    print("\n" + "="*80)

    return final_model, test_within_10


if __name__ == "__main__":
    model, accuracy = optimize_hyperparameters(n_trials=50)
