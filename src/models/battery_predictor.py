"""
Battery prediction model training and inference
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from ..utils.logger import logger
from ..features.feature_engineer import BatteryFeatureEngineer


class BatteryPredictionModel:
    """Machine learning model for predicting battery at start of booking"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.feature_engineer = BatteryFeatureEngineer()
        self.feature_names = []
        self.target_col = 'battery_at_start'

        # Model parameters
        self.model_params = self.config.get('hyperparameters', {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        })

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training (chronological split)

        Args:
            df: DataFrame with booking data
            test_size: Fraction for test set
            validation_size: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Preparing data...")

        # Sort by time
        df_sorted = df.sort_values('starts_at').reset_index(drop=True)

        n = len(df_sorted)
        n_test = int(n * test_size)
        n_val = int(n * validation_size)
        n_train = n - n_test - n_val

        train_df = df_sorted.iloc[:n_train]
        val_df = df_sorted.iloc[n_train:n_train + n_val]
        test_df = df_sorted.iloc[n_train + n_val:]

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train dates: {train_df['starts_at'].min()} to {train_df['starts_at'].max()}")
        logger.info(f"Val dates: {val_df['starts_at'].min()} to {val_df['starts_at'].max()}")
        logger.info(f"Test dates: {test_df['starts_at'].min()} to {test_df['starts_at'].max()}")

        return train_df, val_df, test_df

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Train the model

        Args:
            train_df: Training data
            val_df: Validation data (optional)

        Returns:
            Training metrics dictionary
        """
        logger.info("Training model...")

        # Engineer features on training data
        train_features = self.feature_engineer.create_features(train_df, is_training=True)

        # Store feature names
        self.feature_names = [col for col in train_features.columns if col not in [
            self.target_col, 'booking_id', 'vehicle_id', 'user_id',
            'starts_at', 'ends_at', 'battery_at_end', 'battery_drain',
            'mileage_at_start', 'mileage_at_end', 'prev_booking_end',
            'vehicle_user_combo'
        ]]

        # Prepare training data
        X_train = train_features[self.feature_names]
        y_train = train_features[self.target_col]

        logger.info(f"Training with {len(self.feature_names)} features")
        logger.info(f"Feature names: {self.feature_names[:10]}... ({len(self.feature_names)} total)")

        # Prepare validation data if provided
        eval_set = None
        if val_df is not None:
            val_features = self.feature_engineer.create_features(val_df, is_training=False)
            X_val = val_features[self.feature_names]
            y_val = val_features[self.target_col]
            eval_set = [(X_val, y_val)]

        # Train LightGBM model
        self.model = lgb.LGBMRegressor(**self.model_params)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
        )

        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred, "Train")

        # Calculate validation metrics if available
        val_metrics = {}
        if val_df is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred, "Validation")

        logger.info("Training complete")
        logger.info(f"Train MAE: {train_metrics['mae']:.2f}%")
        if val_metrics:
            logger.info(f"Val MAE: {val_metrics['mae']:.2f}%")

        return {
            'train': train_metrics,
            'validation': val_metrics
        }

    def predict(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions

        Args:
            df: DataFrame with booking data
            return_confidence: If True, also return prediction intervals

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Engineer features (use cached statistics)
        features_df = self.feature_engineer.create_features(df, is_training=False)

        X = features_df[self.feature_names]

        # Make predictions
        predictions = self.model.predict(X)

        # Clip predictions to valid range
        predictions = np.clip(predictions, 0, 100)

        result_df = df.copy()
        result_df['predicted_battery_at_start'] = predictions

        if return_confidence:
            # Estimate prediction intervals using quantile regression or residual std
            # For simplicity, use a fixed percentage of the prediction
            std_estimate = 5.0  # Assume ~5% standard deviation

            result_df['prediction_lower_bound'] = np.clip(predictions - 1.96 * std_estimate, 0, 100)
            result_df['prediction_upper_bound'] = np.clip(predictions + 1.96 * std_estimate, 0, 100)
            result_df['prediction_confidence'] = 0.95

        return result_df

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model on test set

        Args:
            test_df: Test data

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")

        # Engineer features
        test_features = self.feature_engineer.create_features(test_df, is_training=False)

        X_test = test_features[self.feature_names]
        y_test = test_features[self.target_col]

        # Predict
        test_pred = self.model.predict(X_test)
        test_pred = np.clip(test_pred, 0, 100)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, test_pred, "Test")

        logger.info(f"Test MAE: {metrics['mae']:.2f}%")
        logger.info(f"Test RMSE: {metrics['rmse']:.2f}%")
        logger.info(f"Test R²: {metrics['r2']:.4f}")

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict:
        """Calculate evaluation metrics"""

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Percentage of predictions within X% of actual
        errors = np.abs(y_true - y_pred)
        within_5pct = (errors <= 5).mean() * 100
        within_10pct = (errors <= 10).mean() * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct,
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save(self, save_path: str):
        """
        Save model to disk

        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and feature engineer
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'config': self.config
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> 'BatteryPredictionModel':
        """
        Load model from disk

        Args:
            load_path: Path to load model from

        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from {load_path}")

        model_data = joblib.load(load_path)

        instance = cls(config=model_data['config'])
        instance.model = model_data['model']
        instance.feature_engineer = model_data['feature_engineer']
        instance.feature_names = model_data['feature_names']

        logger.info("Model loaded successfully")

        return instance


if __name__ == "__main__":
    # Test model training
    from ..data.data_loader import BookingDataLoader

    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "bookings.csv"
    loader = BookingDataLoader(str(data_path))
    df = loader.load()

    # Create and train model
    model = BatteryPredictionModel()

    # Split data
    train_df, val_df, test_df = model.prepare_data(df)

    # Train
    metrics = model.train(train_df, val_df)

    # Evaluate
    test_metrics = model.evaluate(test_df)

    # Feature importance
    importance = model.get_feature_importance()
    print("\nTop 10 Features:")
    print(importance.head(10))

    # Save model
    save_path = Path(__file__).parent.parent.parent / "models" / "battery_predictor.pkl"
    model.save(str(save_path))
