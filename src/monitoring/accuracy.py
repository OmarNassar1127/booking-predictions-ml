"""
Prediction Accuracy Monitoring

This module calculates and tracks prediction accuracy metrics over time.
It provides insights into model performance and helps identify areas for improvement.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..database.tracking import get_tracker
from ..utils.logger import logger


class AccuracyMonitor:
    """Monitor and calculate prediction accuracy metrics"""

    def __init__(self):
        """Initialize accuracy monitor"""
        self.tracker = get_tracker()

    def calculate_overall_metrics(self, days: int = 30) -> Dict:
        """
        Calculate overall accuracy metrics for recent predictions

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with accuracy metrics
        """
        with self.tracker._get_connection() as conn:
            # Get completed predictions with actuals
            query = """
                SELECT
                    predicted_battery,
                    actual_battery,
                    prediction_error,
                    prediction_method,
                    vehicle_id,
                    predicted_at
                FROM predictions
                WHERE status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
            """
            df = pd.read_sql_query(query, conn, params=[days])

        if df.empty:
            return {
                'total_predictions': 0,
                'message': 'No completed predictions in this period'
            }

        # Calculate metrics
        errors = df['prediction_error'].abs()

        mae = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())

        within_5 = (errors <= 5).sum() / len(errors) * 100
        within_10 = (errors <= 10).sum() / len(errors) * 100
        within_15 = (errors <= 15).sum() / len(errors) * 100
        within_20 = (errors <= 20).sum() / len(errors) * 100

        # Breakdown by method
        method_metrics = {}
        for method in df['prediction_method'].unique():
            if pd.notna(method):
                method_df = df[df['prediction_method'] == method]
                method_errors = method_df['prediction_error'].abs()
                method_metrics[method] = {
                    'count': len(method_df),
                    'mae': float(method_errors.mean()),
                    'within_10': float((method_errors <= 10).sum() / len(method_errors) * 100)
                }

        return {
            'total_predictions': int(len(df)),
            'period_days': days,
            'mae': float(mae),
            'rmse': float(rmse),
            'within_5_pct': float(within_5),
            'within_10_pct': float(within_10),
            'within_15_pct': float(within_15),
            'within_20_pct': float(within_20),
            'best_prediction': float(errors.min()),
            'worst_prediction': float(errors.max()),
            'median_error': float(errors.median()),
            'by_method': method_metrics,
            'calculated_at': datetime.now().isoformat()
        }

    def calculate_vehicle_metrics(self, vehicle_id: int, days: int = 30) -> Dict:
        """
        Calculate accuracy metrics for a specific vehicle

        Args:
            vehicle_id: Vehicle ID
            days: Number of days to look back

        Returns:
            Dictionary with vehicle-specific metrics
        """
        with self.tracker._get_connection() as conn:
            query = """
                SELECT
                    predicted_battery,
                    actual_battery,
                    prediction_error,
                    booking_start_time,
                    predicted_at
                FROM predictions
                WHERE vehicle_id = ?
                  AND status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
                ORDER BY predicted_at DESC
            """
            df = pd.read_sql_query(query, conn, params=[vehicle_id, days])

        if df.empty:
            return {
                'vehicle_id': vehicle_id,
                'total_predictions': 0,
                'message': 'No completed predictions for this vehicle'
            }

        errors = df['prediction_error'].abs()

        return {
            'vehicle_id': vehicle_id,
            'total_predictions': int(len(df)),
            'mae': float(errors.mean()),
            'within_5_pct': float((errors <= 5).sum() / len(errors) * 100),
            'within_10_pct': float((errors <= 10).sum() / len(errors) * 100),
            'recent_predictions': df.head(10).to_dict('records')
        }

    def get_accuracy_over_time(self, days: int = 30, interval: str = 'daily') -> List[Dict]:
        """
        Get accuracy metrics over time (for charts)

        Args:
            days: Number of days to look back
            interval: 'daily' or 'weekly'

        Returns:
            List of metrics by time period
        """
        with self.tracker._get_connection() as conn:
            query = """
                SELECT
                    DATE(predicted_at) as date,
                    prediction_error,
                    prediction_method
                FROM predictions
                WHERE status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
            """
            df = pd.read_sql_query(query, conn, params=[days])

        if df.empty:
            return []

        df['date'] = pd.to_datetime(df['date'])
        df['abs_error'] = df['prediction_error'].abs()

        # Group by date
        grouped = df.groupby('date').agg({
            'abs_error': ['mean', 'count'],
            'prediction_error': lambda x: (x.abs() <= 10).sum() / len(x) * 100
        }).reset_index()

        grouped.columns = ['date', 'mae', 'count', 'within_10_pct']

        return [
            {
                'date': row['date'].strftime('%Y-%m-%d'),
                'mae': float(row['mae']),
                'count': int(row['count']),
                'within_10_pct': float(row['within_10_pct'])
            }
            for _, row in grouped.iterrows()
        ]

    def get_error_distribution(self, days: int = 30) -> Dict:
        """
        Get distribution of prediction errors

        Args:
            days: Number of days to look back

        Returns:
            Error distribution buckets
        """
        with self.tracker._get_connection() as conn:
            query = """
                SELECT prediction_error
                FROM predictions
                WHERE status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
            """
            df = pd.read_sql_query(query, conn, params=[days])

        if df.empty:
            return {'buckets': []}

        errors = df['prediction_error']

        # Create buckets
        buckets = [
            {'range': '< -20%', 'count': int((errors < -20).sum())},
            {'range': '-20 to -15%', 'count': int(((errors >= -20) & (errors < -15)).sum())},
            {'range': '-15 to -10%', 'count': int(((errors >= -15) & (errors < -10)).sum())},
            {'range': '-10 to -5%', 'count': int(((errors >= -10) & (errors < -5)).sum())},
            {'range': '-5 to 0%', 'count': int(((errors >= -5) & (errors < 0)).sum())},
            {'range': '0 to 5%', 'count': int(((errors >= 0) & (errors < 5)).sum())},
            {'range': '5 to 10%', 'count': int(((errors >= 5) & (errors < 10)).sum())},
            {'range': '10 to 15%', 'count': int(((errors >= 10) & (errors < 15)).sum())},
            {'range': '15 to 20%', 'count': int(((errors >= 15) & (errors < 20)).sum())},
            {'range': '> 20%', 'count': int((errors >= 20).sum())}
        ]

        return {
            'total_predictions': int(len(errors)),
            'buckets': buckets
        }

    def get_top_vehicles_by_accuracy(self, limit: int = 10, days: int = 30) -> List[Dict]:
        """
        Get vehicles with best prediction accuracy

        Args:
            limit: Number of vehicles to return
            days: Number of days to look back

        Returns:
            List of vehicles sorted by accuracy
        """
        with self.tracker._get_connection() as conn:
            query = """
                SELECT
                    vehicle_id,
                    COUNT(*) as prediction_count,
                    AVG(ABS(prediction_error)) as mae,
                    AVG(CASE WHEN ABS(prediction_error) <= 10 THEN 1 ELSE 0 END) * 100 as within_10_pct
                FROM predictions
                WHERE status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
                GROUP BY vehicle_id
                HAVING prediction_count >= 3
                ORDER BY mae ASC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[days, limit])

        return df.to_dict('records')

    def get_worst_predictions(self, limit: int = 10, days: int = 30) -> List[Dict]:
        """
        Get predictions with largest errors (for debugging)

        Args:
            limit: Number of predictions to return
            days: Number of days to look back

        Returns:
            List of worst predictions
        """
        with self.tracker._get_connection() as conn:
            query = """
                SELECT
                    booking_id,
                    vehicle_id,
                    predicted_battery,
                    actual_battery,
                    prediction_error,
                    prediction_method,
                    booking_start_time,
                    predicted_at
                FROM predictions
                WHERE status = 'completed'
                  AND actual_battery IS NOT NULL
                  AND predicted_at >= datetime('now', '-' || ? || ' days')
                ORDER BY ABS(prediction_error) DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[days, limit])

        return df.to_dict('records')

    def store_daily_metrics(self):
        """
        Calculate and store daily accuracy metrics

        This should be run daily via cron job
        """
        metrics = self.calculate_overall_metrics(days=1)

        if metrics.get('total_predictions', 0) == 0:
            logger.info("No predictions to store metrics for")
            return

        with self.tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO accuracy_metrics (
                    date, period_type, total_predictions,
                    mae, rmse, within_5pct, within_10pct, within_15pct,
                    model_version, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date(),
                'daily',
                metrics['total_predictions'],
                metrics['mae'],
                metrics['rmse'],
                metrics['within_5_pct'],
                metrics['within_10_pct'],
                metrics['within_15_pct'],
                'v2.0',
                datetime.now()
            ))

        logger.info(f"Stored daily metrics: MAE={metrics['mae']:.2f}%, Within 10%={metrics['within_10_pct']:.1f}%")


# Global instance
_monitor = None

def get_accuracy_monitor() -> AccuracyMonitor:
    """Get global accuracy monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = AccuracyMonitor()
    return _monitor
