"""
Gradio dashboard for battery prediction system

Provides interactive interface for:
- Making predictions
- Viewing vehicle timelines
- Analyzing model performance
- What-if scenarios
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..models.prediction_service import BatteryPredictionService
from ..models.battery_predictor import BatteryPredictionModel
from ..data.data_loader import BookingDataLoader
from ..utils.logger import logger


class BatteryPredictionDashboard:
    """Gradio dashboard for battery prediction"""

    def __init__(self):
        self.service: Optional[BatteryPredictionService] = None
        self.df: Optional[pd.DataFrame] = None
        self.model_loaded = False

    def load_system(self, data_path: str, model_path: str) -> str:
        """Load data and model"""
        try:
            # Load data
            logger.info(f"Loading data from {data_path}")
            loader = BookingDataLoader(data_path)
            self.df = loader.load()

            # Load or create model
            model_path = Path(model_path)

            if model_path.exists():
                logger.info(f"Loading existing model from {model_path}")
                self.service = BatteryPredictionService(
                    model_path=str(model_path),
                    historical_data=self.df
                )
            else:
                logger.info("Training new model...")
                self.service = BatteryPredictionService(historical_data=self.df)

                # Train model
                model = BatteryPredictionModel()
                train_df, val_df, test_df = model.prepare_data(self.df)
                metrics = model.train(train_df, val_df)
                test_metrics = model.evaluate(test_df)

                # Save model
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))

                # Load into service
                self.service.load_model(str(model_path))

                return f"✓ Model trained and loaded\nTest MAE: {test_metrics['mae']:.2f}%\nTest R²: {test_metrics['r2']:.4f}"

            self.model_loaded = True
            return f"✓ System loaded successfully\n  - Bookings: {len(self.df)}\n  - Vehicles: {self.df['vehicle_id'].nunique()}\n  - Users: {self.df['user_id'].nunique()}"

        except Exception as e:
            logger.error(f"Error loading system: {e}")
            return f"❌ Error: {str(e)}"

    def make_prediction(
        self,
        vehicle_id: str,
        user_id: str,
        booking_date: str,
        booking_time: str
    ) -> tuple:
        """Make a prediction"""

        if not self.model_loaded:
            return "❌ Please load the system first", None

        try:
            # Parse datetime
            booking_datetime = datetime.fromisoformat(f"{booking_date}T{booking_time}")

            # Make prediction
            result = self.service.predict_battery_at_start(
                vehicle_id=vehicle_id,
                user_id=user_id,
                booking_start_time=booking_datetime,
                booking_id=f"PRED_{datetime.now().timestamp()}",
                update_timeline=False
            )

            # Format output
            output = f"""
## Prediction Result

**Vehicle:** {result['vehicle_id']}
**User:** {result['user_id']}
**Booking Time:** {result['booking_start_time']}

### Predicted Battery
**{result['predicted_battery_percentage']:.1f}%**

### Confidence Interval (95%)
- Lower Bound: {result['confidence_interval']['lower']:.1f}%
- Upper Bound: {result['confidence_interval']['upper']:.1f}%

### Context
- Last Known Battery: {result['last_known_battery']:.1f if result['last_known_battery'] else 'N/A'}%
- Time Since Last Booking: {result['time_since_last_booking_hours']:.1f if result['time_since_last_booking_hours'] else 'N/A'} hours
- Intermediate Bookings: {result['intermediate_bookings_count']}
            """

            # Create visualization
            fig = go.Figure()

            # Add prediction with confidence interval
            fig.add_trace(go.Bar(
                x=['Prediction'],
                y=[result['predicted_battery_percentage']],
                name='Predicted Battery',
                marker_color='green',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[result['confidence_interval']['upper'] - result['predicted_battery_percentage']],
                    arrayminus=[result['predicted_battery_percentage'] - result['confidence_interval']['lower']]
                )
            ))

            if result['last_known_battery']:
                fig.add_trace(go.Bar(
                    x=['Last Known'],
                    y=[result['last_known_battery']],
                    name='Last Known Battery',
                    marker_color='blue'
                ))

            fig.update_layout(
                title='Battery Prediction',
                yaxis_title='Battery %',
                yaxis_range=[0, 100],
                showlegend=True,
                height=400
            )

            return output, fig

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return f"❌ Error: {str(e)}", None

    def get_vehicle_timeline(self, vehicle_id: str) -> tuple:
        """Get vehicle timeline visualization"""

        if not self.model_loaded:
            return "❌ Please load the system first", None

        try:
            # Get vehicle data
            vehicle_data = self.df[self.df['vehicle_id'] == vehicle_id].sort_values('starts_at')

            if len(vehicle_data) == 0:
                return f"❌ No data found for vehicle {vehicle_id}", None

            # Create timeline plot
            fig = go.Figure()

            # Battery at start
            fig.add_trace(go.Scatter(
                x=vehicle_data['starts_at'],
                y=vehicle_data['battery_at_start'],
                mode='lines+markers',
                name='Battery at Start',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))

            # Battery at end
            fig.add_trace(go.Scatter(
                x=vehicle_data['ends_at'],
                y=vehicle_data['battery_at_end'],
                mode='lines+markers',
                name='Battery at End',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title=f'Battery Timeline for {vehicle_id}',
                xaxis_title='Date',
                yaxis_title='Battery %',
                yaxis_range=[0, 100],
                hovermode='x unified',
                height=500
            )

            # Stats
            stats = f"""
## Vehicle Statistics: {vehicle_id}

- **Total Bookings:** {len(vehicle_data)}
- **Average Battery at Start:** {vehicle_data['battery_at_start'].mean():.1f}%
- **Average Battery at End:** {vehicle_data['battery_at_end'].mean():.1f}%
- **Average Battery Drain:** {(vehicle_data['battery_at_start'] - vehicle_data['battery_at_end']).mean():.1f}%
- **Date Range:** {vehicle_data['starts_at'].min()} to {vehicle_data['ends_at'].max()}
            """

            return stats, fig

        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            return f"❌ Error: {str(e)}", None

    def get_model_insights(self) -> tuple:
        """Get model performance insights"""

        if not self.model_loaded:
            return "❌ Please load the system first", None, None

        try:
            # Get feature importance
            importance = self.service.model.get_feature_importance(top_n=15)

            # Create feature importance plot
            fig1 = go.Figure(go.Bar(
                x=importance['importance'],
                y=importance['feature'],
                orientation='h',
                marker_color='steelblue'
            ))

            fig1.update_layout(
                title='Top 15 Most Important Features',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=500,
                yaxis=dict(autorange="reversed")
            )

            # Get prediction stats
            stats = self.service.get_prediction_stats()

            stats_text = f"""
## Model Information

- **Model Type:** LightGBM Regressor
- **Total Features:** {stats['feature_count']}
- **Historical Bookings:** {stats['total_historical_bookings']}
- **Total Vehicles:** {stats['total_vehicles']}
- **Future Bookings Tracked:** {stats['total_future_bookings']}

### Top 5 Features
{importance.head(5).to_string(index=False)}
            """

            # Create battery distribution plot
            fig2 = go.Figure()

            fig2.add_trace(go.Histogram(
                x=self.df['battery_at_start'],
                name='Battery at Start',
                opacity=0.7,
                marker_color='green'
            ))

            fig2.add_trace(go.Histogram(
                x=self.df['battery_at_end'],
                name='Battery at End',
                opacity=0.7,
                marker_color='red'
            ))

            fig2.update_layout(
                title='Battery Distribution',
                xaxis_title='Battery %',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )

            return stats_text, fig1, fig2

        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return f"❌ Error: {str(e)}", None, None

    def what_if_analysis(
        self,
        vehicle_id: str,
        user_id: str,
        hours_ahead: float
    ) -> tuple:
        """What-if analysis: predict battery at different time gaps"""

        if not self.model_loaded:
            return "❌ Please load the system first", None

        try:
            # Get current time (last booking time for this vehicle)
            vehicle_data = self.df[self.df['vehicle_id'] == vehicle_id]
            if len(vehicle_data) == 0:
                return f"❌ No data found for vehicle {vehicle_id}", None

            last_booking_time = vehicle_data['ends_at'].max()

            # Test different time gaps
            time_gaps = np.arange(1, hours_ahead + 1, max(1, hours_ahead / 20))
            predictions = []

            for gap in time_gaps:
                booking_time = last_booking_time + timedelta(hours=gap)

                result = self.service.predict_battery_at_start(
                    vehicle_id=vehicle_id,
                    user_id=user_id,
                    booking_start_time=booking_time,
                    update_timeline=False
                )

                predictions.append({
                    'hours_ahead': gap,
                    'predicted_battery': result['predicted_battery_percentage'],
                    'lower_bound': result['confidence_interval']['lower'],
                    'upper_bound': result['confidence_interval']['upper']
                })

            predictions_df = pd.DataFrame(predictions)

            # Create plot
            fig = go.Figure()

            # Add prediction line
            fig.add_trace(go.Scatter(
                x=predictions_df['hours_ahead'],
                y=predictions_df['predicted_battery'],
                mode='lines',
                name='Predicted Battery',
                line=dict(color='blue', width=3)
            ))

            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=predictions_df['hours_ahead'].tolist() + predictions_df['hours_ahead'].tolist()[::-1],
                y=predictions_df['upper_bound'].tolist() + predictions_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,250,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True
            ))

            fig.update_layout(
                title=f'What-If Analysis: Battery Prediction vs Time Gap<br>Vehicle: {vehicle_id}',
                xaxis_title='Hours from Now',
                yaxis_title='Predicted Battery %',
                yaxis_range=[0, 100],
                height=500,
                hovermode='x unified'
            )

            summary = f"""
## What-If Analysis Summary

**Vehicle:** {vehicle_id}
**User:** {user_id}
**Analysis Range:** 0 to {hours_ahead} hours ahead

### Key Findings:
- **Immediate (1hr):** {predictions_df.iloc[0]['predicted_battery']:.1f}%
- **Mid-range ({hours_ahead/2:.0f}hr):** {predictions_df.iloc[len(predictions_df)//2]['predicted_battery']:.1f}%
- **Max ({hours_ahead:.0f}hr):** {predictions_df.iloc[-1]['predicted_battery']:.1f}%
- **Expected Change:** {predictions_df.iloc[-1]['predicted_battery'] - predictions_df.iloc[0]['predicted_battery']:.1f}%
            """

            return summary, fig

        except Exception as e:
            logger.error(f"Error in what-if analysis: {e}")
            return f"❌ Error: {str(e)}", None

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""

        with gr.Blocks(title="Battery Prediction System", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🔋 Electric Car Battery Prediction System")
            gr.Markdown("Machine Learning system for predicting battery levels at booking start")

            with gr.Tab("🏠 Setup"):
                gr.Markdown("## Load Data and Model")

                with gr.Row():
                    data_path_input = gr.Textbox(
                        label="Data Path",
                        value="data/processed/cleaned_bookings.csv",
                        placeholder="Path to bookings CSV"
                    )
                    model_path_input = gr.Textbox(
                        label="Model Path",
                        value="models/enhanced_battery_predictor.pkl",
                        placeholder="Path to model file"
                    )

                load_btn = gr.Button("Load System", variant="primary")
                load_output = gr.Textbox(label="Status", lines=5)

                load_btn.click(
                    fn=self.load_system,
                    inputs=[data_path_input, model_path_input],
                    outputs=load_output
                )

            with gr.Tab("🔮 Make Prediction"):
                gr.Markdown("## Predict Battery at Booking Start")

                with gr.Row():
                    with gr.Column():
                        pred_vehicle = gr.Textbox(label="Vehicle ID", placeholder="e.g., V001")
                        pred_user = gr.Textbox(label="User ID", placeholder="e.g., U0001")
                        pred_date = gr.Textbox(label="Booking Date", placeholder="YYYY-MM-DD")
                        pred_time = gr.Textbox(label="Booking Time", value="14:00:00")
                        predict_btn = gr.Button("Predict", variant="primary")

                    with gr.Column():
                        pred_output = gr.Markdown(label="Prediction Result")
                        pred_plot = gr.Plot(label="Visualization")

                predict_btn.click(
                    fn=self.make_prediction,
                    inputs=[pred_vehicle, pred_user, pred_date, pred_time],
                    outputs=[pred_output, pred_plot]
                )

            with gr.Tab("📊 Vehicle Timeline"):
                gr.Markdown("## View Vehicle Booking History")

                timeline_vehicle = gr.Textbox(label="Vehicle ID", placeholder="e.g., V001")
                timeline_btn = gr.Button("Get Timeline", variant="primary")

                timeline_output = gr.Markdown(label="Statistics")
                timeline_plot = gr.Plot(label="Timeline Visualization")

                timeline_btn.click(
                    fn=self.get_vehicle_timeline,
                    inputs=timeline_vehicle,
                    outputs=[timeline_output, timeline_plot]
                )

            with gr.Tab("🧠 Model Insights"):
                gr.Markdown("## Model Performance and Feature Importance")

                insights_btn = gr.Button("Get Insights", variant="primary")

                insights_output = gr.Markdown(label="Model Info")
                with gr.Row():
                    importance_plot = gr.Plot(label="Feature Importance")
                    distribution_plot = gr.Plot(label="Battery Distribution")

                insights_btn.click(
                    fn=self.get_model_insights,
                    outputs=[insights_output, importance_plot, distribution_plot]
                )

            with gr.Tab("🎯 What-If Analysis"):
                gr.Markdown("## Scenario Analysis: Battery Prediction vs Time")

                with gr.Row():
                    whatif_vehicle = gr.Textbox(label="Vehicle ID", placeholder="e.g., V001")
                    whatif_user = gr.Textbox(label="User ID", placeholder="e.g., U0001")
                    whatif_hours = gr.Slider(
                        minimum=1,
                        maximum=72,
                        value=24,
                        step=1,
                        label="Hours Ahead"
                    )

                whatif_btn = gr.Button("Run Analysis", variant="primary")
                whatif_output = gr.Markdown(label="Analysis Summary")
                whatif_plot = gr.Plot(label="Prediction Over Time")

                whatif_btn.click(
                    fn=self.what_if_analysis,
                    inputs=[whatif_vehicle, whatif_user, whatif_hours],
                    outputs=[whatif_output, whatif_plot]
                )

            with gr.Tab("📖 API Documentation"):
                gr.Markdown("""
## API Documentation for Laravel Integration

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Predict Battery (Single)
```
POST /api/v1/predict
```

**Request Body:**
```json
{
  "vehicle_id": "V001",
  "user_id": "U0042",
  "booking_start_time": "2024-12-25T14:30:00Z",
  "booking_id": "BK123456",
  "update_timeline": true
}
```

**Response:**
```json
{
  "booking_id": "BK123456",
  "vehicle_id": "V001",
  "predicted_battery_percentage": 78.5,
  "confidence_interval": {
    "lower": 68.5,
    "upper": 88.5,
    "confidence_level": 0.95
  },
  ...
}
```

#### 2. Batch Predictions
```
POST /api/v1/predict/batch
```

#### 3. Booking Created (Update Timeline)
```
POST /api/v1/booking/created
```

#### 4. Get Vehicle Timeline
```
GET /api/v1/vehicle/{vehicle_id}/timeline
```

#### 5. Model Info
```
GET /api/v1/model/info
```

#### 6. Health Check
```
GET /health
```

### Laravel Example
```php
$response = Http::post('http://localhost:8000/api/v1/predict', [
    'vehicle_id' => $vehicle->id,
    'user_id' => $user->id,
    'booking_start_time' => $booking->starts_at->toIso8601String(),
    'booking_id' => $booking->id,
    'update_timeline' => true
]);

$prediction = $response->json();
$batteryLevel = $prediction['predicted_battery_percentage'];
```
                """)

        return demo


def launch_dashboard():
    """Launch the Gradio dashboard"""
    dashboard = BatteryPredictionDashboard()
    demo = dashboard.create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    launch_dashboard()
