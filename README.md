# Electric Car Battery Prediction System

A machine learning system that predicts battery percentage at the start of electric car bookings, with support for dynamic updates when new bookings are made.

## Features

- **Accurate Predictions**: Uses LightGBM to predict battery levels with confidence intervals
- **Dynamic Updates**: Automatically recalculates predictions when new bookings are added
- **FastAPI REST API**: Easy integration with Laravel or any backend
- **Interactive Dashboard**: Gradio-based UI for testing and visualization
- **Real-time Timeline**: Track vehicle booking history and future predictions
- **What-If Analysis**: Explore different booking scenarios

## System Architecture

```
┌─────────────────┐
│  Laravel Backend │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐     ┌──────────────────┐
│   FastAPI REST  │────►│  ML Prediction   │
│      API        │     │     Service      │
└─────────────────┘     └────────┬─────────┘
                                 │
         ┌───────────────────────┴─────────────┐
         ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│  LightGBM Model │                  │ Timeline Manager│
│  + Features     │                  │ (Dynamic Update)│
└─────────────────┘                  └─────────────────┘
```

## Installation

### 1. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data

Place your booking data in `data/raw/bookings.csv` with these columns:
- `booking_id`: Unique booking identifier
- `vehicle_id`: Vehicle identifier
- `user_id`: User identifier
- `starts_at`: Booking start datetime (ISO format)
- `ends_at`: Booking end datetime (ISO format)
- `battery_at_start`: Battery percentage at start (0-100)
- `battery_at_end`: Battery percentage at end (0-100)
- `mileage_at_start`: Odometer reading at start
- `mileage_at_end`: Odometer reading at end

Optional fields:
- `account_community_id`: Community/account ID
- `charging_at_end`: Whether vehicle was charging at booking end

## Quick Start

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load and validate your data
- Train the LightGBM model
- Evaluate performance
- Save the model to `models/battery_predictor.pkl`

Expected output:
```
Test Metrics:
  MAE: ~8-12% (depending on your data quality)
  Within 10%: 70-85% of predictions
```

### 2. Start the API Server

```bash
python run_api.py
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 3. Start the Dashboard (Optional)

```bash
python run_dashboard.py
```

Dashboard will open at: http://localhost:7860

## API Usage

### Authentication (Optional)

Set `API_KEY` in `.env` and enable authentication in `config/config.yaml`:

```yaml
api:
  authentication:
    enabled: true
    api_key_header: "X-API-Key"
```

### Endpoint 1: Predict Battery for Single Booking

**POST** `/api/v1/predict`

**Request:**
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
  "user_id": "U0042",
  "booking_start_time": "2024-12-25T14:30:00",
  "predicted_battery_percentage": 78.5,
  "confidence_interval": {
    "lower": 68.5,
    "upper": 88.5,
    "confidence_level": 0.95
  },
  "last_known_battery": 65.2,
  "time_since_last_booking_hours": 8.5,
  "intermediate_bookings_count": 0,
  "timestamp": "2024-12-20T10:00:00"
}
```

### Endpoint 2: Batch Predictions

**POST** `/api/v1/predict/batch`

**Request:**
```json
{
  "bookings": [
    {
      "vehicle_id": "V001",
      "user_id": "U0042",
      "booking_start_time": "2024-12-25T14:30:00Z",
      "booking_id": "BK123"
    },
    {
      "vehicle_id": "V002",
      "user_id": "U0043",
      "booking_start_time": "2024-12-25T16:00:00Z",
      "booking_id": "BK124"
    }
  ],
  "update_timeline": false
}
```

### Endpoint 3: Booking Created (Update Timeline)

Call this when a booking is confirmed to update future predictions:

**POST** `/api/v1/booking/created`

**Request:**
```json
{
  "vehicle_id": "V001",
  "user_id": "U0042",
  "booking_id": "BK123456",
  "booking_start_time": "2024-12-25T14:30:00Z",
  "booking_end_time": "2024-12-25T18:30:00Z",
  "actual_battery_at_start": 78.5
}
```

**Response:**
```json
{
  "new_booking": { /* prediction for new booking */ },
  "affected_bookings_count": 3,
  "updated_predictions": [ /* updated predictions for affected bookings */ ]
}
```

### Endpoint 4: Get Vehicle Timeline

**GET** `/api/v1/vehicle/{vehicle_id}/timeline`

Query Parameters:
- `include_historical`: boolean (default: true)
- `include_future`: boolean (default: true)

## Laravel Integration

### 1. Installation

```bash
composer require guzzlehttp/guzzle
```

### 2. Create Service Class

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class BatteryPredictionService
{
    private $baseUrl;
    private $apiKey;

    public function __construct()
    {
        $this->baseUrl = config('services.battery_prediction.url', 'http://localhost:8000');
        $this->apiKey = config('services.battery_prediction.api_key');
    }

    public function predictBattery($vehicleId, $userId, $bookingStartTime, $bookingId = null)
    {
        $response = Http::withHeaders([
            'X-API-Key' => $this->apiKey,
        ])->post("{$this->baseUrl}/api/v1/predict", [
            'vehicle_id' => $vehicleId,
            'user_id' => $userId,
            'booking_start_time' => $bookingStartTime,
            'booking_id' => $bookingId,
            'update_timeline' => false  // Set to true if you want to track this prediction
        ]);

        if ($response->successful()) {
            return $response->json();
        }

        throw new \Exception("Prediction failed: " . $response->body());
    }

    public function notifyBookingCreated($booking)
    {
        $response = Http::withHeaders([
            'X-API-Key' => $this->apiKey,
        ])->post("{$this->baseUrl}/api/v1/booking/created", [
            'vehicle_id' => $booking->vehicle_id,
            'user_id' => $booking->user_id,
            'booking_id' => $booking->id,
            'booking_start_time' => $booking->starts_at->toIso8601String(),
            'booking_end_time' => $booking->ends_at->toIso8601String(),
            'actual_battery_at_start' => $booking->battery_at_start
        ]);

        return $response->json();
    }
}
```

### 3. Usage in Controller

```php
<?php

namespace App\Http\Controllers;

use App\Services\BatteryPredictionService;
use Illuminate\Http\Request;

class BookingController extends Controller
{
    private $predictionService;

    public function __construct(BatteryPredictionService $predictionService)
    {
        $this->predictionService = $predictionService;
    }

    public function showAvailableVehicles(Request $request)
    {
        $vehicles = Vehicle::available()->get();
        $bookingStartTime = $request->input('start_time');

        foreach ($vehicles as $vehicle) {
            try {
                $prediction = $this->predictionService->predictBattery(
                    $vehicle->id,
                    auth()->user()->id,
                    $bookingStartTime
                );

                $vehicle->predicted_battery = $prediction['predicted_battery_percentage'];
                $vehicle->prediction_confidence = $prediction['confidence_interval'];
            } catch (\Exception $e) {
                // Handle error - maybe show default value
                $vehicle->predicted_battery = null;
            }
        }

        return view('bookings.available', compact('vehicles'));
    }

    public function store(Request $request)
    {
        // Create booking
        $booking = Booking::create($request->validated());

        // Notify prediction service (updates timeline for other bookings)
        try {
            $this->predictionService->notifyBookingCreated($booking);
        } catch (\Exception $e) {
            // Log error but don't fail the booking
            \Log::error('Failed to notify prediction service: ' . $e->getMessage());
        }

        return redirect()->route('bookings.show', $booking);
    }
}
```

### 4. Configuration

Add to `config/services.php`:

```php
'battery_prediction' => [
    'url' => env('BATTERY_PREDICTION_URL', 'http://localhost:8000'),
    'api_key' => env('BATTERY_PREDICTION_API_KEY'),
],
```

Add to `.env`:

```env
BATTERY_PREDICTION_URL=http://localhost:8000
BATTERY_PREDICTION_API_KEY=your-secret-key
```

## Dashboard Usage

The Gradio dashboard provides:

1. **Setup Tab**: Load data and model
2. **Make Prediction Tab**: Test single predictions
3. **Vehicle Timeline Tab**: View historical battery levels
4. **Model Insights Tab**: See feature importance and performance
5. **What-If Analysis Tab**: Explore predictions at different time gaps
6. **API Documentation Tab**: Integration examples

## Model Performance

Expected performance metrics (depends on data quality):

- **MAE**: 8-12% (Mean Absolute Error)
- **Within 5%**: 40-60% of predictions
- **Within 10%**: 70-85% of predictions
- **R²**: 0.65-0.80

### Key Factors Affecting Prediction:
1. Time since last booking (charging time)
2. Historical battery patterns for vehicle
3. User behavior patterns
4. Time of day and day of week
5. Previous battery level

## Project Structure

```
.
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed features
│   └── predictions/             # Prediction outputs
├── models/
│   └── battery_predictor.pkl    # Trained model
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── api/                     # FastAPI application
│   ├── dashboard/               # Gradio dashboard
│   ├── data/                    # Data loading utilities
│   ├── features/                # Feature engineering
│   ├── models/                  # ML models and prediction service
│   └── utils/                   # Utilities
├── logs/                        # Log files
├── train_model.py               # Training script
├── run_api.py                   # Start API server
├── run_dashboard.py             # Start dashboard
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- API settings (host, port, CORS)
- Feature engineering parameters
- Logging settings
- Paths

## Troubleshooting

### Model not loading
- Ensure you've run `train_model.py` first
- Check that `models/battery_predictor.pkl` exists
- Verify the path in `config/config.yaml`

### API returning 503
- Check if model is loaded: `GET /health`
- Verify data file exists at configured path
- Check logs in `logs/api.log`

### Poor prediction accuracy
- Need more training data (minimum ~10k bookings recommended)
- Check data quality (missing values, outliers)
- Run the EDA notebook to analyze patterns
- Consider adding more features or tuning hyperparameters

### Memory issues
- Reduce `n_estimators` in model config
- Process data in batches
- Use smaller data sample for testing

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Features

1. Add feature engineering logic in `src/features/feature_engineer.py`
2. Retrain model: `python train_model.py`
3. Test via dashboard or API

### Retraining Model

To retrain with new data:

```bash
# 1. Add new data to data/raw/bookings.csv
# 2. Retrain
python train_model.py

# 3. Restart API
python run_api.py
```

## Production Deployment

### Docker (Recommended)

Coming soon - Dockerfile and docker-compose.yml

### Manual Deployment

1. Use production WSGI server (gunicorn/uvicorn workers)
2. Enable authentication
3. Set up CORS for your Laravel domain
4. Use Redis for prediction caching
5. Set up monitoring and logging
6. Configure backup and model versioning

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review API documentation at `/docs`
- Check configuration in `config/config.yaml`

## License

MIT License - See LICENSE file for details
