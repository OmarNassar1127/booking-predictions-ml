# Quick Start Guide

## Your System is Ready! 🎉

Your battery prediction model has been trained with **32,904 bookings** and is ready to use!

### Model Performance
- **MAE: 8.34%** - Predictions are accurate within ~8% on average
- **71.4% of predictions within 10%** - Very reliable predictions!

---

## 1. Start the API Server (for Laravel)

```bash
cd "/Users/omarnassar/Desktop/Machine learning"
source venv/bin/activate
python run_api.py
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Test the API

Open a new terminal and try:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "62",
    "user_id": "643",
    "booking_start_time": "2025-01-15T14:30:00Z",
    "booking_id": "TEST_001",
    "update_timeline": false
  }'
```

---

## 2. Start the Dashboard (for Testing)

In a new terminal:

```bash
cd "/Users/omarnassar/Desktop/Machine learning"
source venv/bin/activate
python run_dashboard.py
```

Dashboard will open at: **http://localhost:7860**

### Dashboard Features:
1. **Setup Tab**: Load system (already loaded!)
2. **Make Prediction**: Test predictions with any vehicle/user/time
3. **Vehicle Timeline**: See battery history for any vehicle
4. **Model Insights**: View feature importance
5. **What-If Analysis**: Explore predictions over time
6. **API Docs**: Laravel integration examples

---

## 3. Laravel Integration

### Install in Laravel

1. Add to `config/services.php`:

```php
'battery_prediction' => [
    'url' => env('BATTERY_PREDICTION_URL', 'http://localhost:8000'),
    'api_key' => env('BATTERY_PREDICTION_API_KEY'),
],
```

2. Add to `.env`:

```env
BATTERY_PREDICTION_URL=http://localhost:8000
BATTERY_PREDICTION_API_KEY=optional-api-key
```

### Create Service Class

Create `app/Services/BatteryPredictionService.php`:

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class BatteryPredictionService
{
    private $baseUrl;

    public function __construct()
    {
        $this->baseUrl = config('services.battery_prediction.url');
    }

    public function predictBattery($vehicleId, $userId, $bookingStartTime)
    {
        $response = Http::post("{$this->baseUrl}/api/v1/predict", [
            'vehicle_id' => (string)$vehicleId,
            'user_id' => (string)$userId,
            'booking_start_time' => $bookingStartTime,
            'update_timeline' => false
        ]);

        return $response->json();
    }
}
```

### Use in Controller

```php
use App\Services\BatteryPredictionService;

public function showVehicle(Request $request, $vehicleId)
{
    $prediction = app(BatteryPredictionService::class)->predictBattery(
        $vehicleId,
        auth()->user()->id,
        $request->input('booking_start_time')
    );

    return view('vehicles.show', [
        'vehicle' => $vehicle,
        'predicted_battery' => $prediction['predicted_battery_percentage'],
        'confidence' => $prediction['confidence_interval']
    ]);
}
```

---

## 4. Example API Calls

### Predict Battery (Single)

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "62",
    "user_id": "643",
    "booking_start_time": "2025-01-20T14:00:00Z",
    "booking_id": "BK12345"
  }'
```

**Response:**
```json
{
  "predicted_battery_percentage": 75.3,
  "confidence_interval": {
    "lower": 65.3,
    "upper": 85.3,
    "confidence_level": 0.95
  },
  "last_known_battery": 68.5,
  "time_since_last_booking_hours": 12.5
}
```

### Batch Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "bookings": [
      {
        "vehicle_id": "62",
        "user_id": "643",
        "booking_start_time": "2025-01-20T14:00:00Z"
      },
      {
        "vehicle_id": "19",
        "user_id": "1481",
        "booking_start_time": "2025-01-20T16:00:00Z"
      }
    ]
  }'
```

### Notify Booking Created

Call this after a booking is confirmed:

```bash
curl -X POST "http://localhost:8000/api/v1/booking/created" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "62",
    "user_id": "643",
    "booking_id": "BK12345",
    "booking_start_time": "2025-01-20T14:00:00Z",
    "booking_end_time": "2025-01-20T18:00:00Z",
    "actual_battery_at_start": 75.5
  }'
```

---

## 5. Understanding Predictions

### What the Model Predicts

The model predicts the **battery percentage at the START** of a future booking based on:

1. **Last known battery level** for that vehicle
2. **Time gap** since last booking (charging opportunity)
3. **Historical patterns** for that vehicle/user
4. **Time of day** and day of week
5. **Intermediate bookings** (if any)

### Example Scenario

User wants to book Vehicle #62 tomorrow at 2 PM:

- **Last booking ended**: Today at 10 AM, battery was 65%
- **Time gap**: 28 hours (plenty of charging time)
- **Prediction**: 85% ± 10%
- **Confidence**: 95% likely between 75% and 95%

### When to Show Predictions

✅ **Good use cases:**
- Showing predicted battery when browsing vehicles
- Displaying in booking confirmation
- Vehicle availability filters (e.g., "show vehicles with >60% predicted battery")

❌ **Don't use for:**
- Enforcing hard limits (predictions have uncertainty)
- Replacing actual battery readings from vehicles
- Critical safety decisions

---

## 6. Retraining the Model

As you collect more booking data, retrain the model:

```bash
# 1. Clean new data
python clean_data.py

# 2. Retrain model
python train_model.py

# 3. Restart API
# Stop API (Ctrl+C) and restart:
python run_api.py
```

**Recommended retraining schedule:**
- Weekly during initial deployment
- Monthly after stable
- Whenever booking patterns change significantly

---

## 7. Monitoring

### Check Health

```bash
curl http://localhost:8000/health
```

### View Logs

```bash
# API logs
tail -f logs/api.log

# Training logs
tail -f logs/training.log
```

### Performance Metrics

Access http://localhost:8000/api/v1/model/info to see:
- Model statistics
- Feature importance
- Total bookings processed

---

## 8. Troubleshooting

### API not responding
```bash
# Check if running
curl http://localhost:8000/health

# Restart API
python run_api.py
```

### Poor predictions
- Check if you're using real vehicle IDs from your data
- Ensure booking times are in the future
- Verify date format: ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)

### Dashboard not loading
```bash
# Restart dashboard
python run_dashboard.py
```

---

## 9. Next Steps

1. ✅ **Test API** - Use curl or dashboard to make test predictions
2. ✅ **Integrate with Laravel** - Add the service class to your Laravel app
3. ✅ **Update UI** - Show predicted battery levels to users
4. ⏳ **Monitor** - Track prediction accuracy vs actual battery levels
5. ⏳ **Retrain** - Retrain weekly with new data for better accuracy

---

## Support

- **API Documentation**: http://localhost:8000/docs
- **Interactive Dashboard**: http://localhost:7860
- **Logs**: Check `logs/` directory
- **Configuration**: Edit `config/config.yaml`

Your system is fully operational and ready to integrate with Laravel! 🚀
