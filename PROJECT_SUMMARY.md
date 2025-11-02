# Battery Prediction System - Project Summary

## ✅ What Was Built

A complete machine learning system for predicting electric car battery levels at the start of future bookings.

---

## 📊 System Performance

Trained on **32,904 bookings** from your actual data:

| Metric | Value | Meaning |
|--------|-------|---------|
| **MAE** | 8.34% | Average error is 8.34 percentage points |
| **Within 5%** | 58.7% | More than half are very accurate |
| **Within 10%** | 71.4% | Over 70% are highly accurate |
| **R² Score** | 0.5909 | Model explains 59% of variance |

**Example:** If actual battery is 70%, prediction will likely be between 61.66% - 78.34%

---

## 🎯 Key Features

### 1. **Accurate Predictions**
- Uses LightGBM machine learning model
- Considers 50+ features including time, vehicle history, user behavior
- Provides confidence intervals (95% confidence level)

### 2. **Dynamic Updates**
- When a new booking is created, automatically updates predictions for future bookings
- Maintains vehicle timeline to track state
- Handles intermediate bookings

### 3. **REST API for Laravel**
- FastAPI server with full REST API
- Comprehensive API documentation
- Easy integration with HTTP requests

### 4. **Interactive Dashboard**
- Gradio-based web interface
- Test predictions in real-time
- Visualize vehicle timelines
- What-if analysis

---

## 🏗️ Architecture

```
┌──────────────┐
│ Laravel App  │
└──────┬───────┘
       │ HTTP POST /api/v1/predict
       ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  FastAPI     │────▶│  Prediction   │────▶│  LightGBM    │
│  REST API    │     │   Service     │     │    Model     │
└──────────────┘     └───────┬───────┘     └──────────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │   Timeline    │
                     │   Manager     │
                     └───────────────┘
```

---

## 📁 Project Structure

```
Machine learning/
├── config/
│   └── config.yaml                    # Configuration
├── data/
│   ├── raw/
│   │   └── prepared_bookings.csv     # Your original data (33k bookings)
│   └── processed/
│       └── cleaned_bookings.csv      # Cleaned data (32.9k bookings)
├── models/
│   └── battery_predictor.pkl         # Trained model (ready to use!)
├── src/
│   ├── api/                          # FastAPI application
│   │   ├── main.py                   # API server
│   │   └── models.py                 # Request/response models
│   ├── dashboard/
│   │   └── app.py                    # Gradio dashboard
│   ├── data/
│   │   ├── data_loader.py            # Load and validate data
│   │   └── data_generator.py         # Generate synthetic data
│   ├── features/
│   │   └── feature_engineer.py       # Feature engineering (50+ features)
│   ├── models/
│   │   ├── battery_predictor.py      # ML model
│   │   ├── timeline_manager.py       # Vehicle timeline tracking
│   │   └── prediction_service.py     # Main prediction service
│   └── utils/
│       ├── config_loader.py          # Load configuration
│       └── logger.py                 # Logging utilities
├── logs/                             # Log files
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Data exploration
├── train_model.py                    # Train the model
├── run_api.py                        # Start API server
├── run_dashboard.py                  # Start dashboard
├── clean_data.py                     # Clean raw data
├── requirements.txt                  # Python dependencies
├── README.md                         # Full documentation
├── QUICK_START.md                    # Quick start guide
└── PROJECT_SUMMARY.md               # This file
```

---

## 🚀 How to Use

### Option 1: API (for Laravel Integration)

**Start the API:**
```bash
cd "/Users/omarnassar/Desktop/Machine learning"
source venv/bin/activate
python run_api.py
```

**Make prediction from Laravel:**
```php
$response = Http::post('http://localhost:8000/api/v1/predict', [
    'vehicle_id' => '62',
    'user_id' => '643',
    'booking_start_time' => '2025-01-20T14:00:00Z'
]);

$battery = $response->json()['predicted_battery_percentage'];
```

### Option 2: Dashboard (for Testing)

**Start the Dashboard:**
```bash
cd "/Users/omarnassar/Desktop/Machine learning"
source venv/bin/activate
python run_dashboard.py
```

Open http://localhost:7860 in your browser.

---

## 🔍 What the Model Learned

### Top 5 Most Important Features:

1. **Previous Battery Level** (501 importance)
   - Most critical factor
   - Last known battery state for the vehicle

2. **Expected Charging Potential** (371)
   - How much the vehicle could charge in the time gap
   - Calculated from time since last booking

3. **Battery Drain per KM** (317)
   - Vehicle's efficiency pattern
   - Learned from historical usage

4. **Time Since Last Booking** (220)
   - Longer gaps = more charging time
   - Strong predictor of battery increase

5. **Distance Traveled** (204)
   - Trip length affects battery drain
   - Combined with efficiency metrics

### How Predictions Work:

**Example Scenario:**
```
Current State:
  - Vehicle #62
  - Last booking ended at 10:00 AM, battery was 65%
  - User wants to book at 2:00 PM next day (28 hours later)

Model Reasoning:
  1. Starting point: 65% battery
  2. Time gap: 28 hours → high charging potential
  3. Historical pattern: Vehicle #62 usually charges to 80-90% overnight
  4. User behavior: User #643 typically uses vehicles moderately
  5. Time of day: 2 PM is peak usage time

Prediction:
  - Predicted Battery: 85%
  - Confidence Interval: 75% - 95%
  - Confidence Level: 95%
```

---

## 📈 Data Insights

From your 32,904 bookings:

| Metric | Value |
|--------|-------|
| **Vehicles** | 80 unique vehicles |
| **Users** | 2,238 unique users |
| **Date Range** | Oct 2024 - Oct 2025 (1 year) |
| **Avg Duration** | 3.7 hours per booking |
| **Avg Distance** | 47.5 km per booking |
| **Avg Battery Start** | 70.7% |
| **Avg Battery End** | 61.1% |
| **Avg Battery Drain** | -9.6% per booking |

---

## 🔧 Configuration

All settings in `config/config.yaml`:

```yaml
# Key settings you might want to change:

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"  # Change to your Laravel domain in production

model:
  hyperparameters:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 7

prediction:
  confidence_level: 0.95
  max_prediction_horizon_days: 30
```

---

## 🔄 Retraining the Model

As you collect more data, retrain for better accuracy:

```bash
# 1. Update data file (add new bookings to prepared_bookings.csv)
# 2. Clean data
python clean_data.py

# 3. Retrain
python train_model.py

# 4. Restart API
python run_api.py
```

**Retraining Schedule:**
- **First month**: Weekly (as you collect more data)
- **After stable**: Monthly
- **On demand**: When booking patterns change

---

## 🎯 Use Cases

### 1. **Vehicle Selection**
Show predicted battery when users browse vehicles:
```php
foreach ($availableVehicles as $vehicle) {
    $prediction = $predictionService->predict(
        $vehicle->id,
        $user->id,
        $bookingStartTime
    );

    $vehicle->predicted_battery = $prediction['predicted_battery_percentage'];
}
```

### 2. **Booking Confirmation**
Display expected battery in booking confirmation:
```
Your Booking Details:
  Vehicle: #62
  Start: Jan 20, 2025 2:00 PM
  Expected Battery: 85% (±10%)
```

### 3. **Low Battery Warnings**
Warn users if predicted battery is low:
```php
if ($prediction['predicted_battery_percentage'] < 30) {
    return "Warning: This vehicle may have low battery at your booking time.";
}
```

### 4. **Availability Filters**
Let users filter by minimum battery:
```php
$vehicles = Vehicle::whereHas('predictions', function($q) {
    $q->where('predicted_battery', '>=', 60);
});
```

---

## 📊 API Endpoints

### Main Endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/predict` | Predict battery for single booking |
| `POST` | `/api/v1/predict/batch` | Predict for multiple bookings |
| `POST` | `/api/v1/booking/created` | Update timeline when booking created |
| `GET` | `/api/v1/vehicle/{id}/timeline` | Get vehicle history + predictions |
| `GET` | `/api/v1/model/info` | Model statistics |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation |

Full API docs: http://localhost:8000/docs (when API is running)

---

## 🐛 Troubleshooting

### API Returns 503 "Service Unavailable"
**Problem:** Model not loaded

**Solution:**
```bash
# Check if model file exists
ls -lh models/battery_predictor.pkl

# If missing, retrain:
python train_model.py
```

### Predictions Seem Inaccurate
**Possible causes:**
1. Using vehicle IDs not in training data → Model uses averages
2. Booking time too far in future → Uncertainty increases
3. Recent pattern changes → Retrain model

**Solution:** Retrain with latest data

### Dashboard Won't Load
**Solution:**
```bash
# Make sure you're in the right directory
cd "/Users/omarnassar/Desktop/Machine learning"

# Activate environment
source venv/bin/activate

# Run dashboard
python run_dashboard.py
```

---

## 🔐 Security Notes

### For Production:

1. **Enable Authentication:**
   ```yaml
   # config/config.yaml
   api:
     authentication:
       enabled: true
       api_key: "your-secret-key"
   ```

2. **Update CORS:**
   ```yaml
   api:
     cors_origins:
       - "https://yourdomain.com"  # Your Laravel domain
   ```

3. **Use HTTPS:**
   - Deploy behind nginx/caddy
   - Use SSL certificates

4. **Rate Limiting:**
   - Already configured in config.yaml
   - max_requests_per_minute: 100

---

## 📝 Next Steps

1. ✅ **Test API** - Use curl or Postman to test predictions
2. ✅ **Test Dashboard** - Open http://localhost:7860 and explore
3. ⏳ **Integrate Laravel** - Add service class to your Laravel app
4. ⏳ **Update UI** - Show predicted battery to users
5. ⏳ **Monitor** - Track actual vs predicted battery
6. ⏳ **Retrain** - Retrain weekly with new data

---

## 📚 Documentation

- **README.md** - Full system documentation
- **QUICK_START.md** - Quick start guide for immediate use
- **API Docs** - http://localhost:8000/docs (interactive)
- **This File** - High-level summary

---

## 🎉 Summary

**You now have:**
- ✅ Trained ML model (8.34% MAE)
- ✅ REST API server (FastAPI)
- ✅ Interactive dashboard (Gradio)
- ✅ Complete Laravel integration guide
- ✅ Full documentation

**The system can:**
- ✅ Predict battery levels with 71.4% accuracy within 10%
- ✅ Handle dynamic updates when bookings are created
- ✅ Provide confidence intervals for predictions
- ✅ Scale to handle multiple vehicles and users
- ✅ Integrate seamlessly with Laravel

**Ready to integrate with your Laravel backend!** 🚀

---

## 💬 Support

If you need help:
1. Check the logs: `tail -f logs/api.log`
2. Review API docs: http://localhost:8000/docs
3. Test in dashboard: http://localhost:7860
4. Check configuration: `config/config.yaml`

---

*Generated: November 2, 2025*
*Model trained with 32,904 bookings from your production data*
