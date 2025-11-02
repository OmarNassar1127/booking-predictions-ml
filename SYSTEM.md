# Battery Prediction ML System - Architecture Documentation

> **Last Updated**: 2025-11-02
> **Version**: 2.0 (Event-Driven Self-Learning System)
> **Status**: 🚧 In Development

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Current Implementation Status](#current-implementation-status)
4. [API Endpoints](#api-endpoints)
5. [Event-Driven Flow](#event-driven-flow)
6. [Database Schema](#database-schema)
7. [Laravel Integration Guide](#laravel-integration-guide)
8. [Deployment](#deployment)

---

## System Overview

### Purpose
A self-learning ML system that predicts electric vehicle battery levels at booking start times. The system learns continuously from actual outcomes and improves accuracy over time.

### Key Features
- **Real-time predictions** from current vehicle telemetry
- **Event-driven architecture** with Laravel webhooks
- **Self-healing ML pipeline** that learns from every booking
- **Cascade predictions** that update when bookings change
- **86.1% accuracy** (predictions within 10% of actual)
- **Hybrid learning**: Immediate pattern updates + weekly model retraining

### Technology Stack
- **ML Framework**: LightGBM (Gradient Boosting)
- **API**: FastAPI (Python 3.9+)
- **Database**: SQLite (prediction tracking)
- **Dashboard**: Gradio
- **Integration**: REST API webhooks with Laravel

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Laravel Backend                          │
│  (Booking Management, User Interface, Vehicle Telemetry)        │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ Events (webhooks)                  │ Predictions
             │ - booking.created                  │ (JSON responses)
             │ - booking.started                  │
             │ - booking.ended                    │
             │ - booking.modified                 │
             │ - booking.cancelled                │
             ▼                                    ▲
┌─────────────────────────────────────────────────────────────────┐
│                     ML Prediction API                           │
│                    (FastAPI - Port 8000)                        │
├─────────────────────────────────────────────────────────────────┤
│  Event Handlers          │  Prediction Service                  │
│  ├─ booking.created      │  ├─ Historical Mode                  │
│  ├─ booking.started      │  ├─ Real-time Mode (with current %)  │
│  ├─ booking.ended        │  └─ Cascade Re-prediction            │
│  ├─ booking.modified     │                                      │
│  └─ booking.cancelled    │  Pattern Learning Engine             │
│                          │  ├─ Update drain rates               │
│  Tracking Database       │  ├─ Update charging frequencies      │
│  ├─ Predictions          │  └─ Update user patterns             │
│  ├─ Actuals              │                                      │
│  ├─ Vehicle Patterns     │  LightGBM Model (86.1% accuracy)     │
│  └─ Accuracy Metrics     │  └─ 70+ features including charging  │
└─────────────────────────────────────────────────────────────────┘
             │                                    ▲
             │ Callback webhooks                  │
             │ (updated predictions)              │
             ▼                                    │
┌─────────────────────────────────────────────────────────────────┐
│                        Laravel Backend                          │
│              (Stores predictions in database)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Implementation Status

### ✅ Phase 1: Core Prediction System (COMPLETED)
- [x] Data cleaning pipeline
- [x] Feature engineering (70+ features)
- [x] LightGBM model training (86.1% within 10%)
- [x] Historical prediction mode
- [x] Real-time prediction mode (with current battery)
- [x] Cascade prediction through intermediate bookings
- [x] Confidence intervals (95%)

**Files**:
- `src/models/prediction_service.py` - Main prediction logic
- `src/models/battery_predictor.py` - Model wrapper
- `src/features/enhanced_features.py` - Charging-specific features
- `models/enhanced_battery_predictor.pkl` - Trained model (86.1%)

### ✅ Phase 2: API Foundation (COMPLETED)
- [x] FastAPI server setup
- [x] `/api/v1/predict` endpoint (dual-mode)
- [x] Request/response models with Pydantic
- [x] CORS configuration for Laravel
- [x] API authentication (optional)
- [x] Gradio dashboard (port 7860)

**Files**:
- `src/api/main.py` - FastAPI app
- `src/api/models.py` - Pydantic schemas
- `src/dashboard/app.py` - Gradio UI

### ✅ Phase 3: Tracking Infrastructure (COMPLETED)
- [x] SQLite database for predictions vs actuals
- [x] Prediction storage with metadata
- [x] Actual outcome recording
- [x] Accuracy metrics calculation
- [x] Vehicle/user pattern tables

**Files**:
- `src/database/tracking.py` - Database ORM
- Database location: `data/tracking/predictions.db`

**Tables**:
- `predictions` - Stores all predictions with actuals
- `booking_outcomes` - Complete booking history for training
- `vehicle_patterns` - Per-vehicle drain/charging rates
- `user_patterns` - User behavior patterns
- `accuracy_metrics` - Time-series accuracy tracking

### ✅ Phase 4: Event-Driven System (MOSTLY COMPLETED)

#### ✅ Completed
- [x] Tracking database schema (5 tables)
- [x] Event webhook endpoints (5 events)
- [x] Pattern learning engine (EMA-based updates)
- [x] Cascade re-prediction system
- [x] Integration with main API

**Files**:
- `src/api/events.py` - Event webhook handlers
- `src/learning/pattern_updater.py` - Pattern learning with EMA
- `src/learning/cascade_predictor.py` - Cascade re-prediction logic

**How it works**:
1. **Booking Created** → Make prediction → Store in DB
2. **Booking Started** → Compare actual vs predicted → Adjust patterns if error > 5% → Cascade re-predict future bookings
3. **Booking Ended** → Store outcome → Update patterns (drain rates, charging frequency) → Cascade re-predict
4. **Booking Modified/Cancelled** → Re-predict affected bookings

#### 🚧 In Progress
- [ ] Laravel callback webhooks (send updated predictions back)

#### 📋 Planned
- [ ] Batch retraining pipeline (weekly)
- [ ] A/B testing for models
- [ ] Accuracy monitoring dashboard

---

## API Endpoints

### Core Prediction Endpoints

#### `POST /api/v1/predict`
**Description**: Predict battery at booking start (supports both historical and real-time modes)

**Request (Historical Mode)**:
```json
{
  "vehicle_id": "78",
  "user_id": "860",
  "booking_start_time": "2025-11-03T19:00:00Z",
  "booking_id": "BK123456"
}
```

**Request (Real-Time Mode)**:
```json
{
  "vehicle_id": "78",
  "user_id": "860",
  "booking_start_time": "2025-11-03T19:00:00Z",
  "booking_id": "BK123456",
  "current_battery_level": 65.5,
  "current_timestamp": "2025-11-02T12:00:00Z",
  "intermediate_bookings": [
    {
      "starts_at": "2025-11-02T13:00:00Z",
      "ends_at": "2025-11-02T15:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": "78",
  "predicted_battery_percentage": 58.3,
  "confidence_interval": {
    "lower": 48.3,
    "upper": 68.3,
    "confidence_level": 0.95
  },
  "prediction_method": "real_time_cascade",
  "cascade_steps": [
    {
      "type": "idle",
      "duration_hours": 1.0,
      "battery_change": -0.5,
      "battery_after": 65.0
    },
    {
      "type": "booking",
      "duration_hours": 2.0,
      "battery_change": -10.0,
      "battery_after": 55.0
    }
  ],
  "timestamp": "2025-11-02T12:00:00Z"
}
```

### Event Webhook Endpoints ✅

#### `POST /api/v1/events/booking-created`
**Description**: Laravel fires when a new booking is created

**Request**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": 78,
  "user_id": 860,
  "starts_at": "2025-11-03T19:00:00Z",
  "ends_at": "2025-11-03T21:00:00Z",
  "current_battery_level": 65.5,
  "current_timestamp": "2025-11-02T12:00:00Z"
}
```

**Response**:
```json
{
  "status": "success",
  "prediction_id": "PRED_BK123456_123456789",
  "predicted_battery": 58.3,
  "confidence_interval": {
    "lower": 48.3,
    "upper": 68.3
  },
  "prediction_method": "real_time",
  "affected_bookings": []
}
```

#### `POST /api/v1/events/booking-started`
**Description**: User picks up vehicle, actual battery known. **Triggers cascade re-prediction.**

**Request**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": 78,
  "actual_battery_at_start": 60.5,
  "actual_started_at": "2025-11-03T19:05:00Z",
  "future_bookings": [
    {
      "booking_id": "BK789",
      "user_id": 920,
      "starts_at": "2025-11-03T22:00:00Z",
      "ends_at": "2025-11-04T01:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "prediction_error": -2.2,
  "patterns_updated": true,
  "affected_bookings": [
    {
      "booking_id": "BK789",
      "predicted_battery": 47.2,
      "confidence_interval": {
        "lower": 37.2,
        "upper": 57.2
      },
      "starts_at": "2025-11-03T22:00:00Z",
      "prediction_method": "real_time_cascade"
    }
  ]
}
```

#### `POST /api/v1/events/booking-ended`
**Description**: User returns vehicle, complete outcome recorded. **Updates patterns and triggers cascade.**

**Request**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": 78,
  "user_id": 860,
  "starts_at": "2025-11-03T19:05:00Z",
  "ends_at": "2025-11-03T20:50:00Z",
  "battery_at_start": 60.5,
  "battery_at_end": 48.0,
  "mileage_at_start": 1000.0,
  "mileage_at_end": 1025.0,
  "charging_at_end": 0,
  "future_bookings": [
    {
      "booking_id": "BK789",
      "user_id": 920,
      "starts_at": "2025-11-03T22:00:00Z",
      "ends_at": "2025-11-04T01:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "outcome_recorded": true,
  "drain_rate_updated": true,
  "new_avg_drain_rate": 6.8,
  "pattern_confidence": "high",
  "total_bookings": 47,
  "queued_for_retraining": true,
  "affected_bookings": [
    {
      "booking_id": "BK789",
      "predicted_battery": 35.2,
      "confidence_interval": {
        "lower": 25.2,
        "upper": 45.2
      },
      "starts_at": "2025-11-03T22:00:00Z",
      "prediction_method": "real_time_cascade"
    }
  ]
}
```

#### `POST /api/v1/events/booking-modified`
**Description**: Booking time or duration changed. **Triggers cascade re-prediction.**

**Request**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": 78,
  "user_id": 860,
  "new_starts_at": "2025-11-03T18:00:00Z",
  "new_ends_at": "2025-11-03T21:00:00Z",
  "modification_type": "extended",
  "future_bookings": [
    {
      "booking_id": "BK789",
      "user_id": 920,
      "starts_at": "2025-11-03T22:00:00Z",
      "ends_at": "2025-11-04T01:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "prediction_updated": true,
  "affected_bookings": [
    {
      "booking_id": "BK123456",
      "predicted_battery": 55.0,
      "starts_at": "2025-11-03T18:00:00Z"
    },
    {
      "booking_id": "BK789",
      "predicted_battery": 42.0,
      "starts_at": "2025-11-03T22:00:00Z"
    }
  ]
}
```

#### `POST /api/v1/events/booking-cancelled`
**Description**: Booking cancelled before starting. **Triggers cascade re-prediction.**

**Request**:
```json
{
  "booking_id": "BK123456",
  "vehicle_id": 78,
  "future_bookings": [
    {
      "booking_id": "BK789",
      "user_id": 920,
      "starts_at": "2025-11-03T22:00:00Z",
      "ends_at": "2025-11-04T01:00:00Z"
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "prediction_cancelled": true,
  "affected_bookings": [
    {
      "booking_id": "BK789",
      "predicted_battery": 58.0,
      "starts_at": "2025-11-03T22:00:00Z"
    }
  ]
}
```

### Utility Endpoints

#### `GET /api/v1/vehicle/{vehicle_id}/timeline`
Get complete booking timeline with predictions

#### `GET /api/v1/model/info`
Get model information and statistics

#### `GET /health`
Health check endpoint

---

## Event-Driven Flow

### Flow 1: New Booking Created
```
1. User creates booking in Laravel
2. Laravel fires webhook → POST /events/booking-created
3. ML System:
   - Predicts battery at start using current state
   - Stores prediction in tracking DB
   - Returns prediction to Laravel
4. Laravel stores predicted_battery in bookings table
```

### Flow 2: User Picks Up Vehicle (Booking Started)
```
1. User scans QR code, vehicle reports actual battery
2. Laravel fires webhook → POST /events/booking-started
3. ML System:
   - Records actual battery vs predicted (calculates error)
   - Updates vehicle drain pattern if error > 5%
   - Finds all future bookings for this vehicle
   - Re-predicts them with updated battery state
   - Sends callback webhook to Laravel with updates
4. Laravel updates predictions for affected bookings
```

### Flow 3: User Returns Vehicle (Booking Ended)
```
1. User ends trip, vehicle reports final battery
2. Laravel fires webhook → POST /events/booking-ended
3. ML System:
   - Stores complete outcome (battery drain, mileage, etc.)
   - Updates vehicle drain rate immediately
   - Updates charging frequency if charged
   - Adds to retraining queue for weekly model update
   - Re-predicts future bookings for this vehicle
   - Sends callback webhook to Laravel
4. Laravel updates predictions
```

### Flow 4: Booking Extended/Modified
```
1. User extends booking or changes time
2. Laravel fires webhook → POST /events/booking-modified
3. ML System:
   - Finds bookings after this one (cascade effect)
   - Re-predicts affected bookings
   - Sends callback webhook
4. Laravel updates predictions
```

### Flow 5: Booking Cancelled
```
1. Booking cancelled before starting
2. Laravel fires webhook → POST /events/booking-cancelled
3. ML System:
   - Marks prediction as cancelled
   - Re-predicts bookings that were in cascade
   - Sends callback webhook
4. Laravel updates predictions
```

---

## Pattern Learning System

### How Pattern Learning Works

The system uses **Exponential Moving Average (EMA)** to learn from each booking immediately:

```python
new_value = α * observed_value + (1 - α) * current_average
```

Where α = 0.3 (30% weight to new data, 70% to historical average)

### Vehicle Patterns Updated After Each Booking

1. **Drain Rate per Hour** (`avg_drain_rate_per_hour`)
   - Calculated: `(battery_at_start - battery_at_end) / duration_hours`
   - Updated using EMA after each completed booking
   - Used to predict battery consumption during idle time

2. **Drain Rate per km** (`avg_drain_rate_per_km`)
   - Calculated: `battery_drain / distance_km`
   - Only updated when mileage data available
   - More accurate for vehicles with odometer telemetry

3. **Charging Frequency** (`charging_frequency`)
   - Tracks how often vehicle is charged during bookings
   - `charging_frequency = total_charges / total_bookings`
   - Helps predict likelihood of finding vehicle charged

4. **Standard Deviation** (`std_drain_rate_per_hour`)
   - Measures uncertainty in drain rate
   - Used to calculate confidence intervals
   - Increases when large prediction errors detected

### User Patterns Updated After Each Booking

1. **Average Return Battery** - How much battery users typically leave
2. **Charging Behavior** - % of bookings where user charges
3. **Average Booking Duration** - Typical trip length

### Error Correction

When `|actual - predicted| > 5%`:
1. System calculates correction factor
2. Adjusts drain rate conservatively (20% of error)
3. Increases uncertainty (std) by 10%
4. Re-predicts future bookings with updated patterns

### Confidence Levels

- **Low confidence**: < 5 bookings
- **Medium confidence**: 5-20 bookings
- **High confidence**: 20+ bookings

More bookings = tighter confidence intervals = more reliable predictions

---

## Cascade Re-prediction System

### What is Cascade?

When a vehicle's battery state changes, all future bookings for that vehicle need to be re-predicted. This "cascade effect" ensures predictions stay accurate as real-world events unfold.

### When Cascade Triggers

1. **Booking Started** - Actual battery differs from predicted
2. **Booking Ended** - New known battery state after trip
3. **Booking Modified** - Time changes affect subsequent bookings
4. **Booking Cancelled** - Battery won't drain during cancelled time

### How Cascade Works

```
Example: Vehicle has 3 future bookings

Booking A starts with actual battery 60% (predicted was 65%)
↓
System re-predicts:
  Booking B: 60% → drain during idle → predict 58% at start
  ↓
  Booking C: 58% → account for B's drain → predict 35% at start
  ↓
  Booking D: 35% → account for C's drain → predict 18% at start

All 3 updated predictions sent back to Laravel
```

### Cascade Algorithm

1. Sort future bookings chronologically
2. Start from current known battery state
3. For each booking:
   - Simulate idle drain before booking starts
   - Simulate booking drain (using patterns)
   - Predict battery at next booking start
4. Store all updated predictions
5. Return `affected_bookings` array to Laravel

### Laravel Integration

Laravel must send `future_bookings` array in webhook:
```json
{
  "future_bookings": [
    {
      "booking_id": "BK789",
      "user_id": 920,
      "starts_at": "2025-11-03T22:00:00Z",
      "ends_at": "2025-11-04T01:00:00Z"
    }
  ]
}
```

ML system returns updated predictions:
```json
{
  "affected_bookings": [
    {
      "booking_id": "BK789",
      "predicted_battery": 47.2,
      "confidence_interval": {...},
      "prediction_method": "real_time_cascade"
    }
  ]
}
```

---

## Database Schema

### Tracking Database (`data/tracking/predictions.db`)

#### Table: `predictions`
Stores all predictions made by the system

| Column | Type | Description |
|--------|------|-------------|
| prediction_id | TEXT PRIMARY KEY | Unique prediction identifier |
| booking_id | TEXT UNIQUE | Laravel booking ID |
| vehicle_id | INTEGER | Vehicle ID |
| user_id | INTEGER | User ID |
| predicted_at | TIMESTAMP | When prediction was made |
| booking_start_time | TIMESTAMP | Target prediction time |
| predicted_battery | REAL | Predicted battery % |
| confidence_lower | REAL | Lower confidence bound |
| confidence_upper | REAL | Upper confidence bound |
| actual_battery | REAL | Actual battery (filled when starts) |
| prediction_error | REAL | Actual - Predicted |
| model_version | TEXT | Model version used |
| prediction_method | TEXT | historical or real_time_cascade |
| status | TEXT | pending, completed, cancelled |

#### Table: `booking_outcomes`
Complete booking history for training

| Column | Type | Description |
|--------|------|-------------|
| outcome_id | INTEGER PRIMARY KEY | Auto-increment ID |
| booking_id | TEXT UNIQUE | Laravel booking ID |
| vehicle_id | INTEGER | Vehicle ID |
| starts_at | TIMESTAMP | Booking start |
| ends_at | TIMESTAMP | Booking end |
| battery_at_start | REAL | Battery at start |
| battery_at_end | REAL | Battery at end |
| battery_drain | REAL | Total drain |
| charging_at_end | INTEGER | 1 if charged, 0 otherwise |
| distance_km | REAL | Distance traveled |
| drain_rate_per_hour | REAL | Calculated drain rate |
| used_for_training | INTEGER | Whether used in model |

#### Table: `vehicle_patterns`
Per-vehicle learned patterns (updated immediately after each booking)

| Column | Type | Description |
|--------|------|-------------|
| vehicle_id | INTEGER PRIMARY KEY | Vehicle ID |
| avg_drain_rate_per_hour | REAL | Average % drain per hour |
| avg_charging_rate_per_hour | REAL | Average % charge per hour |
| charging_frequency | REAL | How often vehicle charges |
| total_bookings | INTEGER | Total bookings for this vehicle |
| updated_at | TIMESTAMP | Last pattern update |

---

## Laravel Integration Guide

### Step 1: Install Guzzle HTTP Client
```bash
composer require guzzlehttp/guzzle
```

### Step 2: Configure ML API URL
```php
// config/ml.php
return [
    'api_url' => env('ML_API_URL', 'http://localhost:8000'),
    'api_key' => env('ML_API_KEY', null),
    'callback_url' => env('APP_URL') . '/api/ml-callback',
];
```

### Step 3: Create Event Listeners

**app/Events/BookingCreated.php**
```php
<?php
namespace App\Events;

class BookingCreated
{
    public $booking;
    public $currentBattery;

    public function __construct($booking, $currentBattery = null)
    {
        $this->booking = $booking;
        $this->currentBattery = $currentBattery;
    }
}
```

**app/Listeners/SendBookingCreatedToML.php**
```php
<?php
namespace App\Listeners;

use App\Events\BookingCreated;
use Illuminate\Support\Facades\Http;

class SendBookingCreatedToML
{
    public function handle(BookingCreated $event)
    {
        $booking = $event->booking;
        $vehicle = $booking->vehicle;

        $response = Http::post(config('ml.api_url') . '/api/v1/events/booking-created', [
            'booking_id' => $booking->id,
            'vehicle_id' => $vehicle->id,
            'user_id' => $booking->user_id,
            'starts_at' => $booking->starts_at->toIso8601String(),
            'ends_at' => $booking->ends_at->toIso8601String(),
            'current_battery_level' => $event->currentBattery ?? $vehicle->current_battery_level,
        ]);

        if ($response->successful()) {
            $prediction = $response->json();
            $booking->update([
                'predicted_battery_at_start' => $prediction['predicted_battery']
            ]);
        }
    }
}
```

### Step 4: Register Events
```php
// app/Providers/EventServiceProvider.php
protected $listen = [
    BookingCreated::class => [SendBookingCreatedToML::class],
    BookingStarted::class => [SendBookingStartedToML::class],
    BookingEnded::class => [SendBookingEndedToML::class],
    BookingModified::class => [SendBookingModifiedToML::class],
    BookingCancelled::class => [SendBookingCancelledToML::class],
];
```

### Step 5: Fire Events in Controllers

**BookingController.php**
```php
public function store(Request $request)
{
    $booking = Booking::create($request->validated());

    // Fire event to ML system
    event(new BookingCreated($booking, $request->current_battery));

    return response()->json($booking);
}

public function start($id)
{
    $booking = Booking::findOrFail($id);
    $actualBattery = $booking->vehicle->current_battery_level;

    $booking->update(['started_at' => now()]);

    event(new BookingStarted($booking, $actualBattery));

    return response()->json($booking);
}

public function end($id)
{
    $booking = Booking::findOrFail($id);
    $vehicle = $booking->vehicle;

    $booking->update([
        'ended_at' => now(),
        'battery_at_end' => $vehicle->current_battery_level,
    ]);

    event(new BookingEnded($booking));

    return response()->json($booking);
}
```

### Step 6: Create Callback Route ✅

**Purpose**: Receive updated predictions from ML system when cascade re-predictions occur.

**routes/api.php**
```php
Route::post('/ml/predictions-updated', [MLCallbackController::class, 'handlePredictionsUpdated']);
```

**app/Http/Controllers/MLCallbackController.php**
```php
<?php
namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use App\Models\Booking;

class MLCallbackController extends Controller
{
    /**
     * Handle predictions updated callback from ML system
     *
     * This is called when:
     * - A booking starts and cascade re-prediction occurs
     * - A booking ends and future bookings are re-predicted
     * - A booking is modified or cancelled
     */
    public function handlePredictionsUpdated(Request $request)
    {
        // Verify HMAC signature
        if (!$this->verifySignature($request)) {
            Log::warning('Invalid ML callback signature');
            return response()->json(['error' => 'Invalid signature'], 403);
        }

        // Parse payload
        $event = $request->input('event'); // "predictions.updated"
        $triggeredBy = $request->input('triggered_by'); // "booking.started", "booking.ended", etc.
        $affectedBookings = $request->input('affected_bookings', []);
        $metadata = $request->input('metadata', []);

        Log::info("ML Callback: {$event} triggered by {$triggeredBy}", [
            'affected_count' => count($affectedBookings),
            'metadata' => $metadata
        ]);

        // Update predictions for affected bookings
        foreach ($affectedBookings as $update) {
            Booking::where('id', $update['booking_id'])
                ->update([
                    'predicted_battery_at_start' => $update['predicted_battery'],
                    'prediction_confidence_lower' => $update['confidence_interval']['lower'] ?? null,
                    'prediction_confidence_upper' => $update['confidence_interval']['upper'] ?? null,
                    'prediction_method' => $update['prediction_method'] ?? 'cascade',
                    'prediction_updated_at' => now()
                ]);
        }

        // Optionally: Send real-time notifications to users
        // foreach ($affectedBookings as $update) {
        //     $booking = Booking::find($update['booking_id']);
        //     event(new PredictionUpdated($booking, $update['predicted_battery']));
        // }

        return response()->json([
            'status' => 'success',
            'updated_count' => count($affectedBookings)
        ]);
    }

    /**
     * Verify HMAC signature from ML system
     */
    private function verifySignature(Request $request): bool
    {
        $signature = $request->header('X-ML-Signature');
        if (!$signature) {
            return false;
        }

        $secret = config('ml.webhook_secret');
        if (!$secret) {
            return true; // Skip verification if no secret configured
        }

        $payload = $request->getContent();
        $expectedSignature = hash_hmac('sha256', $payload, $secret);

        return hash_equals($expectedSignature, $signature);
    }
}
```

**config/ml.php** (Updated)
```php
return [
    'api_url' => env('ML_API_URL', 'http://localhost:8000'),
    'api_key' => env('ML_API_KEY', null),
    'callback_url' => env('APP_URL') . '/api/ml/predictions-updated',
    'webhook_secret' => env('ML_WEBHOOK_SECRET', 'your-secret-key-here'), // Must match ML config
];
```

### Step 7: Add Database Columns for Predictions

**Migration**
```php
Schema::table('bookings', function (Blueprint $table) {
    $table->float('predicted_battery_at_start')->nullable();
    $table->float('prediction_confidence_lower')->nullable();
    $table->float('prediction_confidence_upper')->nullable();
    $table->string('prediction_method')->nullable(); // 'historical', 'real_time', 'cascade'
    $table->timestamp('prediction_updated_at')->nullable();
});
```

### Step 8: Configure ML System

**In ML system's `config/config.yaml`**:
```yaml
webhooks:
  enabled: true
  callback_url: "https://your-laravel-app.com/api/ml/predictions-updated"
  secret_key: "your-secret-key-here"  # Same as ML_WEBHOOK_SECRET in Laravel
  timeout_seconds: 10
```

---

## Deployment

### Development Setup
```bash
# 1. Clone repository
git clone <repo>
cd "Machine learning"

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Run dashboard (optional)
python run_dashboard.py
```

### Production Deployment
```bash
# Use gunicorn for production
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Recommended for Production)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

---

## Performance Metrics

### Current Model Performance
- **MAE**: 4.37%
- **Within 5%**: 74.8%
- **Within 10%**: 86.1% ✅ (Target: 89%)
- **Within 15%**: 92.7%
- **R²**: 0.8333

### Prediction Latency
- Historical mode: ~100-200ms
- Real-time mode: ~150-300ms
- Pattern updates: <50ms
- Cascade re-prediction: ~100ms per affected booking

### Scalability
- Can handle 1000+ predictions/minute
- Database supports millions of records
- Async processing for callbacks

---

## Changelog

### Version 2.0 (2025-11-02) - Event-Driven System
- ✅ Added real-time prediction with current battery
- ✅ Built prediction tracking database
- 🚧 Building event webhook endpoints
- 🚧 Implementing pattern learning engine
- 📋 Planning cascade re-prediction system

### Version 1.0 (2025-11-01) - Initial Release
- ✅ Trained LightGBM model (86.1% accuracy)
- ✅ Enhanced feature engineering with charging patterns
- ✅ FastAPI REST API
- ✅ Gradio dashboard
- ✅ Historical prediction mode

---

## Contact & Support

For questions or issues:
1. Check this documentation first
2. Review API endpoint examples
3. Test with Gradio dashboard at http://localhost:7860
4. Check logs in `logs/app.log`

---

**Documentation Status**: 🟢 Up to date
**System Status**: 🟡 Core working, Event system in development
