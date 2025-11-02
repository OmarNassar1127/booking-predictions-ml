"""
Generate synthetic booking data for development and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import random


class BookingDataGenerator:
    """Generate realistic synthetic electric car booking data"""

    def __init__(
        self,
        n_bookings: int = 40000,
        n_vehicles: int = 50,
        n_users: int = 500,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        random_seed: int = 42
    ):
        """
        Initialize data generator

        Args:
            n_bookings: Number of bookings to generate
            n_vehicles: Number of unique vehicles
            n_users: Number of unique users
            start_date: Start date for bookings
            end_date: End date for bookings
            random_seed: Random seed for reproducibility
        """
        self.n_bookings = n_bookings
        self.n_vehicles = n_vehicles
        self.n_users = n_users
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.random_seed = random_seed

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Vehicle characteristics (affects battery drain and charging)
        self.vehicle_profiles = self._create_vehicle_profiles()

        # User behavior profiles (affects usage patterns)
        self.user_profiles = self._create_user_profiles()

    def _create_vehicle_profiles(self):
        """Create vehicle characteristics"""
        profiles = {}
        for vehicle_id in range(1, self.n_vehicles + 1):
            profiles[f"V{vehicle_id:03d}"] = {
                'battery_capacity_kwh': np.random.choice([60, 75, 90, 100]),
                'efficiency_km_per_kwh': np.random.uniform(5, 7),  # km per kWh
                'charging_speed_kwh_per_hour': np.random.choice([7, 11, 22, 50]),  # kW charging
                'base_drain_rate': np.random.uniform(0.15, 0.25),  # % per hour of use
            }
        return profiles

    def _create_user_profiles(self):
        """Create user behavior profiles"""
        profiles = {}
        behavior_types = ['conservative', 'moderate', 'heavy_user', 'short_trip']

        for user_id in range(1, self.n_users + 1):
            behavior = np.random.choice(behavior_types, p=[0.3, 0.4, 0.2, 0.1])

            if behavior == 'conservative':
                avg_duration_hours = np.random.uniform(0.5, 2)
                avg_distance_km = np.random.uniform(5, 30)
                return_battery_preference = np.random.uniform(60, 90)
            elif behavior == 'moderate':
                avg_duration_hours = np.random.uniform(2, 6)
                avg_distance_km = np.random.uniform(30, 100)
                return_battery_preference = np.random.uniform(40, 70)
            elif behavior == 'heavy_user':
                avg_duration_hours = np.random.uniform(4, 12)
                avg_distance_km = np.random.uniform(80, 250)
                return_battery_preference = np.random.uniform(20, 50)
            else:  # short_trip
                avg_duration_hours = np.random.uniform(0.25, 1)
                avg_distance_km = np.random.uniform(3, 15)
                return_battery_preference = np.random.uniform(70, 95)

            profiles[f"U{user_id:04d}"] = {
                'behavior_type': behavior,
                'avg_duration_hours': avg_duration_hours,
                'avg_distance_km': avg_distance_km,
                'return_battery_preference': return_battery_preference,
                'booking_frequency_days': np.random.uniform(3, 21),
            }

        return profiles

    def _simulate_battery_drain(self, vehicle_id: str, user_id: str, duration_hours: float, distance_km: float) -> float:
        """Simulate realistic battery drain"""
        vehicle = self.vehicle_profiles[vehicle_id]

        # Base drain from driving
        energy_used_kwh = distance_km / vehicle['efficiency_km_per_kwh']
        battery_drain_percent = (energy_used_kwh / vehicle['battery_capacity_kwh']) * 100

        # Additional drain from time (HVAC, etc.)
        time_drain = vehicle['base_drain_rate'] * duration_hours

        # Add some randomness (traffic, weather, driving style)
        noise = np.random.normal(0, 5)

        total_drain = battery_drain_percent + time_drain + noise

        # Ensure drain is reasonable (min 2%, max 95%)
        return np.clip(total_drain, 2, 95)

    def _simulate_charging(self, vehicle_id: str, time_gap_hours: float, battery_at_end: float) -> float:
        """Simulate charging between bookings"""
        vehicle = self.vehicle_profiles[vehicle_id]

        # Probability of charging depends on battery level and time gap
        charging_probability = 1 - (battery_at_end / 100)  # Lower battery = higher chance

        if time_gap_hours < 1:
            # Very short gap - unlikely to charge
            charging_probability *= 0.1
        elif time_gap_hours > 8:
            # Overnight - very likely to charge
            charging_probability = min(charging_probability * 1.5, 0.95)

        if np.random.random() < charging_probability:
            # Calculate how much can be charged
            max_charge_kwh = vehicle['charging_speed_kwh_per_hour'] * time_gap_hours
            max_charge_percent = (max_charge_kwh / vehicle['battery_capacity_kwh']) * 100

            # Target: charge to 80-100% (most charging stations limit to 80%)
            target_battery = np.random.uniform(80, 100)
            charge_needed = target_battery - battery_at_end

            actual_charge = min(charge_needed, max_charge_percent)

            # Add some randomness (user might unplug early, etc.)
            actual_charge *= np.random.uniform(0.7, 1.0)

            return battery_at_end + actual_charge
        else:
            # No charging, just natural drain (vampire drain)
            vampire_drain = time_gap_hours * 0.01  # ~0.01% per hour
            return max(battery_at_end - vampire_drain, battery_at_end)

    def generate(self) -> pd.DataFrame:
        """Generate complete booking dataset"""
        print(f"Generating {self.n_bookings} bookings...")

        bookings = []
        vehicle_last_booking = {}  # Track last booking per vehicle

        # Generate bookings
        total_days = (self.end_date - self.start_date).days

        for i in range(self.n_bookings):
            # Random vehicle and user
            vehicle_id = f"V{np.random.randint(1, self.n_vehicles + 1):03d}"
            user_id = f"U{np.random.randint(1, self.n_users + 1):04d}"

            user_profile = self.user_profiles[user_id]

            # Generate booking start time
            random_day = np.random.randint(0, total_days)
            hour_probabilities = np.array([
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am
                0.04, 0.06, 0.08, 0.07, 0.06, 0.05,  # 6-11am
                0.05, 0.04, 0.05, 0.06, 0.08, 0.09,  # 12-5pm
                0.08, 0.06, 0.04, 0.03, 0.02, 0.01   # 6-11pm
            ])
            hour_probabilities = hour_probabilities / hour_probabilities.sum()  # Normalize to sum to 1
            random_hour = int(np.random.choice(range(24), p=hour_probabilities))

            starts_at = self.start_date + timedelta(days=int(random_day), hours=random_hour)
            starts_at += timedelta(minutes=np.random.randint(0, 60))

            # Generate booking duration and distance based on user profile
            duration_hours = max(0.25, np.random.normal(
                user_profile['avg_duration_hours'],
                user_profile['avg_duration_hours'] * 0.3
            ))

            distance_km = max(1, np.random.normal(
                user_profile['avg_distance_km'],
                user_profile['avg_distance_km'] * 0.4
            ))

            ends_at = starts_at + timedelta(hours=duration_hours)

            # Calculate battery levels
            if vehicle_id in vehicle_last_booking:
                last_booking = vehicle_last_booking[vehicle_id]
                time_gap = (starts_at - last_booking['ends_at']).total_seconds() / 3600

                if time_gap > 0:
                    # Simulate charging between bookings
                    battery_at_start = self._simulate_charging(
                        vehicle_id,
                        time_gap,
                        last_booking['battery_at_end']
                    )
                else:
                    # Overlapping bookings shouldn't happen, but handle edge case
                    battery_at_start = 80
            else:
                # First booking for this vehicle - start with random good battery
                battery_at_start = np.random.uniform(70, 100)

            # Ensure battery is capped at 100%
            battery_at_start = min(battery_at_start, 100)

            # Calculate battery drain
            drain = self._simulate_battery_drain(vehicle_id, user_id, duration_hours, distance_km)
            battery_at_end = max(5, battery_at_start - drain)  # Minimum 5% battery

            # Calculate mileage
            if vehicle_id in vehicle_last_booking:
                mileage_at_start = last_booking['mileage_at_end']
            else:
                mileage_at_start = np.random.uniform(5000, 50000)  # Random starting mileage

            mileage_at_end = mileage_at_start + distance_km

            # Create booking record
            booking = {
                'booking_id': i + 1,
                'vehicle_id': vehicle_id,
                'user_id': user_id,
                'starts_at': starts_at,
                'ends_at': ends_at,
                'battery_at_start': round(battery_at_start, 2),
                'battery_at_end': round(battery_at_end, 2),
                'mileage_at_start': round(mileage_at_start, 2),
                'mileage_at_end': round(mileage_at_end, 2),
            }

            bookings.append(booking)

            # Update last booking for this vehicle
            vehicle_last_booking[vehicle_id] = {
                'ends_at': ends_at,
                'battery_at_end': battery_at_end,
                'mileage_at_end': mileage_at_end
            }

            if (i + 1) % 5000 == 0:
                print(f"Generated {i + 1}/{self.n_bookings} bookings...")

        # Create DataFrame and sort by time
        df = pd.DataFrame(bookings)
        df = df.sort_values('starts_at').reset_index(drop=True)

        print(f"✓ Generated {len(df)} bookings")
        print(f"  Date range: {df['starts_at'].min()} to {df['starts_at'].max()}")
        print(f"  Vehicles: {df['vehicle_id'].nunique()}")
        print(f"  Users: {df['user_id'].nunique()}")
        print(f"  Avg battery at start: {df['battery_at_start'].mean():.1f}%")
        print(f"  Avg battery at end: {df['battery_at_end'].mean():.1f}%")

        return df

    def save_to_csv(self, output_path: str):
        """Generate and save data to CSV"""
        df = self.generate()
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return df


if __name__ == "__main__":
    from pathlib import Path

    # Generate sample data
    generator = BookingDataGenerator(
        n_bookings=40000,
        n_vehicles=50,
        n_users=500,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "raw" / "bookings.csv"

    df = generator.save_to_csv(str(output_path))

    # Show sample
    print("\nSample data:")
    print(df.head(10))
