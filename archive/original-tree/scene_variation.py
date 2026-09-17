"""
scene_variation.py
Module for adding environmental variety to AirSim scenes
Includes weather effects, time of day changes, and lighting variations
"""

import random
import math
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

try:
    import airsim
except Exception:
    import cosysairsim as airsim


class SceneVariation:
    """Manages scene variations including weather, time of day, and lighting"""

    # Weather parameter IDs in AirSim
    WEATHER_PARAMS = {
        'Rain': 0,
        'Roadwetness': 1,
        'Snow': 2,
        'RoadSnow': 3,
        'MapleLeaf': 4,
        'RoadLeaf': 5,
        'Dust': 6,
        'Fog': 7,
        'Enabled': 8
    }

    # Predefined weather presets
    WEATHER_PRESETS = {
        'clear': {
            'Rain': 0.0,
            'Snow': 0.0,
            'Fog': 0.0,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'light_fog': {
            'Rain': 0.0,
            'Snow': 0.0,
            'Fog': 0.3,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'heavy_fog': {
            'Rain': 0.0,
            'Snow': 0.0,
            'Fog': 0.7,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'light_rain': {
            'Rain': 0.3,
            'Roadwetness': 0.3,
            'Snow': 0.0,
            'Fog': 0.1,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'heavy_rain': {
            'Rain': 0.8,
            'Roadwetness': 0.8,
            'Snow': 0.0,
            'Fog': 0.2,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'light_snow': {
            'Rain': 0.0,
            'Snow': 0.3,
            'RoadSnow': 0.3,
            'Fog': 0.1,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'heavy_snow': {
            'Rain': 0.0,
            'Snow': 0.8,
            'RoadSnow': 0.8,
            'Fog': 0.3,
            'Dust': 0.0,
            'Enabled': 1.0
        },
        'dusty': {
            'Rain': 0.0,
            'Snow': 0.0,
            'Fog': 0.0,
            'Dust': 0.5,
            'Enabled': 1.0
        },
        'autumn': {
            'Rain': 0.1,
            'MapleLeaf': 0.5,
            'RoadLeaf': 0.5,
            'Fog': 0.05,
            'Dust': 0.0,
            'Enabled': 1.0
        }
    }

    # Time of day presets (hour, minute)
    TIME_PRESETS = {
        'dawn': (6, 0),
        'early_morning': (7, 30),
        'morning': (9, 0),
        'late_morning': (11, 0),
        'noon': (12, 0),
        'afternoon': (14, 30),
        'late_afternoon': (16, 30),
        'golden_hour': (17, 30),
        'sunset': (18, 30),
        'dusk': (19, 30),
        'night': (22, 0),
        'midnight': (0, 0),
        'late_night': (3, 0)
    }

    def __init__(self, client: airsim.MultirotorClient,
                 enable_weather: bool = True,
                 enable_time_changes: bool = True):
        """
        Initialize the SceneVariation manager

        Args:
            client: AirSim client connection
            enable_weather: Whether to enable weather variations
            enable_time_changes: Whether to enable time of day changes
        """
        self.client = client
        self.enable_weather = enable_weather
        self.enable_time_changes = enable_time_changes
        self.current_weather = 'clear'
        self.current_time = 'noon'
        self.weather_transition_progress = 0.0
        self.last_weather_change = time.time()

    def set_weather(self, weather_name: str) -> bool:
        """
        Set weather to a predefined preset

        Args:
            weather_name: Name of weather preset

        Returns:
            Success status
        """
        if not self.enable_weather:
            return False

        if weather_name not in self.WEATHER_PRESETS:
            print(f"Warning: Unknown weather preset '{weather_name}'")
            return False

        preset = self.WEATHER_PRESETS[weather_name]

        try:
            for param_name, value in preset.items():
                if param_name in self.WEATHER_PARAMS:
                    param_id = self.WEATHER_PARAMS[param_name]
                    self.client.simSetWeatherParameter(param_id, value)

            self.current_weather = weather_name
            return True

        except Exception as e:
            print(f"Error setting weather: {e}")
            return False

    def set_random_weather(self, exclude: Optional[List[str]] = None) -> str:
        """
        Set a random weather condition

        Args:
            exclude: List of weather names to exclude

        Returns:
            Name of selected weather
        """
        exclude = exclude or []
        available = [w for w in self.WEATHER_PRESETS.keys() if w not in exclude]

        if not available:
            available = list(self.WEATHER_PRESETS.keys())

        weather = random.choice(available)
        self.set_weather(weather)
        return weather

    def interpolate_weather(self, weather1: str, weather2: str, factor: float):
        """
        Smoothly interpolate between two weather states

        Args:
            weather1: Starting weather preset
            weather2: Target weather preset
            factor: Interpolation factor (0-1)
        """
        if not self.enable_weather:
            return

        if weather1 not in self.WEATHER_PRESETS or weather2 not in self.WEATHER_PRESETS:
            return

        preset1 = self.WEATHER_PRESETS[weather1]
        preset2 = self.WEATHER_PRESETS[weather2]

        try:
            # Interpolate common parameters
            common_params = set(preset1.keys()) & set(preset2.keys())

            for param_name in common_params:
                if param_name in self.WEATHER_PARAMS:
                    value1 = preset1.get(param_name, 0)
                    value2 = preset2.get(param_name, 0)
                    interpolated = value1 + (value2 - value1) * factor

                    param_id = self.WEATHER_PARAMS[param_name]
                    self.client.simSetWeatherParameter(param_id, interpolated)

        except Exception as e:
            print(f"Error interpolating weather: {e}")

    def set_time_of_day(self, time_name: str) -> bool:
        """
        Set time of day to a predefined preset

        Args:
            time_name: Name of time preset

        Returns:
            Success status
        """
        if not self.enable_time_changes:
            return False

        if time_name not in self.TIME_PRESETS:
            print(f"Warning: Unknown time preset '{time_name}'")
            return False

        hour, minute = self.TIME_PRESETS[time_name]

        try:
            # Create datetime for today with specified time
            now = datetime.now()
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Convert to string format expected by AirSim
            time_str = target_time.strftime("%Y-%m-%d %H:%M:%S")

            # Set time of day with reasonable parameters
            self.client.simSetTimeOfDay(
                is_enabled=True,
                start_datetime=time_str,
                is_start_datetime_dst=False,
                celestial_clock_speed=1.0,
                update_interval_secs=60.0,
                move_sun=True
            )

            self.current_time = time_name
            return True

        except Exception as e:
            print(f"Error setting time of day: {e}")
            return False

    def set_random_time(self, exclude: Optional[List[str]] = None) -> str:
        """
        Set a random time of day

        Args:
            exclude: List of time names to exclude

        Returns:
            Name of selected time
        """
        exclude = exclude or []
        available = [t for t in self.TIME_PRESETS.keys() if t not in exclude]

        if not available:
            available = list(self.TIME_PRESETS.keys())

        time_name = random.choice(available)
        self.set_time_of_day(time_name)
        return time_name

    def create_varied_conditions(self,
                                weather_probability: float = 0.3,
                                time_probability: float = 0.2) -> Dict[str, str]:
        """
        Randomly vary scene conditions based on probabilities

        Args:
            weather_probability: Probability of changing weather
            time_probability: Probability of changing time

        Returns:
            Dictionary with current conditions
        """
        conditions = {
            'weather': self.current_weather,
            'time': self.current_time,
            'changed': False
        }

        # Possibly change weather
        if random.random() < weather_probability:
            conditions['weather'] = self.set_random_weather([self.current_weather])
            conditions['changed'] = True

        # Possibly change time
        if random.random() < time_probability:
            conditions['time'] = self.set_random_time([self.current_time])
            conditions['changed'] = True

        return conditions

    def cycle_time_of_day(self, speed_multiplier: float = 1.0):
        """
        Continuously cycle through time of day

        Args:
            speed_multiplier: How fast to cycle (1.0 = real-time)
        """
        if not self.enable_time_changes:
            return

        try:
            self.client.simSetTimeOfDay(
                is_enabled=True,
                start_datetime="",  # Use current time
                is_start_datetime_dst=False,
                celestial_clock_speed=speed_multiplier * 60.0,  # 60x for visible change
                update_interval_secs=1.0,
                move_sun=True
            )
        except Exception as e:
            print(f"Error cycling time of day: {e}")

    def get_current_conditions(self) -> Dict[str, any]:
        """
        Get current scene conditions

        Returns:
            Dictionary with current weather and time settings
        """
        return {
            'weather': self.current_weather,
            'time': self.current_time,
            'weather_enabled': self.enable_weather,
            'time_changes_enabled': self.enable_time_changes
        }

    def reset_to_default(self):
        """Reset scene to default conditions"""
        self.set_weather('clear')
        self.set_time_of_day('noon')

    def create_dataset_variety_sequence(self, num_frames: int,
                                       change_interval: int = 100) -> List[Dict]:
        """
        Create a sequence of varied conditions for dataset collection

        Args:
            num_frames: Total number of frames to plan for
            change_interval: Frames between condition changes

        Returns:
            List of condition dictionaries with frame numbers
        """
        sequence = []

        # Define a progression through different conditions
        weather_sequence = ['clear', 'light_fog', 'clear', 'light_rain',
                          'clear', 'dusty', 'clear', 'heavy_fog',
                          'clear', 'light_snow', 'clear']

        time_sequence = ['dawn', 'morning', 'noon', 'afternoon',
                        'golden_hour', 'sunset', 'night', 'midnight']

        weather_idx = 0
        time_idx = 0

        for frame in range(0, num_frames, change_interval):
            condition = {
                'frame': frame,
                'weather': weather_sequence[weather_idx % len(weather_sequence)],
                'time': time_sequence[time_idx % len(time_sequence)]
            }

            sequence.append(condition)

            # Advance indices with different rates for variety
            if frame % (change_interval * 2) == 0:
                weather_idx += 1
            if frame % (change_interval * 3) == 0:
                time_idx += 1

        return sequence


def test_scene_variation():
    """Test function to demonstrate scene variation capabilities"""

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Create scene variation manager
    scene_var = SceneVariation(client, enable_weather=True, enable_time_changes=True)

    print("Testing Scene Variations...")
    print("-" * 50)

    # Test weather presets
    print("\nTesting Weather Presets:")
    for weather in ['clear', 'light_fog', 'heavy_rain', 'light_snow']:
        print(f"  Setting weather to: {weather}")
        scene_var.set_weather(weather)
        time.sleep(2)

    # Test time presets
    print("\nTesting Time of Day Presets:")
    for time_name in ['dawn', 'noon', 'sunset', 'night']:
        print(f"  Setting time to: {time_name}")
        scene_var.set_time_of_day(time_name)
        time.sleep(2)

    # Test random variations
    print("\nTesting Random Variations:")
    for i in range(5):
        conditions = scene_var.create_varied_conditions(
            weather_probability=0.7,
            time_probability=0.5
        )
        print(f"  Iteration {i+1}: {conditions}")
        time.sleep(2)

    # Reset to default
    print("\nResetting to default conditions...")
    scene_var.reset_to_default()

    print("\nScene variation test complete!")


if __name__ == "__main__":
    test_scene_variation()