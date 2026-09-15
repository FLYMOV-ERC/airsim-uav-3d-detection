"""ARCHIVED. Automation script running dataset collection across several AirSim environments.

Manages launching the different environment binaries and collecting varied
datasets. Documented in archive/docs/. Superseded by the campaign recorder
(dataset_generation/record_campaign.py), but preserved with its collector.
"""

import os
import sys
import json
import time
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import signal

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_port as _airsim_port


class EnvironmentManager:
    """Manages multiple AirSim environments and automates dataset collection"""

    # Common AirSim environment executables
    KNOWN_ENVIRONMENTS = {
        'Blocks': {
            'path': None,  # Will be set from args
            'description': 'Simple blocks environment',
            'recommended_frames': 500
        },
        'Neighborhood': {
            'path': None,
            'description': 'Suburban neighborhood environment',
            'recommended_frames': 800
        },
        'City': {
            'path': None,
            'description': 'Urban city environment',
            'recommended_frames': 1000
        },
        'Mountains': {
            'path': None,
            'description': 'Mountain landscape environment',
            'recommended_frames': 600
        },
        'Africa': {
            'path': None,
            'description': 'African savanna environment',
            'recommended_frames': 700
        },
        'AbandonedPark': {
            'path': None,
            'description': 'Abandoned park environment',
            'recommended_frames': 600
        }
    }

    def __init__(self, base_output_dir: str = 'dataset_multi_env',
                 airsim_port: int = None):
        """
        Initialize environment manager

        Args:
            base_output_dir: Base directory for all dataset outputs
            airsim_port: Port for AirSim connection
        """
        self.base_output_dir = Path(base_output_dir)
        self.airsim_port = airsim_port if airsim_port is not None else _airsim_port()
        self.current_env_process = None
        self.collection_process = None

    def find_environments(self, search_paths: List[str]) -> Dict[str, str]:
        """
        Search for AirSim environments in given paths

        Args:
            search_paths: List of directories to search

        Returns:
            Dictionary of found environment paths
        """
        found = {}

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue

            # Look for common AirSim executable patterns
            patterns = [
                '*.exe',  # Windows
                '*.sh',   # Linux launch scripts
                '*.app',  # macOS
                '*Environment*',  # Generic pattern
                '*Blocks*', '*City*', '*Neighborhood*', '*Mountains*'
            ]

            for pattern in patterns:
                for file in path.glob(f"**/{pattern}"):
                    if file.is_file() or (file.is_dir() and file.suffix == '.app'):
                        # Try to identify environment type from filename
                        filename = file.stem.lower()
                        for env_name in self.KNOWN_ENVIRONMENTS:
                            if env_name.lower() in filename:
                                found[env_name] = str(file)
                                print(f"Found {env_name} environment at: {file}")
                                break

        return found

    def launch_environment(self, env_path: str, env_name: str) -> bool:
        """
        Launch an AirSim environment

        Args:
            env_path: Path to environment executable
            env_name: Name of the environment

        Returns:
            Success status
        """
        if self.current_env_process:
            self.stop_current_environment()

        print(f"\nLaunching {env_name} environment...")
        print(f"Path: {env_path}")

        try:
            # Determine launch command based on file type
            path = Path(env_path)

            if path.suffix == '.sh':
                # Linux shell script
                cmd = ['bash', env_path]
            elif path.suffix == '.exe':
                # Windows executable
                cmd = [env_path]
            elif path.suffix == '.app':
                # macOS application
                cmd = ['open', env_path]
            else:
                # Try direct execution
                cmd = [env_path]

            # Add common AirSim arguments
            cmd.extend(['-windowed', '-ResX=1280', '-ResY=720'])

            # Launch environment
            self.current_env_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for environment to start
            print("Waiting for environment to initialize...")
            time.sleep(10)  # Give time for AirSim to start

            return True

        except Exception as e:
            print(f"Error launching environment: {e}")
            return False

    def stop_current_environment(self):
        """Stop the currently running environment"""
        if self.current_env_process:
            print("Stopping current environment...")
            try:
                self.current_env_process.terminate()
                time.sleep(2)
                if self.current_env_process.poll() is None:
                    self.current_env_process.kill()
            except:
                pass
            self.current_env_process = None

    def run_collection(self, env_name: str, output_dir: str,
                      frames: int, weather_variations: bool = True,
                      time_variations: bool = True) -> bool:
        """
        Run dataset collection for current environment

        Args:
            env_name: Name of the environment
            output_dir: Output directory for this environment's data
            frames: Number of frames to collect
            weather_variations: Enable weather changes
            time_variations: Enable time of day changes

        Returns:
            Success status
        """
        print(f"\nStarting data collection for {env_name}...")
        print(f"Output directory: {output_dir}")
        print(f"Frames to collect: {frames}")

        # Build collection command
        cmd = [
            sys.executable,  # Python interpreter
            'collect-dataset-multi.py',
            '--out', output_dir,
            '--frames', str(frames),
            '--port', str(self.airsim_port),
            '--intruders', 'Intruder1', 'Intruder2', 'Intruder3', 'Intruder4'
        ]

        if weather_variations:
            cmd.append('--vary_weather')
            cmd.extend(['--weather_interval', '150'])

        if time_variations:
            cmd.append('--vary_time')
            cmd.extend(['--time_interval', '200'])

        try:
            # Run collection script
            self.collection_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output
            for line in iter(self.collection_process.stdout.readline, ''):
                if line:
                    print(f"  {line.rstrip()}")

            # Wait for completion
            return_code = self.collection_process.wait()

            if return_code == 0:
                print(f"Collection completed successfully for {env_name}")
                return True
            else:
                print(f"Collection failed for {env_name} with code {return_code}")
                return False

        except KeyboardInterrupt:
            print("\n Collection interrupted by user")
            if self.collection_process:
                self.collection_process.terminate()
            return False

        except Exception as e:
            print(f"Error during collection: {e}")
            return False

    def create_combined_metadata(self, environments_collected: List[str]):
        """
        Create a combined metadata file for all collected environments

        Args:
            environments_collected: List of environment names that were collected
        """
        metadata = {
            'collection_info': {
                'timestamp': time.time(),
                'total_environments': len(environments_collected),
                'environments': environments_collected,
                'base_directory': str(self.base_output_dir)
            },
            'environment_details': {}
        }

        for env_name in environments_collected:
            env_dir = self.base_output_dir / env_name
            if env_dir.exists():
                # Count frames collected
                image_dir = env_dir / 'images' / 'front_center'
                if image_dir.exists():
                    num_frames = len(list(image_dir.glob('*.png')))
                else:
                    num_frames = 0

                metadata['environment_details'][env_name] = {
                    'frames_collected': num_frames,
                    'path': str(env_dir),
                    'has_weather_variations': True,
                    'has_time_variations': True
                }

        # Save metadata
        metadata_path = self.base_output_dir / 'collection_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved collection metadata to: {metadata_path}")

    def run_multi_environment_collection(self, environments: Dict[str, Dict],
                                       frames_per_env: Optional[int] = None):
        """
        Run collection across multiple environments

        Args:
            environments: Dictionary of environment configurations
            frames_per_env: Override frames per environment (None to use defaults)
        """
        print("\n" + "="*60)
        print("MULTI-ENVIRONMENT DATASET COLLECTION")
        print("="*60)

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy settings.json to user's Documents/AirSim directory
        self.setup_airsim_settings()

        collected_envs = []

        for env_name, env_config in environments.items():
            if not env_config.get('path'):
                print(f"\n Skipping {env_name}: No path specified")
                continue

            print(f"\n{'='*40}")
            print(f"Environment: {env_name}")
            print(f"{'='*40}")

            # Launch environment
            if not self.launch_environment(env_config['path'], env_name):
                print(f"Failed to launch {env_name}, skipping...")
                continue

            # Determine frames to collect
            if frames_per_env:
                frames = frames_per_env
            else:
                frames = env_config.get('recommended_frames', 500)

            # Create output directory for this environment
            output_dir = str(self.base_output_dir / env_name)

            # Run collection
            success = self.run_collection(
                env_name, output_dir, frames,
                weather_variations=True,
                time_variations=True
            )

            if success:
                collected_envs.append(env_name)

            # Stop environment
            self.stop_current_environment()

            # Brief pause between environments
            print(f"\nPausing before next environment...")
            time.sleep(5)

        # Create combined metadata
        if collected_envs:
            self.create_combined_metadata(collected_envs)

        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print(f"Successfully collected data from {len(collected_envs)} environments:")
        for env in collected_envs:
            print(f"  - {env}")
        print(f"Output directory: {self.base_output_dir.resolve()}")
        print("="*60)

    def setup_airsim_settings(self):
        """Copy settings.json to AirSim directory"""
        # Typical AirSim settings locations
        possible_paths = [
            Path.home() / 'Documents' / 'AirSim' / 'settings.json',
            Path.home() / 'AirSim' / 'settings.json',
            Path('/Users') / os.environ.get('USER', '') / 'Documents' / 'AirSim' / 'settings.json'
        ]

        source = Path('settings.json')
        if not source.exists():
            print("Warning: settings.json not found in current directory")
            return

        for dest_path in possible_paths:
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_path)
                print(f"Copied settings.json to: {dest_path}")
                return
            except:
                continue

        print("Warning: Could not copy settings.json to AirSim directory")


def main():
    parser = argparse.ArgumentParser(
        description="Automate multi-environment dataset collection for AirSim"
    )
    parser.add_argument(
        '--env_dir',
        default='./environments',
        help='Directory containing AirSim environments'
    )
    parser.add_argument(
        '--search_paths',
        nargs='+',
        help='Additional paths to search for environments'
    )
    parser.add_argument(
        '--output',
        default='dataset_multi_env',
        help='Base output directory for datasets'
    )
    parser.add_argument(
        '--frames',
        type=int,
        help='Override number of frames per environment'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='AirSim port'
    )
    parser.add_argument(
        '--environments',
        nargs='+',
        help='Specific environments to use (default: all found)'
    )

    args = parser.parse_args()

    # Create manager
    manager = EnvironmentManager(
        base_output_dir=args.output,
        airsim_port=args.port
    )

    # Search for environments
    search_paths = [args.env_dir]
    if args.search_paths:
        search_paths.extend(args.search_paths)

    print("Searching for AirSim environments...")
    found_envs = manager.find_environments(search_paths)

    if not found_envs:
        print("\n No AirSim environments found!")
        print("Please specify environment locations using --env_dir or --search_paths")
        print("\nYou can download AirSim environments from:")
        print("  https://github.com/microsoft/AirSim/releases")
        return

    # Filter environments if specified
    if args.environments:
        filtered = {}
        for env in args.environments:
            if env in found_envs:
                filtered[env] = {'path': found_envs[env]}
                if env in manager.KNOWN_ENVIRONMENTS:
                    filtered[env].update(manager.KNOWN_ENVIRONMENTS[env])
            else:
                print(f"Warning: Requested environment '{env}' not found")
        environments = filtered
    else:
        # Use all found environments
        environments = {}
        for env_name, env_path in found_envs.items():
            environments[env_name] = {'path': env_path}
            if env_name in manager.KNOWN_ENVIRONMENTS:
                environments[env_name].update(manager.KNOWN_ENVIRONMENTS[env_name])

    if not environments:
        print("No valid environments to process")
        return

    print(f"\nFound {len(environments)} environment(s) to process:")
    for env_name in environments:
        print(f"  - {env_name}")

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n Stopping collection...")
        manager.stop_current_environment()
        if manager.collection_process:
            manager.collection_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run collection
    try:
        manager.run_multi_environment_collection(
            environments,
            frames_per_env=args.frames
        )
    except Exception as e:
        print(f"\n Error during collection: {e}")
        manager.stop_current_environment()


if __name__ == "__main__":
    main()