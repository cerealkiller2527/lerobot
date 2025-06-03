#!/usr/bin/env python3

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for Xbox controller integration with LeRobot.
This script tests the Xbox controller configuration and robot wrapper without requiring actual hardware.
"""

import time
from dataclasses import asdict

from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig
from lerobot.common.robot_devices.control_configs import (
    XboxTeleoperateControlConfig,
    XboxRecordControlConfig,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.xbox_manipulator import XboxManipulatorRobot


def test_xbox_controller_config():
    """Test Xbox controller configuration."""
    print("Testing Xbox controller configuration...")
    
    config = XboxControllerConfig()
    print(f"Xbox controller config: {asdict(config)}")
    
    # Test custom configuration
    custom_config = XboxControllerConfig(
        device_index=0,
        dead_zone_sticks=0.15,
        dead_zone_triggers=0.08,
        position_limits={
            "shoulder_pan": (-150.0, 150.0),
            "shoulder_lift": (-90.0, 90.0),
        }
    )
    print(f"Custom Xbox controller config: {asdict(custom_config)}")
    print("✓ Xbox controller configuration test passed")


def test_xbox_teleoperate_config():
    """Test Xbox teleoperate configuration."""
    print("\nTesting Xbox teleoperate configuration...")
    
    config = XboxTeleoperateControlConfig()
    print(f"Xbox teleoperate config: {asdict(config)}")
    
    # Test custom configuration
    custom_config = XboxTeleoperateControlConfig(
        fps=30,
        teleop_time_s=60.0,
        target_arm="main",
        enable_xbox=True,
        display_data=True
    )
    print(f"Custom Xbox teleoperate config: {asdict(custom_config)}")
    print("✓ Xbox teleoperate configuration test passed")


def test_xbox_record_config():
    """Test Xbox record configuration."""
    print("\nTesting Xbox record configuration...")
    
    config = XboxRecordControlConfig(
        repo_id="test/xbox_controller_dataset",
        single_task="Xbox controller teleoperation test"
    )
    print(f"Xbox record config: {asdict(config)}")
    
    # Test with custom settings
    custom_config = XboxRecordControlConfig(
        repo_id="test/xbox_controller_custom",
        single_task="Custom Xbox teleoperation",
        fps=30,
        num_episodes=5,
        target_arm="main",
        enable_xbox=True
    )
    print(f"Custom Xbox record config: {asdict(custom_config)}")
    print("✓ Xbox record configuration test passed")


def test_xbox_manipulator_robot():
    """Test Xbox manipulator robot wrapper (mock mode)."""
    print("\nTesting Xbox manipulator robot wrapper...")
    
    # Create robot config with mock mode
    robot_config = So100RobotConfig(mock=True)
    xbox_config = XboxControllerConfig()
    
    try:
        # Create Xbox manipulator robot
        xbox_robot = XboxManipulatorRobot(
            config=robot_config,
            xbox_controller_config=xbox_config,
            target_arm="main"
        )
        
        print(f"Xbox robot created successfully")
        print(f"Target arm: {xbox_robot.target_arm}")
        print(f"Robot type: {xbox_robot.robot_type}")
        print(f"Is connected: {xbox_robot.is_connected}")
        print(f"Is Xbox connected: {xbox_robot.is_xbox_connected}")
        
        # Test connection (in mock mode)
        print("Testing connection...")
        xbox_robot.connect()
        print(f"After connect - Robot: {xbox_robot.is_connected}, Xbox: {xbox_robot.is_xbox_connected}")
        
        # Test disconnection
        print("Testing disconnection...")
        xbox_robot.disconnect()
        print(f"After disconnect - Robot: {xbox_robot.is_connected}, Xbox: {xbox_robot.is_xbox_connected}")
        
        print("✓ Xbox manipulator robot test passed")
        
    except Exception as e:
        print(f"✗ Xbox manipulator robot test failed: {e}")
        raise


def test_motor_name_mapping():
    """Test motor name mapping for Xbox controller."""
    print("\nTesting motor name mapping...")
    
    robot_config = So100RobotConfig()
    expected_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    actual_motors = list(robot_config.follower_arms["main"].motors.keys())
    
    print(f"Expected motors: {expected_motors}")
    print(f"Actual motors: {actual_motors}")
    
    if expected_motors == actual_motors:
        print("✓ Motor name mapping test passed")
    else:
        print("✗ Motor name mapping test failed - motor names don't match")
        print(f"Missing: {set(expected_motors) - set(actual_motors)}")
        print(f"Extra: {set(actual_motors) - set(expected_motors)}")


def main():
    """Run all Xbox controller integration tests."""
    print("=" * 60)
    print("Xbox Controller LeRobot Integration Tests")
    print("=" * 60)
    
    try:
        test_xbox_controller_config()
        test_xbox_teleoperate_config()
        test_xbox_record_config()
        test_motor_name_mapping()
        test_xbox_manipulator_robot()
        
        print("\n" + "=" * 60)
        print("✓ All Xbox controller integration tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main() 