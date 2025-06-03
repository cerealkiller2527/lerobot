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

import time
from typing import Dict, Tuple

import numpy as np
import torch
import logging

from lerobot.common.robot_devices.controllers.xbox_controller import XboxController
from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot, ensure_safe_goal_position
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError


class XboxManipulatorRobot(ManipulatorRobot):
    """
    Xbox controller-based manipulator robot that replaces leader arms with Xbox controller input.
    Inherits from ManipulatorRobot and overrides teleop_step to use Xbox controller.
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig,
        xbox_controller_config: XboxControllerConfig,
        target_arm: str = "main",
    ):
        """
        Initialize Xbox manipulator robot.
        
        Args:
            config: ManipulatorRobotConfig for the robot
            xbox_controller_config: XboxControllerConfig for Xbox controller
            target_arm: Which follower arm to control (e.g., "main", "left", "right")
        """
        super().__init__(config)
        
        self.xbox_controller_config = xbox_controller_config
        self.xbox_controller = XboxController(xbox_controller_config)
        self.target_arm = target_arm
        self.is_xbox_connected = False
        
        # Store the last known follower position for inverse kinematics
        self.last_follower_pos = {}

    def connect(self):
        """Connect to robot without requiring leader arms."""
        logging.info("Connecting Xbox-controlled robot (no leader arms required)")
        
        # Only connect follower arms, skip leader arms entirely
        for name, robot_arm in self.follower_arms.items():
            logging.info(f"Connecting {name} follower arm.")
            robot_arm.connect()
        
        # Connect cameras
        for name, camera in self.cameras.items():
            logging.info(f"Connecting {name} camera.")
            camera.connect()
        
        # Initialize Xbox controller
        self.xbox_controller.connect()
        logging.info("Xbox controller connected")
        
        self.is_connected = True

    def disconnect(self):
        """Disconnect robot and Xbox controller."""
        if not self.is_connected:
            return
        
        # Disconnect Xbox controller
        if self.xbox_controller:
            self.xbox_controller.disconnect()
        
        # Only disconnect follower arms
        for name, robot_arm in self.follower_arms.items():
            logging.info(f"Disconnecting {name} follower arm.")
            robot_arm.disconnect()
        
        # Disconnect cameras
        for name, camera in self.cameras.items():
            logging.info(f"Disconnecting {name} camera.")
            camera.disconnect()
        
        self.is_connected = False

    def teleop_step(
        self, record_data: bool = False
    ) -> dict[str, torch.Tensor] | None:
        """
        Execute one teleoperation step using Xbox controller input.
        This replaces reading from leader arms with Xbox controller input.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Robot is not connected. You need to run `robot.connect()`."
            )

        # Update Xbox controller state
        self.xbox_controller.update()
        
        # Get Xbox controller positions for the target arm
        xbox_positions = self.xbox_controller.get_motor_positions()
        
        # Convert to torch tensor format expected by LeRobot
        leader_pos = {}
        target_arm_key = f"{self.target_arm}_arm_pos"
        
        # Map Xbox controller positions to expected format
        motor_positions = []
        for motor_name in self.xbox_controller.motor_names:
            if motor_name in xbox_positions:
                motor_positions.append(xbox_positions[motor_name])
            else:
                # Fallback to current position if motor not controlled by Xbox
                current_pos = self.follower_arms[self.target_arm].read("Present_Position")
                motor_idx = list(self.follower_arms[self.target_arm].motors.keys()).index(motor_name)
                motor_positions.append(current_pos[motor_idx])
        
        leader_pos[target_arm_key] = torch.tensor(motor_positions, dtype=torch.float32)
        
        # Write commands to follower arms  
        for name, robot_arm in self.follower_arms.items():
            if name == self.target_arm:
                arm_key = f"{name}_arm_pos"
                if arm_key in leader_pos:
                    robot_arm.write("Goal_Position", leader_pos[arm_key])

        # Capture images if needed
        images = {}
        if record_data:
            for name, camera in self.cameras.items():
                image = camera.read()
                images[name] = image

        # Return data in expected format
        if record_data:
            data_dict = {}
            data_dict.update(leader_pos)
            data_dict.update(images)
            return data_dict
        
        return None

    def __del__(self):
        """Cleanup on deletion."""
        if getattr(self, "is_connected", False) or getattr(self, "is_xbox_connected", False):
            self.disconnect() 