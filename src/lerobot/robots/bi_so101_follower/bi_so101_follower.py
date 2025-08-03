#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.camera_manager import CameraManager
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from ..robot import Robot
from .config_bi_so101_follower import BiSO101FollowerConfig

logger = logging.getLogger(__name__)


class BiSO101Follower(Robot):
    """
    [Bimanual SO-101 Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    This bimanual robot uses two SO-101 follower arms for dual-arm manipulation tasks.
    """

    config_class = BiSO101FollowerConfig
    name = "bi_so101_follower"

    def __init__(self, config: BiSO101FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=self.calibration_dir,  # Use parent's calibration_dir
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        right_arm_config = SO101FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=self.calibration_dir,  # Use parent's calibration_dir
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Ultimate simplification - all camera complexity handled by manager
        self.camera_manager = CameraManager(self.cameras, config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return self.camera_manager.get_features()

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and self.camera_manager.is_all_connected
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        self.camera_manager.connect_all()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Sequential motor reading - optimized
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()

        # Add prefixes - optimized string processing
        for key, value in left_obs.items():
            obs_dict[f"left_{key}"] = value
        for key, value in right_obs.items():
            obs_dict[f"right_{key}"] = value

        # Ultimate simplification - all camera complexity in 1 line (generous timeout for depth cameras)
        obs_dict.update(self.camera_manager.read_all(timeout_ms=200))

        return obs_dict
    


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Optimized action parsing - avoid multiple dictionary iterations
        left_action = {}
        right_action = {}
        
        for key, value in action.items():
            if key.startswith("left_"):
                left_action[key[5:]] = value  # Remove "left_" prefix (5 chars)
            elif key.startswith("right_"):
                right_action[key[6:]] = value  # Remove "right_" prefix (6 chars)

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back - optimized
        result = {}
        for key, value in send_action_left.items():
            result[f"left_{key}"] = value
        for key, value in send_action_right.items():
            result[f"right_{key}"] = value

        return result

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        self.camera_manager.disconnect_all()