"""Configuration classes for robot input controllers."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import draccus


@dataclass
class XboxControllerConfig(draccus.ChoiceRegistry):
    """Configuration for Xbox controller input."""
    
    # Controller settings
    device_index: int = 0  # Xbox controller device index
    dead_zone_sticks: float = 0.1  # Deadzone for analog sticks (simplified)
    dead_zone_triggers: float = 0.1  # Deadzone for triggers (simplified)
    
    # Robot arm settings - Individual speeds for each axis
    shoulder_pan_speed: float = 0.025  # Base rotation speed
    x_axis_speed: float = 0.025  # Forward/backward movement speed
    y_axis_speed: float = 0.025  # Up/down movement speed
    wrist_flex_speed: float = 0.025  # Wrist up/down speed (was 0.25)
    wrist_roll_speed: float = 0.025  # Wrist rotation speed (was 0.25)
    gripper_speed: float = 0.03  # Gripper open/close speed
    
    # Robot arm settings
    l1: float = 117.0  # Link 1 length in mm
    l2: float = 136.0  # Link 2 length in mm
    
    # Motor configuration
    motor_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", 
        "wrist_flex", "wrist_roll", "gripper"
    ])
    
    # Initial robot position (safe starting pose)
    initial_position: List[float] = field(default_factory=lambda: [0, 170, 170, 0, 0, 10])
    
    # Macro positions for preset poses (activated by face buttons)
    macros: Dict[str, List[float]] = field(default_factory=lambda: {
        "A": [0, 170, 170, 0, 0, 10],     # Home position
        "B": [0, 160, 140, 20, 0, 0],     # Reach forward
        "X": [0, 50, 130, -90, 90, 80],   # Pick position
        "Y": [0, 130, 150, 70, 90, 80],   # Place position
    })
    
    # Safety limits for each motor and end-effector
    position_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "shoulder_pan": (-80, 80),
        "shoulder_lift": (-5, 185),
        "elbow_flex": (-5, 185),
        "wrist_flex": (-110, 110),
        "wrist_roll": (-110, 110),
        "gripper": (0, 100),
        # End-effector workspace limits (mm)
        "x": (15, 250),
        "y": (-110, 250),
    })
    
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__) 