"""Xbox controller configuration for robot arm teleoperation."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class XboxControllerConfig:
    """Xbox controller configuration for robot arm teleoperation."""
    
    # Controller settings
    device_index: int = 0
    dead_zone_sticks: float = 0.1
    dead_zone_triggers: float = 0.1
    
    # Movement speeds per axis
    shoulder_pan_speed: float = 0.25
    x_axis_speed: float = 0.25
    y_axis_speed: float = 0.25
    wrist_flex_speed: float = 0.25
    wrist_roll_speed: float = 0.25
    gripper_speed: float = 0.3
    
    # 2-link arm kinematics (mm)
    l1: float = 117.0  # First link length
    l2: float = 136.0  # Second link length
    
    # Motor configuration
    motor_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", 
        "wrist_flex", "wrist_roll", "gripper"
    ])
    
    # Initial safe position
    initial_position: List[float] = field(default_factory=lambda: [0, 170, 170, 0, 0, 10])
    
    # Preset poses (face button macros)
    macros: Dict[str, List[float]] = field(default_factory=lambda: {
        "A": [0, 170, 170, 0, 0, 10],     # Home
        "B": [0, 160, 140, 20, 0, 0],     # Reach forward
        "X": [0, 50, 130, -90, 90, 80],   # Pick
        "Y": [0, 130, 150, 70, 90, 80],   # Place
    })
    
    # Safety limits (degrees for motors, mm for workspace)
    position_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "shoulder_pan": (-70, 70),
        "shoulder_lift": (-5, 185),
        "elbow_flex": (-5, 165),
        "wrist_flex": (-110, 110),
        "wrist_roll": (-110, 110),
        "gripper": (0, 100),
        "x": (15, 250),
        "y": (-110, 250),
    }) 