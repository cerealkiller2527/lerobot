"""Xbox controller teleoperation for robot arms.

Controller mapping:
- Left stick: Base rotation (X) + Forward/backward (Y)  
- Right stick: Wrist roll (X) + Wrist flex (Y)
- Triggers: Gripper control (LT=open, RT=close)
- D-pad: Up/down movement + additional base rotation
- Face buttons: Preset positions (A=home, B=reach, X=pick, Y=place)
"""

import copy
import logging
import math
from typing import Dict

import pygame

from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig


class XboxController:
    """Xbox controller for robot arm teleoperation with 2-link kinematics."""
    
    def __init__(self, config: XboxControllerConfig):
        """Initialize Xbox controller."""
        self.config = config
        
        # Motor state
        self.motor_names = config.motor_names
        self.current_positions = dict(zip(self.motor_names, config.initial_position, strict=False))
        self.new_positions = self.current_positions.copy()
        
        # Kinematics
        self.l1 = config.l1
        self.l2 = config.l2
        self.x, self.y = self._compute_position(
            self.current_positions["shoulder_lift"], 
            self.current_positions["elbow_flex"]
        )
        
        # Controller state
        self.axes = {
            "LX": 0.0, "LY": 0.0, "RX": 0.0, "RY": 0.0, "LT": 0.0, "RT": 0.0,
        }
        
        self.buttons = {
            "A": 0, "B": 0, "X": 0, "Y": 0, "LB": 0, "RB": 0,
            "BACK": 0, "START": 0, "LS": 0, "RS": 0,
            "DPAD_UP": 0, "DPAD_DOWN": 0, "DPAD_LEFT": 0, "DPAD_RIGHT": 0,
        }
        
        self.previous_buttons = self.buttons.copy()
        
        # Connection
        self.joystick = None
        self.device_index = config.device_index
        self.connected = False
        
        self._init_pygame()
        self.connect()
    
    def _init_pygame(self):
        """Initialize pygame joystick subsystem."""
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            logging.error(f"Failed to initialize pygame: {e}")
    
    def connect(self) -> bool:
        """Connect to Xbox controller."""
        try:
            joystick_count = pygame.joystick.get_count()
            if joystick_count == 0:
                logging.error("No joysticks found")
                return False
                
            if self.device_index >= joystick_count:
                logging.error(f"Device index {self.device_index} not available. Found {joystick_count} device(s)")
                return False
                
            self.joystick = pygame.joystick.Joystick(self.device_index)
            self.joystick.init()
            
            logging.info(f"Connected to controller: {self.joystick.get_name()}")
            self.connected = True
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to Xbox controller: {e}")
            self.joystick = None
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Xbox controller."""
        self.connected = False
        if self.joystick:
            try:
                self.joystick.quit()
            except Exception as e:
                logging.error(f"Error disconnecting controller: {e}")
            finally:
                self.joystick = None
    
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        return self.joystick is not None and self.connected
    
    def update(self):
        """Update controller state and process input."""
        if not self.is_connected():
            return False
            
        try:
            pygame.event.pump()
            self._read_controller_state()
            return True
            
        except Exception as e:
            logging.error(f"Error updating controller: {e}")
            self.connected = False
            return False
    
    def _read_controller_state(self):
        """Read current controller state and update positions."""
        if not self.joystick:
            return
            
        # Read analog sticks
        if self.joystick.get_numaxes() >= 4:
            self.axes["LX"] = self._filter_deadzone(self.joystick.get_axis(0), self.config.dead_zone_sticks)
            self.axes["LY"] = self._filter_deadzone(-self.joystick.get_axis(1), self.config.dead_zone_sticks)  # Invert Y
            self.axes["RX"] = self._filter_deadzone(self.joystick.get_axis(2), self.config.dead_zone_sticks)
            self.axes["RY"] = self._filter_deadzone(-self.joystick.get_axis(3), self.config.dead_zone_sticks)  # Invert Y
        
        # Read triggers (if available)
        if self.joystick.get_numaxes() >= 6:
            # Triggers are typically on axes 4 and 5, range -1 to 1, convert to 0 to 1
            lt_raw = (self.joystick.get_axis(4) + 1.0) / 2.0  # Convert -1,1 to 0,1
            rt_raw = (self.joystick.get_axis(5) + 1.0) / 2.0
            self.axes["LT"] = self._filter_deadzone(lt_raw, self.config.dead_zone_triggers)
            self.axes["RT"] = self._filter_deadzone(rt_raw, self.config.dead_zone_triggers)
        
        # Read buttons
        num_buttons = self.joystick.get_numbuttons()
        if num_buttons >= 10:  # Standard Xbox controller has at least 10 buttons
            self.buttons["A"] = self.joystick.get_button(0)
            self.buttons["B"] = self.joystick.get_button(1) 
            self.buttons["X"] = self.joystick.get_button(2)
            self.buttons["Y"] = self.joystick.get_button(3)
            self.buttons["LB"] = self.joystick.get_button(4)
            self.buttons["RB"] = self.joystick.get_button(5)
            self.buttons["BACK"] = self.joystick.get_button(6)
            self.buttons["START"] = self.joystick.get_button(7)
            self.buttons["LS"] = self.joystick.get_button(8)
            self.buttons["RS"] = self.joystick.get_button(9)
        
        # Read D-pad (usually via hat)
        if self.joystick.get_numhats() >= 1:
            hat = self.joystick.get_hat(0)
            self.buttons["DPAD_LEFT"] = 1 if hat[0] < 0 else 0
            self.buttons["DPAD_RIGHT"] = 1 if hat[0] > 0 else 0
            self.buttons["DPAD_DOWN"] = 1 if hat[1] < 0 else 0
            self.buttons["DPAD_UP"] = 1 if hat[1] > 0 else 0
        
        # Store previous button state for edge detection
        self.previous_buttons = self.buttons.copy()
        
        # Update robot positions based on controller input
        self._update_positions(self.axes, self.buttons)
    
    def _filter_deadzone(self, value: float, threshold: float = 0.1) -> float:
        """Apply deadzone filter to controller input."""
        if abs(value) < threshold:
            return 0.0
        return value
    
    def _update_positions(self, axes: Dict[str, float], buttons: Dict[str, int]):
        """Update robot positions based on controller input."""
        temp_positions = self.current_positions.copy()
        temp_x, temp_y = self.x, self.y
        
        # Handle macro buttons first
        used_macros = False
        macro_buttons = ["A", "B", "X", "Y"]
        for button in macro_buttons:
            if buttons.get(button):
                temp_positions = self._execute_macro(button, temp_positions)
                temp_x, temp_y = self._compute_position(
                    temp_positions["shoulder_lift"], temp_positions["elbow_flex"]
                )
                used_macros = True
                break
        
        if not used_macros:
            # Manual control
            temp_positions["wrist_roll"] += axes["RX"] * self.config.wrist_roll_speed
            temp_positions["wrist_flex"] -= axes["RY"] * self.config.wrist_flex_speed
            
            temp_positions["gripper"] -= self.config.gripper_speed * axes["RT"]
            temp_positions["gripper"] += self.config.gripper_speed * axes["LT"]
            
            temp_positions["shoulder_pan"] += (
                axes["LX"] - buttons["DPAD_LEFT"] + buttons["DPAD_RIGHT"]
            ) * self.config.shoulder_pan_speed
            
            temp_x = self.x + axes["LY"] * self.config.x_axis_speed
            temp_y = self.y + (buttons["DPAD_UP"] - buttons["DPAD_DOWN"]) * self.config.y_axis_speed
            
            # Compute shoulder_lift and elbow_flex using inverse kinematics
            try:
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"] = (
                    self._compute_inverse_kinematics(temp_x, temp_y)
                )
                
                # Adjust wrist_flex to maintain orientation
                shoulder_lift_change = temp_positions["shoulder_lift"] - self.current_positions["shoulder_lift"]
                elbow_flex_change = temp_positions["elbow_flex"] - self.current_positions["elbow_flex"]
                temp_positions["wrist_flex"] += shoulder_lift_change - elbow_flex_change
                
            except ValueError:
                temp_x = self.x
                temp_y = self.y
        else:
            temp_x, temp_y = self._compute_position(
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"]
            )
        
        # Validate and apply positions selectively
        self._apply_valid_positions(temp_positions, temp_x, temp_y)
    
    def _apply_valid_positions(self, temp_positions: Dict[str, float], temp_x: float, temp_y: float):
        """Apply only valid position changes, keeping current values for invalid ones."""
        updated_positions = self.current_positions.copy()
        any_limits_hit = False
        
        # Check each motor position individually
        for motor in temp_positions:
            if motor in self.config.position_limits:
                min_val, max_val = self.config.position_limits[motor]
                if min_val <= temp_positions[motor] <= max_val:
                    updated_positions[motor] = temp_positions[motor]
                else:
                    # Keep current position for this motor
                    any_limits_hit = True
            else:
                # No limits defined for this motor, apply change
                updated_positions[motor] = temp_positions[motor]
        
        # Check workspace limits for x,y coordinates
        x_limits = self.config.position_limits.get("x", (-float('inf'), float('inf')))
        y_limits = self.config.position_limits.get("y", (-float('inf'), float('inf')))
        
        temp_x_valid = x_limits[0] <= temp_x <= x_limits[1]
        temp_y_valid = y_limits[0] <= temp_y <= y_limits[1]
        
        # If x,y are valid, use the computed shoulder/elbow positions
        # If not, recompute shoulder/elbow from current x,y to maintain position
        if temp_x_valid and temp_y_valid:
            self.current_positions = updated_positions
            self.x = temp_x
            self.y = temp_y
        else:
            # Apply non-kinematic changes (wrist_roll, wrist_flex, gripper, shoulder_pan)
            non_kinematic_motors = ["wrist_roll", "wrist_flex", "gripper", "shoulder_pan"]
            for motor in non_kinematic_motors:
                if motor in updated_positions:
                    self.current_positions[motor] = updated_positions[motor]
            
            # Keep current x,y and recompute shoulder_lift/elbow_flex if needed
            if not temp_x_valid or not temp_y_valid:
                any_limits_hit = True
        
        if any_limits_hit:
            self.indicate_error()
    
    def _execute_macro(self, button: str, positions: Dict[str, float]) -> Dict[str, float]:
        """Execute predefined macro for button press."""
        if button in self.config.macros:
            macro_positions = self.config.macros[button]
            for name, pos in zip(self.motor_names, macro_positions, strict=False):
                positions[name] = pos
        return positions
    
    def _compute_inverse_kinematics(self, x: float, y: float) -> tuple[float, float]:
        """Compute inverse kinematics for 2-link arm."""
        l1 = self.l1
        l2 = self.l2
        
        distance = math.hypot(x, y)
        
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x}, {y}) is out of reach")
        
        # Compute elbow angle
        cos_theta2 = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        theta2_rad = math.acos(cos_theta2)
        theta2_deg = math.degrees(theta2_rad)
        
        offset = math.degrees(math.asin(32 / l1))
        motor3_angle = 180 - (theta2_deg - offset)
        
        # Compute shoulder angle  
        cos_theta1 = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        theta1_rad = math.acos(cos_theta1)
        theta1_deg = math.degrees(theta1_rad)
        alpha_rad = math.atan2(y, x)
        alpha_deg = math.degrees(alpha_rad)
        
        beta_deg = 180 - alpha_deg - theta1_deg
        motor2_angle = 180 - beta_deg + offset
        
        return motor2_angle, motor3_angle
    
    def _compute_position(self, motor2_angle: float, motor3_angle: float) -> tuple[float, float]:
        """Compute forward kinematics for 2-link arm."""
        l1 = self.l1
        l2 = self.l2
        offset = math.degrees(math.asin(32 / l1))
        
        beta_deg = 180 - motor2_angle + offset
        beta_rad = math.radians(beta_deg)
        
        theta2_deg = 180 - motor3_angle + offset
        theta2_rad = math.radians(theta2_deg)
        
        y = l1 * math.sin(beta_rad) - l2 * math.sin(beta_rad - theta2_rad)
        x = -l1 * math.cos(beta_rad) + l2 * math.cos(beta_rad - theta2_rad)
        
        return x, y
    
    def _is_position_valid(self, positions: Dict[str, float], x: float, y: float) -> bool:
        """Validate all motor positions and end-effector coordinates."""
        # Check motor limits
        for motor, (min_val, max_val) in self.config.position_limits.items():
            if motor in positions:
                if not (min_val <= positions[motor] <= max_val):
                    return False
        
        # Check workspace limits
        x_limits = self.config.position_limits["x"]
        y_limits = self.config.position_limits["y"]
        
        if not (x_limits[0] <= x <= x_limits[1]):
            return False
            
        if not (y_limits[0] <= y <= y_limits[1]):
            return False
        
        return True
    
    def indicate_error(self):
        """Indicate error condition to user."""
        logging.warning("Controller limits reached")
    
    def get_command(self) -> Dict[str, float]:
        """Get current motor position commands."""
        return self.current_positions.copy()
    
    def stop(self):
        """Stop controller and clean up resources."""
        self.disconnect()
        pygame.quit()


class XboxArmController(XboxController):
    """Extended Xbox controller for arm control."""
    
    def __init__(self, config: XboxControllerConfig):
        """Initialize Xbox arm controller."""
        super().__init__(config)
    
    def get_current_end_effector_position(self) -> tuple[float, float]:
        """Get current end-effector position."""
        return self.x, self.y
    
    def get_motor_positions(self) -> Dict[str, float]:
        """Get current motor positions."""
        return self.get_command() 