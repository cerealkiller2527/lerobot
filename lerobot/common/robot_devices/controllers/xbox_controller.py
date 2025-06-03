"""Xbox controller teleoperation using pygame for input handling.

This module adapts the PS4 controller logic to work with Xbox controllers
via pygame, providing the same functionality including simple inverse
kinematics, macros, and safety validation.

XBOX CONTROLLER SCHEME:
=======================
🕹️ LEFT STICK: Mixed Control
   • X-axis: Base rotation (shoulder pan)
   • Y-axis: Up/Down movement (Z-axis)

🕹️ RIGHT STICK: Mixed Control
   • X-axis: Forward/Backward movement
   • Y-axis: Wrist flex (up/down)

🎯 TRIGGERS: Gripper Control
   • Left Trigger (LT): Open gripper
   • Right Trigger (RT): Close gripper

🔘 SHOULDER BUTTONS: Wrist Rotation
   • Left Bumper (LB): Rotate wrist left
   • Right Bumper (RB): Rotate wrist right

⬆️ D-PAD: Fine Adjustment (optional)
   • Up/Down: Fine Y-axis adjustment (50% speed)
   • Left/Right: Fine X-axis adjustment (50% speed)

🔴 FACE BUTTONS: Preset Positions (Macros)
   • A: Home position
   • B: Reach forward  
   • X: Pick position
   • Y: Place position
"""

import copy
import logging
import math
import time
from typing import Dict, Any

import pygame

from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig


class XboxController:
    """Xbox controller for robot arm teleoperation.
    
    Provides the same functionality as the PS4 controller but uses pygame
    for input handling instead of the HID library. Includes simple 2-link
    inverse kinematics, macro commands, and safety validation.
    
    Note: This controller requires regular calls to update() from the main thread
    to process pygame events and controller input.
    """
    
    def __init__(self, config: XboxControllerConfig):
        """Initialize Xbox controller with configuration.
        
        Args:
            config: XboxControllerConfig with all settings
        """
        self.config = config
        
        # Motor and position state (from PS4 code)
        self.motor_names = config.motor_names
        self.current_positions = dict(zip(self.motor_names, config.initial_position, strict=False))
        self.new_positions = self.current_positions.copy()
        
        # Simple kinematics parameters (from PS4 code)
        self.l1 = config.l1  # First arm segment length (mm)
        self.l2 = config.l2  # Second arm segment length (mm)
        
        # Current end-effector position in mm
        self.x, self.y = self._compute_position(
            self.current_positions["shoulder_lift"], 
            self.current_positions["elbow_flex"]
        )
        
        # Controller state tracking
        self.axes = {
            "LX": 0.0,  # Left stick X
            "LY": 0.0,  # Left stick Y
            "RX": 0.0,  # Right stick X
            "RY": 0.0,  # Right stick Y
            "LT": 0.0,  # Left trigger (0.0 to 1.0)
            "RT": 0.0,  # Right trigger (0.0 to 1.0)
        }
        
        self.buttons = {
            "A": 0, "B": 0, "X": 0, "Y": 0,
            "LB": 0, "RB": 0,  # Left/Right bumpers
            "BACK": 0, "START": 0,
            "LS": 0, "RS": 0,  # Left/Right stick buttons
            "DPAD_UP": 0, "DPAD_DOWN": 0, 
            "DPAD_LEFT": 0, "DPAD_RIGHT": 0,
        }
        
        self.previous_buttons = self.buttons.copy()
        
        # Pygame joystick objects
        self.joystick = None
        self.device_index = config.device_index
        
        # Connection state
        self.connected = False
        
        # Initialize pygame and connect
        self._init_pygame()
        self.connect()
    
    def _init_pygame(self):
        """Initialize pygame joystick subsystem."""
        try:
            pygame.init()
            pygame.joystick.init()
            logging.info(f"Pygame initialized. Found {pygame.joystick.get_count()} joystick(s)")
        except Exception as e:
            logging.error(f"Failed to initialize pygame: {e}")
    
    def connect(self) -> bool:
        """Connect to Xbox controller.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
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
            logging.info(f"Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
            
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
                logging.info("Xbox controller disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting controller: {e}")
            finally:
                self.joystick = None
    
    def is_connected(self) -> bool:
        """Check if controller is connected.
        
        Returns:
            bool: True if connected
        """
        return self.joystick is not None and self.connected
    
    def update(self):
        """Update controller state and process input.
        
        This method should be called regularly from the main thread to:
        1. Process pygame events
        2. Read controller input
        3. Update robot positions
        
        Returns:
            bool: True if update was successful, False if controller disconnected
        """
        if not self.is_connected():
            return False
            
        try:
            # Process pygame events (required for joystick state updates)
            pygame.event.pump()
            
            # Read and process controller input
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
        """Apply deadzone filter to controller input to avoid drift.
        
        Simple approach copied from PS4 controller - just zero out values below threshold.
        
        Args:
            value: Raw controller input (-1.0 to 1.0)
            threshold: Deadzone threshold
            
        Returns:
            float: Filtered value with deadzone applied
        """
        if abs(value) < threshold:
            return 0.0
        return value
    
    def _update_positions(self, axes: Dict[str, float], buttons: Dict[str, int]):
        """Update robot positions based on controller input.
        
        This is adapted from the PS4 controller _update_positions method.
        
        Args:
            axes: Dictionary of analog stick and trigger values
            buttons: Dictionary of button states
        """
        temp_positions = self.current_positions.copy()
        
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
            # Manual control mode - intuitive movement with both sticks
            
            # === LEFT STICK: MIXED CONTROL ===
            # Left stick X controls shoulder_pan (base rotation) - SWITCHED!
            temp_positions["shoulder_pan"] += axes["LX"] * self.config.shoulder_pan_speed
            
            # Left stick Y changes Y coordinate (up/down)
            temp_y = self.y + axes["LY"] * self.config.y_axis_speed  # mm per update
            
            # === RIGHT STICK: MIXED CONTROL ===
            # Right stick X controls forward/backward movement (X coordinate) - Left = Forward!
            temp_x = self.x - axes["RX"] * self.config.x_axis_speed  # mm per update (REVERSED)
            # Right stick Y controls wrist_flex (up/down)
            temp_positions["wrist_flex"] -= axes["RY"] * self.config.wrist_flex_speed  # Inverted for intuitive control
            
            # === SHOULDER BUTTONS: WRIST ROTATION ===
            # Left/Right bumpers control wrist_roll - SWITCHED!
            temp_positions["wrist_roll"] += (buttons["RB"] - buttons["LB"]) * self.config.wrist_roll_speed
            
            # === TRIGGERS: GRIPPER ===
            temp_positions["gripper"] -= self.config.gripper_speed * axes["RT"]  # Right trigger closes
            temp_positions["gripper"] += self.config.gripper_speed * axes["LT"]  # Left trigger opens
            
            # === D-PAD: FINE ARM POSITION ADJUSTMENT ===
            # D-pad controls fine adjustment of arm position (not wrist)
            temp_x += (buttons["DPAD_LEFT"] - buttons["DPAD_RIGHT"]) * self.config.x_axis_speed * 0.5  # Fine X control
            temp_y += (buttons["DPAD_UP"] - buttons["DPAD_DOWN"]) * self.config.y_axis_speed * 0.5  # Fine Y control
            
            # Compute shoulder_lift and elbow_flex angles using inverse kinematics
            try:
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"] = (
                    self._compute_inverse_kinematics(temp_x, temp_y)
                )
                
                # Adjust wrist_flex to maintain end-effector orientation
                shoulder_lift_change = temp_positions["shoulder_lift"] - self.current_positions["shoulder_lift"]
                elbow_flex_change = temp_positions["elbow_flex"] - self.current_positions["elbow_flex"]
                temp_positions["wrist_flex"] += shoulder_lift_change - elbow_flex_change
                
                correct_ik = True
                
            except ValueError as e:
                logging.error(f"Inverse kinematics error: {e}")
                temp_x = self.x  # Revert to current position
                temp_y = self.y
                correct_ik = False
        else:
            # Macro was used, compute X/Y from the macro position
            temp_x, temp_y = self._compute_position(
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"]
            )
            correct_ik = True
        
        # Validate positions individually and only discard invalid ones
        any_invalid = False
        
        # Check each motor position individually
        for motor, (min_val, max_val) in self.config.position_limits.items():
            if motor in temp_positions:
                if not (min_val <= temp_positions[motor] <= max_val):
                    logging.warning(f"Motor '{motor}' position {temp_positions[motor]:.1f} out of range [{min_val}, {max_val}] - reverting this motor only")
                    temp_positions[motor] = self.current_positions[motor]  # Revert only this motor
                    any_invalid = True
        
        # Check end-effector position limits
        x_limits = self.config.position_limits["x"]
        y_limits = self.config.position_limits["y"]
        
        if not (x_limits[0] <= temp_x <= x_limits[1]):
            logging.warning(f"X position {temp_x:.1f} out of range {x_limits} - reverting X movement")
            temp_x = self.x  # Revert X position
            any_invalid = True
            
        if not (y_limits[0] <= temp_y <= y_limits[1]):
            logging.warning(f"Y position {temp_y:.1f} out of range {y_limits} - reverting Y movement")
            temp_y = self.y  # Revert Y position
            any_invalid = True
        
        # If X or Y was invalid, recompute shoulder_lift and elbow_flex
        if any_invalid and correct_ik:
            try:
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"] = (
                    self._compute_inverse_kinematics(temp_x, temp_y)
                )
                # Recompute wrist_flex adjustment
                shoulder_lift_change = temp_positions["shoulder_lift"] - self.current_positions["shoulder_lift"]
                elbow_flex_change = temp_positions["elbow_flex"] - self.current_positions["elbow_flex"]
                temp_positions["wrist_flex"] += shoulder_lift_change - elbow_flex_change
                
                correct_ik = True
                
            except ValueError as e:
                logging.error(f"Inverse kinematics error: {e}")
                temp_x = self.x  # Revert to current position
                temp_y = self.y
                correct_ik = False
        else:
            # Macro was used, compute X/Y from the macro position
            temp_x, temp_y = self._compute_position(
                temp_positions["shoulder_lift"], temp_positions["elbow_flex"]
            )
            correct_ik = True
        
        # Apply all valid changes (some positions may have been reverted individually)
        self.current_positions = temp_positions
        self.x = temp_x
        self.y = temp_y
        
        # Indicate error only if any limits were hit
        if any_invalid:
            self.indicate_error()
    
    def _execute_macro(self, button: str, positions: Dict[str, float]) -> Dict[str, float]:
        """Execute predefined macro for button press.
        
        Args:
            button: Button name ("A", "B", "X", "Y")
            positions: Current position dictionary to modify
            
        Returns:
            Dict[str, float]: Updated positions dictionary
        """
        if button in self.config.macros:
            macro_positions = self.config.macros[button]
            for name, pos in zip(self.motor_names, macro_positions, strict=False):
                positions[name] = pos
            logging.info(f"Macro '{button}' executed: {macro_positions}")
        return positions
    
    def _compute_inverse_kinematics(self, x: float, y: float) -> tuple[float, float]:
        """Compute inverse kinematics for 2-link arm.
        
        This is the exact implementation from the PS4 controller code.
        
        Args:
            x: Target X coordinate (mm)
            y: Target Y coordinate (mm)
            
        Returns:
            tuple[float, float]: (shoulder_lift_angle, elbow_flex_angle) in degrees
            
        Raises:
            ValueError: If target point is unreachable
        """
        l1 = self.l1
        l2 = self.l2
        
        # Compute distance from motor 2 to desired point
        distance = math.hypot(x, y)
        
        # Check if point is reachable
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x}, {y}) is out of reach (distance: {distance:.1f}mm)")
        
        # Compute angle for motor3 (elbow_flex)
        cos_theta2 = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        theta2_rad = math.acos(cos_theta2)
        theta2_deg = math.degrees(theta2_rad)
        
        # Adjust motor3 angle
        offset = math.degrees(math.asin(32 / l1))
        motor3_angle = 180 - (theta2_deg - offset)
        
        # Compute angle for motor2 (shoulder_lift)
        cos_theta1 = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        theta1_rad = math.acos(cos_theta1)
        theta1_deg = math.degrees(theta1_rad)
        alpha_rad = math.atan2(y, x)
        alpha_deg = math.degrees(alpha_rad)
        
        beta_deg = 180 - alpha_deg - theta1_deg
        motor2_angle = 180 - beta_deg + offset
        
        return motor2_angle, motor3_angle
    
    def _compute_position(self, motor2_angle: float, motor3_angle: float) -> tuple[float, float]:
        """Compute forward kinematics for 2-link arm.
        
        This is the exact implementation from the PS4 controller code.
        
        Args:
            motor2_angle: Shoulder lift angle (degrees)
            motor3_angle: Elbow flex angle (degrees)
            
        Returns:
            tuple[float, float]: (x, y) position in mm
        """
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
        """Validate all motor positions and end-effector coordinates.
        
        Args:
            positions: Dictionary of motor positions
            x: End-effector X coordinate 
            y: End-effector Y coordinate
            
        Returns:
            bool: True if all positions are valid
        """
        # Check motor position limits
        for motor, (min_val, max_val) in self.config.position_limits.items():
            if motor in positions:
                if not (min_val <= positions[motor] <= max_val):
                    logging.error(f"Motor '{motor}' position {positions[motor]:.1f} out of range [{min_val}, {max_val}]")
                    return False
        
        # Check end-effector position limits
        x_limits = self.config.position_limits["x"]
        y_limits = self.config.position_limits["y"]
        
        if not (x_limits[0] <= x <= x_limits[1]):
            logging.error(f"X position {x:.1f} out of range {x_limits}")
            return False
            
        if not (y_limits[0] <= y <= y_limits[1]):
            logging.error(f"Y position {y:.1f} out of range {y_limits}")
            return False
        
        return True
    
    def indicate_error(self):
        """Indicate error condition to user.
        
        For Xbox controller, we log the error. In future phases,
        we could add controller vibration feedback.
        """
        logging.warning("Invalid move attempted - controller limits reached")
        # TODO: Add controller vibration feedback if supported
    
    def get_command(self) -> Dict[str, float]:
        """Get current motor position commands.
        
        Returns:
            Dict[str, float]: Current motor positions
        """
        return self.current_positions.copy()
    
    def stop(self):
        """Stop controller and clean up resources."""
        self.disconnect()
        pygame.quit()
        logging.info("Xbox controller stopped")


class XboxArmController(XboxController):
    """Extended Xbox controller specifically for arm control.
    
    This class extends the base XboxController with additional
    arm-specific functionality and will be used in later phases
    for integration with the LeRobot system.
    """
    
    def __init__(self, config: XboxControllerConfig):
        """Initialize Xbox arm controller.
        
        Args:
            config: XboxControllerConfig with arm-specific settings
        """
        super().__init__(config)
        logging.info("Xbox arm controller initialized")
    
    def get_current_end_effector_position(self) -> tuple[float, float]:
        """Get current end-effector position.
        
        Returns:
            tuple[float, float]: Current (x, y) position in mm
        """
        return self.x, self.y
    
    def get_motor_positions(self) -> Dict[str, float]:
        """Get current motor positions.
        
        Returns:
            Dict[str, float]: Motor name to position mapping
        """
        return self.get_command() 