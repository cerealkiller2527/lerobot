"""Tests for Xbox controller functionality.

This test suite verifies that the Xbox controller implementation
works correctly, including input processing, kinematics, macros,
and safety validation.
"""

import math
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig
from lerobot.common.robot_devices.controllers.xbox_controller import XboxController, XboxArmController


class TestXboxControllerConfig(unittest.TestCase):
    """Test XboxControllerConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = XboxControllerConfig()
        
        # Check default values
        self.assertEqual(config.device_index, 0)
        self.assertEqual(config.dead_zone_sticks, 0.1)
        self.assertEqual(config.speed, 0.3)
        self.assertEqual(config.l1, 117.0)
        self.assertEqual(config.l2, 136.0)
        
        # Check motor names
        expected_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.assertEqual(config.motor_names, expected_motors)
        
        # Check initial position
        expected_initial = [90, 170, 170, 0, 0, 10]
        self.assertEqual(config.initial_position, expected_initial)
        
        # Check macros exist
        self.assertIn("A", config.macros)
        self.assertIn("B", config.macros)
        self.assertIn("X", config.macros)
        self.assertIn("Y", config.macros)
    
    def test_position_limits(self):
        """Test position limit validation."""
        config = XboxControllerConfig()
        
        # Check that all required limits are present
        required_limits = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper", "x", "y"]
        for limit in required_limits:
            self.assertIn(limit, config.position_limits)
            self.assertEqual(len(config.position_limits[limit]), 2)  # min, max


class TestXboxControllerKinematics(unittest.TestCase):
    """Test kinematics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = XboxControllerConfig()
        
        # Mock pygame to avoid requiring actual controller
        with patch('pygame.init'), patch('pygame.joystick.init'), patch('pygame.joystick.get_count', return_value=0):
            self.controller = XboxController(self.config)
    
    def test_forward_kinematics(self):
        """Test forward kinematics calculation."""
        # Test with known values
        motor2_angle = 170.0  # shoulder_lift
        motor3_angle = 170.0  # elbow_flex
        
        x, y = self.controller._compute_position(motor2_angle, motor3_angle)
        
        # Verify the result is reasonable (end-effector should be within arm reach)
        distance = math.hypot(x, y)
        max_reach = self.config.l1 + self.config.l2
        self.assertLessEqual(distance, max_reach)
        
        # Test with initial position from config
        initial_shoulder = self.config.initial_position[1]  # shoulder_lift
        initial_elbow = self.config.initial_position[2]     # elbow_flex
        
        x_init, y_init = self.controller._compute_position(initial_shoulder, initial_elbow)
        self.assertIsInstance(x_init, float)
        self.assertIsInstance(y_init, float)
    
    def test_inverse_kinematics(self):
        """Test inverse kinematics calculation."""
        # Test with reachable point
        x, y = 100.0, 50.0  # mm
        
        try:
            motor2, motor3 = self.controller._compute_inverse_kinematics(x, y)
            self.assertIsInstance(motor2, float)
            self.assertIsInstance(motor3, float)
            
            # Verify forward kinematics gives back original point
            x_check, y_check = self.controller._compute_position(motor2, motor3)
            self.assertAlmostEqual(x, x_check, places=1)
            self.assertAlmostEqual(y, y_check, places=1)
            
        except ValueError:
            self.fail("Inverse kinematics failed for reachable point")
    
    def test_inverse_kinematics_unreachable(self):
        """Test inverse kinematics with unreachable point."""
        # Point far outside reach
        x, y = 1000.0, 1000.0  # mm
        
        with self.assertRaises(ValueError):
            self.controller._compute_inverse_kinematics(x, y)
    
    def test_kinematics_consistency(self):
        """Test forward and inverse kinematics consistency."""
        # Start with some joint angles
        test_angles = [
            (150.0, 150.0),
            (170.0, 170.0), 
            (130.0, 160.0),
        ]
        
        for motor2, motor3 in test_angles:
            # Forward kinematics
            x, y = self.controller._compute_position(motor2, motor3)
            
            # Check if point is reachable
            distance = math.hypot(x, y)
            if distance <= (self.config.l1 + self.config.l2) and distance >= abs(self.config.l1 - self.config.l2):
                # Inverse kinematics
                motor2_back, motor3_back = self.controller._compute_inverse_kinematics(x, y)
                
                # Check consistency (allowing for multiple solutions)
                x_back, y_back = self.controller._compute_position(motor2_back, motor3_back)
                self.assertAlmostEqual(x, x_back, places=1)
                self.assertAlmostEqual(y, y_back, places=1)


class TestXboxControllerSafety(unittest.TestCase):
    """Test safety and validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = XboxControllerConfig()
        
        # Mock pygame
        with patch('pygame.init'), patch('pygame.joystick.init'), patch('pygame.joystick.get_count', return_value=0):
            self.controller = XboxController(self.config)
    
    def test_position_validation_valid(self):
        """Test position validation with valid positions."""
        # Use initial position (should be valid)
        positions = dict(zip(self.config.motor_names, self.config.initial_position, strict=False))
        x, y = self.controller._compute_position(positions["shoulder_lift"], positions["elbow_flex"])
        
        is_valid = self.controller._is_position_valid(positions, x, y)
        self.assertTrue(is_valid)
    
    def test_position_validation_invalid_motor(self):
        """Test position validation with invalid motor position."""
        positions = dict(zip(self.config.motor_names, self.config.initial_position, strict=False))
        
        # Set shoulder_pan out of range
        positions["shoulder_pan"] = 300.0  # Max is 190
        x, y = self.controller._compute_position(positions["shoulder_lift"], positions["elbow_flex"])
        
        is_valid = self.controller._is_position_valid(positions, x, y)
        self.assertFalse(is_valid)
    
    def test_position_validation_invalid_coordinates(self):
        """Test position validation with invalid end-effector coordinates."""
        positions = dict(zip(self.config.motor_names, self.config.initial_position, strict=False))
        
        # Invalid X coordinate
        x, y = 500.0, 0.0  # X max is 250
        
        is_valid = self.controller._is_position_valid(positions, x, y)
        self.assertFalse(is_valid)
    
    def test_deadzone_filter(self):
        """Test deadzone filtering."""
        # Values within deadzone should be filtered to 0
        self.assertEqual(self.controller._filter_deadzone(0.05, 0.1), 0.0)
        self.assertEqual(self.controller._filter_deadzone(-0.05, 0.1), 0.0)
        
        # Values outside deadzone should pass through
        self.assertEqual(self.controller._filter_deadzone(0.15, 0.1), 0.15)
        self.assertEqual(self.controller._filter_deadzone(-0.15, 0.1), -0.15)


class TestXboxControllerMacros(unittest.TestCase):
    """Test macro functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = XboxControllerConfig()
        
        # Mock pygame
        with patch('pygame.init'), patch('pygame.joystick.init'), patch('pygame.joystick.get_count', return_value=0):
            self.controller = XboxController(self.config)
    
    def test_macro_execution(self):
        """Test macro execution."""
        initial_positions = self.controller.current_positions.copy()
        
        # Execute macro A (initial position)
        new_positions = self.controller._execute_macro("A", initial_positions.copy())
        
        # Should match the macro values
        expected = dict(zip(self.config.motor_names, self.config.macros["A"], strict=False))
        for motor in self.config.motor_names:
            self.assertEqual(new_positions[motor], expected[motor])
    
    def test_all_macros_valid(self):
        """Test that all predefined macros have valid positions."""
        for button, macro_values in self.config.macros.items():
            positions = dict(zip(self.config.motor_names, macro_values, strict=False))
            
            # Compute end-effector position for the macro
            x, y = self.controller._compute_position(positions["shoulder_lift"], positions["elbow_flex"])
            
            # Validate the macro position
            is_valid = self.controller._is_position_valid(positions, x, y)
            self.assertTrue(is_valid, f"Macro '{button}' has invalid position: {macro_values}")


class TestXboxControllerUpdate(unittest.TestCase):
    """Test controller update functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = XboxControllerConfig()
    
    @patch('pygame.init')
    @patch('pygame.joystick.init')
    @patch('pygame.joystick.get_count')
    @patch('pygame.joystick.Joystick')
    @patch('pygame.event.pump')
    def test_update_success(self, mock_pump, mock_joystick_class, mock_get_count, mock_joystick_init, mock_pygame_init):
        """Test successful controller update."""
        # Mock successful connection
        mock_get_count.return_value = 1
        mock_joystick = MagicMock()
        mock_joystick.get_name.return_value = "Xbox Controller"
        mock_joystick.get_numaxes.return_value = 6
        mock_joystick.get_numbuttons.return_value = 11
        mock_joystick.get_numhats.return_value = 1
        mock_joystick.get_axis.return_value = 0.0
        mock_joystick.get_button.return_value = 0
        mock_joystick.get_hat.return_value = (0, 0)
        mock_joystick_class.return_value = mock_joystick
        
        controller = XboxController(self.config)
        
        # Test update
        result = controller.update()
        
        # Should succeed
        self.assertTrue(result)
        mock_pump.assert_called_once()
    
    def test_update_not_connected(self):
        """Test update when controller is not connected."""
        with patch('pygame.init'), patch('pygame.joystick.init'), patch('pygame.joystick.get_count', return_value=0):
            controller = XboxController(self.config)
            
            # Update should fail
            result = controller.update()
            self.assertFalse(result)


@patch('pygame.init')
@patch('pygame.joystick.init') 
@patch('pygame.joystick.get_count')
@patch('pygame.joystick.Joystick')
class TestXboxControllerConnection(unittest.TestCase):
    """Test controller connection and pygame integration."""
    
    def test_connection_success(self, mock_joystick_class, mock_get_count, mock_joystick_init, mock_pygame_init):
        """Test successful controller connection."""
        # Mock successful connection
        mock_get_count.return_value = 1
        mock_joystick = MagicMock()
        mock_joystick.get_name.return_value = "Xbox Controller"
        mock_joystick.get_numaxes.return_value = 6
        mock_joystick.get_numbuttons.return_value = 11
        mock_joystick_class.return_value = mock_joystick
        
        config = XboxControllerConfig()
        controller = XboxController(config)
        
        # Verify connection
        self.assertTrue(controller.is_connected())
        self.assertEqual(controller.joystick, mock_joystick)
        mock_joystick.init.assert_called_once()
    
    def test_connection_no_controllers(self, mock_joystick_class, mock_get_count, mock_joystick_init, mock_pygame_init):
        """Test connection when no controllers are available."""
        mock_get_count.return_value = 0
        
        config = XboxControllerConfig()
        controller = XboxController(config)
        
        # Should fail to connect
        self.assertFalse(controller.is_connected())
        self.assertIsNone(controller.joystick)
    
    def test_connection_invalid_index(self, mock_joystick_class, mock_get_count, mock_joystick_init, mock_pygame_init):
        """Test connection with invalid device index."""
        mock_get_count.return_value = 1  # Only one controller
        
        config = XboxControllerConfig(device_index=5)  # Request index 5
        controller = XboxController(config)
        
        # Should fail to connect
        self.assertFalse(controller.is_connected())
        self.assertIsNone(controller.joystick)


class TestXboxArmController(unittest.TestCase):
    """Test XboxArmController extended functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = XboxControllerConfig()
        
        # Mock pygame
        with patch('pygame.init'), patch('pygame.joystick.init'), patch('pygame.joystick.get_count', return_value=0):
            self.controller = XboxArmController(self.config)
    
    def test_initialization(self):
        """Test XboxArmController initialization."""
        self.assertIsInstance(self.controller, XboxController)
        self.assertIsNotNone(self.controller.config)
    
    def test_get_current_end_effector_position(self):
        """Test getting current end-effector position."""
        x, y = self.controller.get_current_end_effector_position()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
    
    def test_get_motor_positions(self):
        """Test getting motor positions."""
        positions = self.controller.get_motor_positions()
        
        # Should return dictionary with all motor names
        for motor in self.config.motor_names:
            self.assertIn(motor, positions)
            self.assertIsInstance(positions[motor], (int, float))


if __name__ == "__main__":
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main() 