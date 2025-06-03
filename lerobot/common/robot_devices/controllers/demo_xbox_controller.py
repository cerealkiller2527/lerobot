#!/usr/bin/env python3
"""Demo script for testing Xbox controller functionality.

This script demonstrates the Xbox controller implementation without
requiring a real robot. It shows controller input, kinematics, and
macro execution in real-time.

Usage:
    python demo_xbox_controller.py

Requirements:
    - Xbox controller connected via USB or wireless
    - pygame library installed
"""

import logging
import time
import signal
import sys
from typing import Dict, Any

from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig
from lerobot.common.robot_devices.controllers.xbox_controller import XboxArmController


class XboxControllerDemo:
    """Demo class to showcase Xbox controller functionality."""
    
    def __init__(self):
        """Initialize the demo."""
        self.setup_logging()
        self.controller = None
        self.running = False
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def setup_logging(self):
        """Configure logging for demo."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logging.info("Shutting down demo...")
        self.running = False
        if self.controller:
            self.controller.stop()
        sys.exit(0)
    
    def run(self):
        """Run the Xbox controller demo."""
        logging.info("="*60)
        logging.info("Xbox Controller Demo for LeRobot")
        logging.info("="*60)
        logging.info("Controls:")
        logging.info("  Left Stick X/Y: Arm cartesian movement")
        logging.info("  Right Stick X/Y: Wrist control")
        logging.info("  Triggers: Gripper control")
        logging.info("  D-Pad: Precise movement")
        logging.info("  A/B/X/Y Buttons: Macro positions")
        logging.info("  Ctrl+C: Exit")
        logging.info("="*60)
        
        # Create controller config
        config = XboxControllerConfig()
        logging.info(f"Controller config: speed={config.speed}, device_index={config.device_index}")
        
        # Initialize controller
        try:
            self.controller = XboxArmController(config)
            
            if not self.controller.is_connected():
                logging.error("Failed to connect to Xbox controller")
                logging.error("Please ensure:")
                logging.error("1. Xbox controller is connected via USB or wireless")
                logging.error("2. Controller drivers are installed")
                logging.error("3. pygame can detect the controller")
                return
                
            logging.info("Xbox controller connected successfully!")
            logging.info(f"Controller: {self.controller.joystick.get_name()}")
            
        except Exception as e:
            logging.error(f"Failed to initialize controller: {e}")
            return
        
        # Main demo loop
        self.running = True
        last_print_time = 0
        last_positions = None
        
        try:
            while self.running:
                current_time = time.time()
                
                # Update controller (process pygame events and input)
                if not self.controller.update():
                    logging.error("Controller update failed - controller may be disconnected")
                    break
                
                # Print status every 0.5 seconds
                if current_time - last_print_time >= 0.5:
                    self.print_status()
                    last_print_time = current_time
                
                # Check for position changes
                current_positions = self.controller.get_motor_positions()
                if current_positions != last_positions:
                    self.print_position_change(current_positions)
                    last_positions = current_positions.copy()
                
                time.sleep(0.05)  # 20Hz main loop
                
        except KeyboardInterrupt:
            pass
        finally:
            if self.controller:
                self.controller.stop()
            logging.info("Demo completed")
    
    def print_status(self):
        """Print current controller and robot status."""
        if not self.controller or not self.controller.is_connected():
            logging.warning("Controller not connected")
            return
        
        # Get current state
        positions = self.controller.get_motor_positions()
        x, y = self.controller.get_current_end_effector_position()
        
        # Get raw controller inputs
        axes = self.controller.axes
        buttons = {k: v for k, v in self.controller.buttons.items() if v}
        
        # Format active buttons properly
        active_buttons = ', '.join(buttons.keys()) if buttons else 'None'
        
        # Print compact status
        print(f"\r[{time.strftime('%H:%M:%S')}] "
              f"EE: ({x:6.1f}, {y:6.1f}) | "
              f"Shoulder: {positions['shoulder_pan']:5.1f}° | "
              f"Gripper: {positions['gripper']:5.1f} | "
              f"Active: {active_buttons:<10}", 
              end='', flush=True)
    
    def print_position_change(self, positions: Dict[str, float]):
        """Print position changes when they occur."""
        x, y = self.controller.get_current_end_effector_position()
        
        logging.info(f"Position update:")
        logging.info(f"  End-effector: ({x:.1f}, {y:.1f}) mm")
        logging.info(f"  Shoulder pan: {positions['shoulder_pan']:.1f}°")
        logging.info(f"  Shoulder lift: {positions['shoulder_lift']:.1f}°")
        logging.info(f"  Elbow flex: {positions['elbow_flex']:.1f}°")
        logging.info(f"  Wrist flex: {positions['wrist_flex']:.1f}°")
        logging.info(f"  Wrist roll: {positions['wrist_roll']:.1f}°")
        logging.info(f"  Gripper: {positions['gripper']:.1f}")


def test_controller_detection():
    """Test and display available controllers."""
    import pygame
    
    try:
        pygame.init()
        pygame.joystick.init()
        
        count = pygame.joystick.get_count()
        print(f"Found {count} joystick(s):")
        
        for i in range(count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            print(f"  {i}: {joystick.get_name()}")
            print(f"      Axes: {joystick.get_numaxes()}")
            print(f"      Buttons: {joystick.get_numbuttons()}")
            print(f"      Hats: {joystick.get_numhats()}")
            joystick.quit()
            
        return count > 0
        
    except Exception as e:
        print(f"Error detecting controllers: {e}")
        return False
    finally:
        pygame.quit()


def test_kinematics():
    """Test the kinematics calculations."""
    from lerobot.common.robot_devices.controllers.configs import XboxControllerConfig
    from lerobot.common.robot_devices.controllers.xbox_controller import XboxController
    
    import unittest.mock
    
    print("Testing kinematics...")
    
    config = XboxControllerConfig()
    
    # Mock pygame to avoid needing controller for kinematics test
    with unittest.mock.patch('pygame.init'), \
         unittest.mock.patch('pygame.joystick.init'), \
         unittest.mock.patch('pygame.joystick.get_count', return_value=0):
        
        controller = XboxController(config)
        
        # Test forward kinematics
        print("Forward kinematics test:")
        for angles in [(170, 170), (150, 160), (130, 140)]:
            shoulder, elbow = angles
            x, y = controller._compute_position(shoulder, elbow)
            print(f"  Angles ({shoulder}°, {elbow}°) -> Position ({x:.1f}, {y:.1f}) mm")
        
        # Test inverse kinematics
        print("\nInverse kinematics test:")
        for point in [(100, 50), (150, 100), (80, 30)]:
            x, y = point
            try:
                shoulder, elbow = controller._compute_inverse_kinematics(x, y)
                print(f"  Position ({x}, {y}) mm -> Angles ({shoulder:.1f}°, {elbow:.1f}°)")
                
                # Verify by forward kinematics
                x_check, y_check = controller._compute_position(shoulder, elbow)
                error = ((x - x_check)**2 + (y - y_check)**2)**0.5
                print(f"    Verification error: {error:.2f} mm")
                
            except ValueError as e:
                print(f"  Position ({x}, {y}) mm -> {e}")
        
        print("\nKinematics test completed!")


if __name__ == "__main__":
    print("Xbox Controller Demo for LeRobot")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-detection":
        test_controller_detection()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-kinematics":
        test_kinematics()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("  python demo_xbox_controller.py              # Run full demo")
        print("  python demo_xbox_controller.py --test-detection  # Test controller detection")
        print("  python demo_xbox_controller.py --test-kinematics # Test kinematics")
        sys.exit(0)
    
    # Test controller detection first
    if not test_controller_detection():
        print("\nNo controllers found. Please connect an Xbox controller and try again.")
        sys.exit(1)
    
    # Run main demo
    demo = XboxControllerDemo()
    demo.run() 