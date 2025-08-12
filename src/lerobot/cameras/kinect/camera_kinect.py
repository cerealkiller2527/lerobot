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
Provides the KinectCamera class for capturing frames from Microsoft Kinect v2 cameras.
"""

import gc
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    import pylibfreenect2
    from pylibfreenect2 import (
        FrameMap,
        FrameType,
        Freenect2,
        Freenect2Device,
        LoggerLevel,
        Registration,
        SyncMultiFrameListener,
        createConsoleLogger,
        setGlobalLogger,
    )

    KINECT_AVAILABLE = True

    # Set libfreenect2 logging to only show errors (suppress verbose output)
    logger_freenect = createConsoleLogger(LoggerLevel.Error)
    setGlobalLogger(logger_freenect)

except ImportError as e:
    KINECT_AVAILABLE = False
    logging.info(f"Could not import pylibfreenect2: {e}")

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_kinect import KinectCameraConfig, KinectPipeline

logger = logging.getLogger(__name__)


class KinectCamera(Camera):
    """
    Manages interactions with Microsoft Kinect v2 cameras for BGRA frame recording.

    This class provides an interface similar to `RealSenseCamera` but tailored for
    Kinect v2 devices, leveraging the `pylibfreenect2` library. It supports multiple
    GPU-accelerated processing pipelines (CUDA, OpenCL, OpenGL) with automatic fallback
    to CPU processing.

    The Kinect v2 provides:
    - Color stream: 1920x1080 @ 30 FPS (BGRA format)

    Key features:
    - GPU acceleration: Automatic pipeline selection for optimal performance
    - BGRA to RGB/BGR conversion

    Use the provided utility script to find available Kinect devices:
    ```bash
    python -m lerobot.find_cameras kinect
    ```

    Example:
        ```python
        from lerobot.cameras.kinect import KinectCamera, KinectCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with auto-detection
        config = KinectCameraConfig(device_index=0)
        camera = KinectCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)  # (1080, 1920, 3)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect
        camera.disconnect()

        ```
    """

    def __init__(self, config: KinectCameraConfig):
        """
        Initializes the KinectCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        if not KINECT_AVAILABLE:
            import os

            libfreenect2_prefix = os.environ.get("LIBFREENECT2_INSTALL_PREFIX")
            raise ImportError(
                "\n" + "=" * 60 + "\n"
                "Kinect v2 camera support is not available.\n\n"
                "To use Kinect v2 cameras, you need to:\n\n"
                "1. Install libfreenect2 library first:\n"
                "   - Windows: Follow https://github.com/OpenKinect/libfreenect2#windows\n"
                "   - Linux: Follow https://github.com/OpenKinect/libfreenect2#linux\n"
                "   - macOS: Follow https://github.com/OpenKinect/libfreenect2#macos\n\n"
                "2. Set the LIBFREENECT2_INSTALL_PREFIX environment variable:\n"
                f"   Current value: {libfreenect2_prefix or 'NOT SET'}\n"
                "   Example: export LIBFREENECT2_INSTALL_PREFIX=/path/to/libfreenect2\n\n"
                "3. Install the Python bindings:\n"
                "   pip install 'lerobot[kinect]'\n"
                "   or\n"
                "   pip install git+https://github.com/cerealkiller2527/pylibfreenect2-py310.git\n\n"
                "Note: Python 3.10+ is required for Kinect v2 support.\n" + "=" * 60
            )

        super().__init__(config)

        self.config = config
        self.device_index = config.device_index
        self.serial_number = config.serial_number
        self.fps = config.fps or 30  # Kinect v2 default
        self.color_mode = config.color_mode
        self.pipeline_type = config.pipeline
        self.warmup_s = config.warmup_s

        # Kinect v2 native color resolution
        self.color_width = 1920
        self.color_height = 1080

        # Set output dimensions - default to 960x540 for better performance
        if config.width is None or config.height is None:
            # Default to 960x540 - perfect 1:2 scaling of 1920x1080
            # This reduces data by 4x compared to full HD while maintaining aspect ratio
            self.width = 960
            self.height = 540
        else:
            # Support common resolutions
            # Perfect scaling (16:9): 1920x1080, 1280x720, 960x540, 640x360, 480x270, 320x180
            # Near 16:9: 848x480, 640x480 (4:3, requires padding)
            self.width = config.width
            self.height = config.height
            
            # Log resolution info for debugging
            scale_factor = (self.color_width * self.color_height) / (self.width * self.height)
            logger.debug(f"Kinect output resolution: {self.width}x{self.height} (scale factor: {scale_factor:.1f}x)")

        # libfreenect2 objects
        self.fn2: Freenect2 | None = None
        self.device: Freenect2Device | None = None
        self.listener: SyncMultiFrameListener | None = None
        self.registration: Registration | None = None
        self.pipeline = None

        # Threading for async reads
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # Handle rotation for capture dimensions
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width
        
        # Pre-compute processing flags for performance
        self._needs_resize = (self.width != self.color_width or self.height != self.color_height)
        self._has_rotation = self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        
        # Pre-allocate FrameMap for reuse
        self._frame_map = None

    def __str__(self) -> str:
        identifier = self.serial_number if self.serial_number else f"index_{self.device_index}"
        return f"{self.__class__.__name__}({identifier})"
    

    @property
    def is_connected(self) -> bool:
        """Checks if the Kinect device is connected and started."""
        return self.device is not None and self.listener is not None
    
    
    def connect(self, warmup: bool = True):
        """
        Connects to the Kinect v2 camera specified in the configuration.

        Initializes the libfreenect2 pipeline, configures the color stream,
        starts the device, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If no Kinect devices are found or connection fails.
            RuntimeError: If the device starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Simple approach: Initialize Freenect2 with a small delay for USB stability
        time.sleep(0.5)  # Small delay for USB stability after previous disconnect
        
        # Initialize Freenect2
        self.fn2 = Freenect2()

        # Check for available devices
        num_devices = self.fn2.enumerateDevices()
        if num_devices == 0:
            # One retry after a delay
            time.sleep(1.0)
            num_devices = self.fn2.enumerateDevices()
            if num_devices == 0:
                raise ConnectionError(
                    "No Kinect v2 devices found. Ensure device is connected to USB 3.0 port."
                )

        # Select pipeline
        self.pipeline = self._create_pipeline()

        # Open device - simplified approach
        try:
            if self.serial_number:
                # Convert to string if bytes
                serial_str = self.serial_number
                if isinstance(serial_str, bytes):
                    serial_str = serial_str.decode("utf-8")
                self.device = self.fn2.openDevice(serial_str, pipeline=self.pipeline)
            elif self.device_index is not None:
                if self.device_index >= num_devices:
                    raise ConnectionError(
                        f"Device index {self.device_index} not found. "
                        f"Only {num_devices} Kinect device(s) available."
                    )
                device_serial = self.fn2.getDeviceSerialNumber(self.device_index)
                self.device = self.fn2.openDevice(device_serial, pipeline=self.pipeline)
            else:
                # Open first available device
                self.device = self.fn2.openDefaultDevice(pipeline=self.pipeline)
        except Exception as e:
            # If we get LIBUSB_ERROR_OTHER, suggest a restart
            if "LIBUSB_ERROR_OTHER" in str(e):
                raise ConnectionError(
                    f"USB error opening Kinect: {e}\n"
                    "This usually happens after an improper shutdown.\n" 
                    "Please unplug and replug the Kinect, then try again."
                )
            raise

        if self.device is None:
            raise ConnectionError(f"Failed to open {self}. Ensure Kinect v2 is connected to a USB 3.0 port.")

        # Store actual serial number
        if not self.serial_number:
            self.serial_number = self.device.getSerialNumber()

        # Configure frame listener for color only
        frame_types = FrameType.Color
        self.listener = SyncMultiFrameListener(frame_types)
        self.device.setColorFrameListener(self.listener)

        # Start device
        self.device.start()

        # Start async thread for better performance
        self._start_read_thread()

        if warmup:
            time.sleep(0.5)  # Initial warmup
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    # Use async_read during warmup to populate the buffer
                    self.async_read(timeout_ms=100)
                except Exception:
                    pass  # nosec
                time.sleep(0.1)

        logger.info(
            f"{self} connected with {self.pipeline.__class__.__name__}: "
            f"color={self.width}x{self.height}@{self.fps}, "
            f"rotation={'none' if self.rotation is None else self.rotation}"
        )

    def _create_pipeline(self):
        """Creates the appropriate processing pipeline based on configuration."""
        if self.pipeline_type == KinectPipeline.AUTO:
            # Try pipelines in order of performance: CUDA > OpenCL > CPU
            try:
                if hasattr(pylibfreenect2, "CudaPacketPipeline"):
                    return pylibfreenect2.CudaPacketPipeline()
            except Exception:
                pass
            
            try:
                if hasattr(pylibfreenect2, "OpenCLPacketPipeline"):
                    return pylibfreenect2.OpenCLPacketPipeline()
            except Exception:
                pass
            
            # CPU pipeline is always available
            return pylibfreenect2.CpuPacketPipeline()
        
        # Specific pipeline requested
        elif self.pipeline_type == KinectPipeline.CUDA:
            try:
                return pylibfreenect2.CudaPacketPipeline()
            except Exception as e:
                raise RuntimeError(f"CUDA pipeline not available: {e}")
        
        elif self.pipeline_type == KinectPipeline.OPENCL:
            try:
                return pylibfreenect2.OpenCLPacketPipeline()
            except Exception as e:
                raise RuntimeError(f"OpenCL pipeline not available: {e}")
        
        elif self.pipeline_type == KinectPipeline.OPENGL:
            try:
                return pylibfreenect2.OpenGLPacketPipeline()
            except Exception as e:
                raise RuntimeError(f"OpenGL pipeline not available: {e}")
        
        else:  # CPU
            return pylibfreenect2.CpuPacketPipeline()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Kinect v2 cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing camera information.

        Raises:
            ImportError: If pylibfreenect2 is not installed.
        """
        if not KINECT_AVAILABLE:
            import os

            libfreenect2_prefix = os.environ.get("LIBFREENECT2_INSTALL_PREFIX")
            raise ImportError(
                "\n" + "=" * 60 + "\n"
                "Cannot detect Kinect cameras - pylibfreenect2 is not installed.\n\n"
                "To use Kinect v2 cameras:\n"
                "1. Install libfreenect2 library\n"
                "2. Set LIBFREENECT2_INSTALL_PREFIX environment variable\n"
                f"   (Current: {libfreenect2_prefix or 'NOT SET'})\n"
                "3. Install: pip install 'lerobot[kinect]'\n" + "=" * 60
            )

        found_cameras = []
        fn2 = Freenect2()
        try:
            num_devices = fn2.enumerateDevices()

            for i in range(num_devices):
                serial = fn2.getDeviceSerialNumber(i)
                # Decode serial number if it's bytes
                if isinstance(serial, bytes):
                    serial = serial.decode("utf-8")
                camera_info = {
                    "type": "Kinect v2",
                    "id": serial,
                    "index": i,
                    "name": f"Kinect v2 #{i}",
                    "serial_number": serial,
                    "color_resolution": "1920x1080",
                    "ir_resolution": "512x424",
                    "max_fps": 30,
                    "available_pipelines": [],
                }

                # Check available pipelines
                if hasattr(pylibfreenect2, "CudaPacketPipeline"):
                    camera_info["available_pipelines"].append("CUDA")
                if hasattr(pylibfreenect2, "OpenCLPacketPipeline"):
                    camera_info["available_pipelines"].append("OpenCL")
                if hasattr(pylibfreenect2, "OpenGLPacketPipeline"):
                    camera_info["available_pipelines"].append("OpenGL")
                camera_info["available_pipelines"].append("CPU")

                found_cameras.append(camera_info)
        finally:
            # Important: Clean up the Freenect2 instance to release USB device
            del fn2
            gc.collect()  # Force garbage collection to ensure device is released
            time.sleep(0.1)  # Give USB device time to reset

        return found_cameras

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single color frame synchronously from the camera.

        For performance, this method uses the async thread to get frames instead of
        blocking on waitForNewFrame. This matches the pattern used by other cameras
        in LeRobot and avoids the ~1 second delays seen with direct synchronous reads.

        Args:
            color_mode: Desired color mode for the output frame.
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured color frame as a NumPy array (1080, 1920, 3),
                        processed according to color_mode and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames fails.
            ValueError: If an invalid color_mode is requested.
        """
        # Use async_read for better performance - this is what all robots use anyway
        return self.async_read(timeout_ms=timeout_ms)

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.
        Optimized for minimal latency - processes BGRA to RGB conversion.
        """
        # Pre-allocate FrameMap once to avoid repeated allocations
        if self._frame_map is None:
            self._frame_map = FrameMap()
        frames = self._frame_map
        
        # Pre-allocate buffers for optimized frame retrieval
        color_buffer_bgr = np.empty((self.color_height, self.color_width, 3), dtype=np.uint8)
        
        # Calculate optimal timeout based on FPS (with small buffer)
        timeout_ms = int(1000 / self.fps * 1.5) if self.fps else 50
        
        while not self.stop_event.is_set():
            try:
                # Reuse the same FrameMap object
                frames = self._frame_map
                # Use FPS-appropriate timeout
                if self.listener.waitForNewFrame(frames, timeout_ms):

                    try:
                        # Process color frame - Use optimized method if available
                        color_frame = frames[FrameType.Color]
                        
                        # Try to use optimized method first (40-60% faster)
                        if hasattr(color_frame, 'asarray_optimized'):
                            # asarray_optimized directly converts BGRA→BGR and drops alpha
                            # This avoids the 8.3MB BGRA allocation and uses pre-allocated buffer
                            color_bgr = color_frame.asarray_optimized(color_buffer_bgr)
                            
                            # Optimize: combine resize + color conversion when downscaling
                            if self._needs_resize and self.width < self.color_width:
                                # Resize first (reduces data), then convert color
                                # INTER_AREA is best for downscaling quality + speed
                                color_bgr_resized = cv2.resize(
                                    color_bgr, (self.width, self.height), 
                                    interpolation=cv2.INTER_AREA
                                )
                                color_data = cv2.cvtColor(color_bgr_resized, cv2.COLOR_BGR2RGB)
                            elif self._needs_resize:
                                # Upscaling: convert color first, then resize
                                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                                color_data = cv2.resize(
                                    color_rgb, (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR
                                )
                            else:
                                # No resize needed
                                color_data = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                        else:
                            # Fallback to standard method
                            color_bgra = color_frame.asarray()
                            
                            # Optimize: resize BGRA first when downscaling (processes less data)
                            if self._needs_resize and self.width < self.color_width:
                                # Resize with alpha channel intact
                                color_bgra_resized = cv2.resize(
                                    color_bgra, (self.width, self.height),
                                    interpolation=cv2.INTER_AREA
                                )
                                # Then convert to RGB (drops alpha)
                                color_data = cv2.cvtColor(color_bgra_resized, cv2.COLOR_BGRA2RGB)
                            elif self._needs_resize:
                                # Upscaling: convert first, then resize
                                color_rgb = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)
                                color_data = cv2.resize(
                                    color_rgb, (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR
                                )
                            else:
                                # No resize needed
                                color_data = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2RGB)

                        # Use pre-computed rotation flag
                        if self._has_rotation:
                            color_data = cv2.rotate(color_data, self.rotation)

                        # Store frame thread-safely (minimal time in lock)
                        with self.frame_lock:
                            self.latest_frame = color_data  # No need for copy here
                        self.new_frame_event.set()

                    finally:
                        self.listener.release(frames)

            except DeviceNotConnectedError:
                logger.info(f"{self} disconnected, stopping read loop")
                break
            except Exception as e:
                logger.warning(f"Error in Kinect read loop for {self}: {e}")
                time.sleep(0.1)  # Prevent busy-looping on repeated errors

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        # If thread is already running, nothing to do
        if self.thread is not None and self.thread.is_alive():
            return

        # Clean up any previous thread
        if self.thread is not None:
            self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

        # Give thread time to start capturing
        time.sleep(0.1)

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame asynchronously.

        When use_depth=True, returns colorized depth. Otherwise returns color frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The latest captured color frame in RGB format.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within timeout.
            RuntimeError: If the background thread died unexpectedly.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # Don't wait if we already have a frame - return immediately
        # This is the key optimization for parallel reading
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame
                # Don't clear the event here - other threads might be waiting
                return frame

        # Only wait if no frame is available yet
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms}ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            # Return color frame
            frame = self.latest_frame
            self.new_frame_event.clear()  # Clear event after reading, matching RealSense pattern

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self, suppress_errors: bool = False):
        """
        Disconnects from the camera and releases all resources.
        
        Args:
            suppress_errors: If True, suppress disconnect errors (useful for keyboard interrupts).

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (unless suppress_errors=True).
        """
        if not self.is_connected and self.thread is None:
            if not suppress_errors:
                raise DeviceNotConnectedError(
                    f"Attempted to disconnect {self}, but it appears already disconnected."
                )
            return

        # Stop the read thread first
        if self.thread is not None:
            self._stop_read_thread()

        # Clear any pending frames before stopping device
        if self.listener is not None and self.device is not None:
            try:
                frames = FrameMap()
                if self.listener.waitForNewFrame(frames, 1):  # 1ms timeout
                    self.listener.release(frames)
            except Exception:
                pass  # Ignore errors during buffer clear

        # Stop and close device in proper order
        if self.device is not None:
            try:
                self.device.stop()
                time.sleep(1.0)  # Small delay to ensure stop completes
                self.device.close()
            except Exception as e:
                if not suppress_errors:
                    logger.debug(f"Error during {self} disconnect: {e}")
            finally:
                self.device = None

        # Clean up listener and other resources
        self.listener = None
        self.registration = None
        self.pipeline = None
        self._frame_map = None
        
        # Clean up Freenect2 instance - MUST be last and wait for USB to stabilize
        if self.fn2 is not None:
            try:
                del self.fn2
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.fn2 = None
                gc.collect()  # Force garbage collection
                time.sleep(1.0)  # Increased wait for USB device to be fully released on Windows

        logger.info(f"{self} disconnected.")
