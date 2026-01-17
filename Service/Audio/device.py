"""
Audio Device Management Module
===============================
Comprehensive audio device detection, validation, and management.

Features:
- Device listing and filtering
- Device validation with detailed checks
- Auto-selection with fallback chain
- Device capability detection
- Sample rate and channel validation
- Device monitoring and health checks
- Cross-platform compatibility

Usage:
    from Asr.device import (
        list_devices,
        get_default_input_device,
        validate_device,
        DeviceSelector,
    )
    
    # List all devices
    devices = list_devices(input_only=True)
    
    # Get default device
    device = get_default_input_device()
    
    # Validate device
    validate_device(device_id=0, sample_rate=16000, channels=1)
"""

import logging
import platform
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

import sounddevice as sd
import numpy as np

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class DeviceConfig:
    """Device management configuration."""
    DEFAULT_CHANNELS: int = 1
    DEFAULT_SAMPLE_RATE: int = 16000
    
    # Supported sample rates (in priority order)
    SUPPORTED_SAMPLE_RATES: List[int] = [16000, 44100, 48000, 22050, 8000, 32000]
    
    # Validation
    MIN_CHANNELS: int = 1
    MAX_CHANNELS: int = 32
    
    # Testing
    TEST_DURATION: float = 0.1  # seconds
    TEST_ENABLED: bool = True


# ============================================================================
# Device Types
# ============================================================================

class DeviceType(Enum):
    """Audio device types."""
    INPUT = "input"
    OUTPUT = "output"
    DUPLEX = "duplex"  # Both input and output


@dataclass
class DeviceInfo:
    """Comprehensive device information."""
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    
    # Optional capabilities
    is_default_input: bool = False
    is_default_output: bool = False
    supported_sample_rates: List[int] = None
    
    def __post_init__(self):
        if self.supported_sample_rates is None:
            self.supported_sample_rates = []
    
    @property
    def device_type(self) -> DeviceType:
        """Determine device type."""
        has_input = self.max_input_channels > 0
        has_output = self.max_output_channels > 0
        
        if has_input and has_output:
            return DeviceType.DUPLEX
        elif has_input:
            return DeviceType.INPUT
        elif has_output:
            return DeviceType.OUTPUT
        else:
            raise ValueError("Device has no input or output channels")
    
    @property
    def is_input_capable(self) -> bool:
        """Check if device supports input."""
        return self.max_input_channels > 0
    
    @property
    def is_output_capable(self) -> bool:
        """Check if device supports output."""
        return self.max_output_channels > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "hostapi": self.hostapi,
            "max_input_channels": self.max_input_channels,
            "max_output_channels": self.max_output_channels,
            "default_sample_rate": self.default_sample_rate,
            "device_type": self.device_type.value,
            "is_default_input": self.is_default_input,
            "is_default_output": self.is_default_output,
            "supported_sample_rates": self.supported_sample_rates,
        }
    
    def __str__(self) -> str:
        """String representation."""
        type_str = self.device_type.value
        channels_str = f"in:{self.max_input_channels}" if self.is_input_capable else ""
        if self.is_output_capable:
            channels_str += f" out:{self.max_output_channels}" if channels_str else f"out:{self.max_output_channels}"
        
        default_str = ""
        if self.is_default_input:
            default_str = " [DEFAULT INPUT]"
        elif self.is_default_output:
            default_str = " [DEFAULT OUTPUT]"
        
        return f"[{self.index}] {self.name} ({type_str}, {channels_str}){default_str}"


# ============================================================================
# Device Discovery
# ============================================================================

def list_devices(
    input_only: bool = False,
    output_only: bool = False,
    include_defaults: bool = True,
) -> List[DeviceInfo]:
    """
    List all available audio devices.
    
    Args:
        input_only: Only list input devices
        output_only: Only list output devices
        include_defaults: Mark default devices
    
    Returns:
        List of DeviceInfo objects
    
    Examples:
        >>> devices = list_devices(input_only=True)
        >>> for dev in devices:
        ...     print(dev)
    """
    try:
        raw_devices = sd.query_devices()
        
        # Get default devices
        default_input = None
        default_output = None
        
        if include_defaults:
            try:
                default_input = sd.default.device[0]
                default_output = sd.default.device[1]
            except (TypeError, IndexError):
                pass
        
        # Convert to DeviceInfo objects
        devices = []
        
        for i, dev in enumerate(raw_devices):
            max_in = int(dev.get('max_input_channels', 0))
            max_out = int(dev.get('max_output_channels', 0))
            
            # Filter by type
            if input_only and max_in == 0:
                continue
            if output_only and max_out == 0:
                continue
            
            # Get host API name
            try:
                hostapi_index = dev.get('hostapi', 0)
                hostapi_info = sd.query_hostapis(hostapi_index)
                hostapi_name = hostapi_info.get('name', 'Unknown')
            except Exception:
                hostapi_name = 'Unknown'
            
            device_info = DeviceInfo(
                index=i,
                name=dev.get('name', f'Device {i}'),
                hostapi=hostapi_name,
                max_input_channels=max_in,
                max_output_channels=max_out,
                default_sample_rate=float(dev.get('default_samplerate', 44100)),
                is_default_input=(i == default_input),
                is_default_output=(i == default_output),
            )
            
            devices.append(device_info)
        
        return devices
    
    except Exception as e:
        log.error(f"Failed to list devices: {e}")
        raise RuntimeError(f"Device enumeration failed: {e}")


def list_input_devices() -> List[DeviceInfo]:
    """List all input devices."""
    return list_devices(input_only=True)


def list_output_devices() -> List[DeviceInfo]:
    """List all output devices."""
    return list_devices(output_only=True)


def print_devices(devices: Optional[List[DeviceInfo]] = None):
    """
    Print device list to console.
    
    Args:
        devices: List of devices (None = list all)
    """
    if devices is None:
        devices = list_devices()
    
    if not devices:
        print("No audio devices found")
        return
    
    print(f"\n{'='*80}")
    print(f"Audio Devices ({len(devices)} found)")
    print(f"{'='*80}")
    
    for dev in devices:
        print(dev)
    
    print(f"{'='*80}\n")


# ============================================================================
# Device Validation
# ============================================================================

class DeviceValidationError(Exception):
    """Device validation error."""
    pass


def validate_device(
    device_id: Optional[int] = None,
    input: bool = True,
    channels: Optional[int] = None,
    sample_rate: Optional[int] = None,
    test_recording: bool = False,
) -> DeviceInfo:
    """
    Validate device and check capabilities.
    
    Args:
        device_id: Device index (None = default)
        input: True for input device, False for output
        channels: Required number of channels
        sample_rate: Required sample rate
        test_recording: Actually test recording/playback
    
    Returns:
        DeviceInfo object
    
    Raises:
        DeviceValidationError: If validation fails
    
    Examples:
        >>> device = validate_device(device_id=0, channels=1, sample_rate=16000)
        >>> print(f"Using: {device.name}")
    """
    # Resolve device ID
    if device_id is None:
        try:
            defaults = sd.default.device
            device_id = defaults[0] if input else defaults[1]
            
            if device_id is None or device_id < 0:
                raise DeviceValidationError("No default device available")
            
            log.debug(f"Using default {'input' if input else 'output'} device: {device_id}")
        
        except Exception as e:
            raise DeviceValidationError(f"Failed to get default device: {e}")
    
    # Query device
    try:
        raw_info = sd.query_devices(device_id)
    except Exception as e:
        raise DeviceValidationError(f"Device {device_id} not found: {e}")
    
    # Build DeviceInfo
    max_in = int(raw_info.get('max_input_channels', 0))
    max_out = int(raw_info.get('max_output_channels', 0))
    
    try:
        hostapi_index = raw_info.get('hostapi', 0)
        hostapi_info = sd.query_hostapis(hostapi_index)
        hostapi_name = hostapi_info.get('name', 'Unknown')
    except Exception:
        hostapi_name = 'Unknown'
    
    device_info = DeviceInfo(
        index=device_id,
        name=raw_info.get('name', f'Device {device_id}'),
        hostapi=hostapi_name,
        max_input_channels=max_in,
        max_output_channels=max_out,
        default_sample_rate=float(raw_info.get('default_samplerate', 44100)),
    )
    
    # Validate direction
    if input and not device_info.is_input_capable:
        raise DeviceValidationError(
            f"Device '{device_info.name}' has no input channels"
        )
    
    if not input and not device_info.is_output_capable:
        raise DeviceValidationError(
            f"Device '{device_info.name}' has no output channels"
        )
    
    # Validate channels
    if channels is not None:
        max_ch = max_in if input else max_out
        
        if channels < DeviceConfig.MIN_CHANNELS:
            raise DeviceValidationError(
                f"Channels ({channels}) must be >= {DeviceConfig.MIN_CHANNELS}"
            )
        
        if channels > max_ch:
            raise DeviceValidationError(
                f"Device '{device_info.name}' supports max {max_ch} channels, "
                f"requested {channels}"
            )
    
    # Validate sample rate
    if sample_rate is not None:
        try:
            if input:
                sd.check_input_settings(
                    device=device_id,
                    channels=channels or 1,
                    samplerate=sample_rate,
                )
            else:
                sd.check_output_settings(
                    device=device_id,
                    channels=channels or 1,
                    samplerate=sample_rate,
                )
        except Exception as e:
            raise DeviceValidationError(
                f"Device '{device_info.name}' does not support "
                f"{sample_rate}Hz @ {channels or 1}ch: {e}"
            )
    
    # Test recording/playback
    if test_recording and DeviceConfig.TEST_ENABLED:
        try:
            _test_device(
                device_id=device_id,
                input=input,
                channels=channels or 1,
                sample_rate=sample_rate or int(device_info.default_sample_rate),
            )
            log.info(f"Device test passed: {device_info.name}")
        except Exception as e:
            raise DeviceValidationError(f"Device test failed: {e}")
    
    log.info(f"Device validated: {device_info}")
    return device_info


def _test_device(
    device_id: int,
    input: bool,
    channels: int,
    sample_rate: int,
):
    """Test device by recording/playing a short clip."""
    duration = DeviceConfig.TEST_DURATION
    
    if input:
        # Test recording
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device_id,
            dtype='float32',
        )
        sd.wait()
        
        # Validate recording
        if recording.size == 0:
            raise RuntimeError("Recording is empty")
        
        # Check for data
        if np.max(np.abs(recording)) < 1e-6:
            log.warning("Recording appears to be silent")
    
    else:
        # Test playback
        test_signal = np.random.randn(
            int(duration * sample_rate),
            channels
        ).astype('float32') * 0.1
        
        sd.play(
            test_signal,
            samplerate=sample_rate,
            device=device_id,
        )
        sd.wait()


# ============================================================================
# Device Selection
# ============================================================================

def get_default_input_device() -> DeviceInfo:
    """
    Get default input device.
    
    Returns:
        DeviceInfo for default input device
    
    Raises:
        DeviceValidationError: If no default device
    """
    return validate_device(input=True)


def get_default_output_device() -> DeviceInfo:
    """Get default output device."""
    return validate_device(input=False)


def pick_input_device(
    explicit_index: int = -1,
    channels: int = DeviceConfig.DEFAULT_CHANNELS,
    sample_rate: int = DeviceConfig.DEFAULT_SAMPLE_RATE,
) -> Tuple[int, int]:
    """
    Pick input device with fallback chain.
    
    Args:
        explicit_index: Explicit device index (-1 = auto)
        channels: Required channels
        sample_rate: Required sample rate
    
    Returns:
        (device_index, actual_channels) tuple
    
    Raises:
        RuntimeError: If no suitable device found
    """
    # Try explicit device
    if explicit_index >= 0:
        try:
            device = validate_device(
                device_id=explicit_index,
                input=True,
                channels=channels,
                sample_rate=sample_rate,
            )
            actual_channels = min(channels, device.max_input_channels)
            log.info(f"Using explicit device {explicit_index}: {device.name}")
            return explicit_index, actual_channels
        
        except DeviceValidationError as e:
            log.warning(f"Explicit device {explicit_index} validation failed: {e}")
            log.info("Falling back to default device")
    
    # Try default device
    try:
        defaults = sd.default.device
        default_in = defaults[0] if isinstance(defaults, (list, tuple)) else None
        
        if default_in is not None and default_in >= 0:
            device = validate_device(
                device_id=default_in,
                input=True,
                channels=channels,
                sample_rate=sample_rate,
            )
            actual_channels = min(channels, device.max_input_channels)
            log.info(f"Using default input device: {device.name}")
            return default_in, actual_channels
    
    except Exception as e:
        log.warning(f"Default device validation failed: {e}")
    
    # Auto-select first suitable device
    log.info("Auto-selecting first suitable input device...")
    
    devices = list_input_devices()
    
    for device in devices:
        try:
            validate_device(
                device_id=device.index,
                input=True,
                channels=channels,
                sample_rate=sample_rate,
            )
            
            actual_channels = min(channels, device.max_input_channels)
            log.info(f"Auto-selected device {device.index}: {device.name}")
            
            # Set as default
            try:
                defaults = sd.default.device
                out_dev = defaults[1] if isinstance(defaults, (list, tuple)) else None
                sd.default.device = (device.index, out_dev)
            except Exception:
                pass
            
            return device.index, actual_channels
        
        except DeviceValidationError:
            continue
    
    # No suitable device found
    raise RuntimeError(
        f"No suitable input device found "
        f"(required: {channels}ch @ {sample_rate}Hz)"
    )


# Backward compatibility
_pick_input_device = pick_input_device


# ============================================================================
# Device Selector (Interactive)
# ============================================================================

class DeviceSelector:
    """Interactive device selector."""
    
    @staticmethod
    def select_input_device(
        channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> DeviceInfo:
        """
        Interactively select input device.
        
        Args:
            channels: Required channels (None = any)
            sample_rate: Required sample rate (None = any)
        
        Returns:
            Selected DeviceInfo
        """
        devices = list_input_devices()
        
        if not devices:
            raise RuntimeError("No input devices found")
        
        # Filter by requirements
        if channels or sample_rate:
            suitable_devices = []
            for dev in devices:
                try:
                    validate_device(
                        device_id=dev.index,
                        input=True,
                        channels=channels,
                        sample_rate=sample_rate,
                    )
                    suitable_devices.append(dev)
                except DeviceValidationError:
                    continue
            
            if not suitable_devices:
                raise RuntimeError(
                    f"No devices support requirements "
                    f"(channels={channels}, sample_rate={sample_rate})"
                )
            
            devices = suitable_devices
        
        # Display options
        print("\n" + "="*80)
        print("Available Input Devices")
        print("="*80)
        
        for i, dev in enumerate(devices):
            print(f"{i}: {dev}")
        
        print("="*80)
        
        # Get selection
        while True:
            try:
                choice = input(f"\nSelect device (0-{len(devices)-1}): ").strip()
                idx = int(choice)
                
                if 0 <= idx < len(devices):
                    selected = devices[idx]
                    print(f"\nSelected: {selected.name}\n")
                    return selected
                else:
                    print(f"Invalid selection. Please enter 0-{len(devices)-1}")
            
            except ValueError:
                print("Please enter a valid integer")
            except KeyboardInterrupt:
                print("\nSelection cancelled")
                raise
    
    @staticmethod
    def select_output_device() -> DeviceInfo:
        """Interactively select output device."""
        devices = list_output_devices()
        
        if not devices:
            raise RuntimeError("No output devices found")
        
        print("\n" + "="*80)
        print("Available Output Devices")
        print("="*80)
        
        for i, dev in enumerate(devices):
            print(f"{i}: {dev}")
        
        print("="*80)
        
        while True:
            try:
                choice = input(f"\nSelect device (0-{len(devices)-1}): ").strip()
                idx = int(choice)
                
                if 0 <= idx < len(devices):
                    selected = devices[idx]
                    print(f"\nSelected: {selected.name}\n")
                    return selected
                else:
                    print(f"Invalid selection. Please enter 0-{len(devices)-1}")
            
            except ValueError:
                print("Please enter a valid integer")
            except KeyboardInterrupt:
                print("\nSelection cancelled")
                raise


# ============================================================================
# System Information
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive audio system information.
    
    Returns:
        Dictionary with system info
    """
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "sounddevice_version": sd.__version__,
        "numpy_version": np.__version__,
    }
    
    # Host APIs
    try:
        hostapis = []
        for i in range(sd.query_hostapis.__wrapped__.__code__.co_argcount or 10):
            try:
                api = sd.query_hostapis(i)
                hostapis.append({
                    "index": i,
                    "name": api.get('name', 'Unknown'),
                    "device_count": api.get('device_count', 0),
                })
            except Exception:
                break
        
        info["host_apis"] = hostapis
    except Exception as e:
        log.warning(f"Failed to query host APIs: {e}")
        info["host_apis"] = []
    
    # Devices
    try:
        devices = list_devices()
        info["device_count"] = len(devices)
        info["input_device_count"] = sum(1 for d in devices if d.is_input_capable)
        info["output_device_count"] = sum(1 for d in devices if d.is_output_capable)
    except Exception as e:
        log.warning(f"Failed to query devices: {e}")
        info["device_count"] = 0
    
    # Defaults
    try:
        defaults = sd.default.device
        info["default_input"] = defaults[0] if isinstance(defaults, (list, tuple)) else None
        info["default_output"] = defaults[1] if isinstance(defaults, (list, tuple)) else None
    except Exception:
        info["default_input"] = None
        info["default_output"] = None
    
    return info


def print_system_info():
    """Print system information to console."""
    info = get_system_info()
    
    print("\n" + "="*80)
    print("Audio System Information")
    print("="*80)
    print(f"Platform: {info['platform']} {info['platform_version']}")
    print(f"sounddevice: {info['sounddevice_version']}")
    print(f"numpy: {info['numpy_version']}")
    print(f"\nDevices: {info['device_count']} total "
          f"({info['input_device_count']} input, {info['output_device_count']} output)")
    
    if info['host_apis']:
        print(f"\nHost APIs:")
        for api in info['host_apis']:
            print(f"  [{api['index']}] {api['name']} ({api['device_count']} devices)")
    
    print("="*80 + "\n")


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Print system info
    print_system_info()
    
    # List devices
    print_devices()
    
    # Interactive selection if requested
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        try:
            device = DeviceSelector.select_input_device(
                channels=1,
                sample_rate=16000,
            )
            print(f"\nYou selected: {device.name} (index {device.index})")
        except KeyboardInterrupt:
            print("\nCancelled")
