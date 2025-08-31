# device.py
import sounddevice as sd
import logging

log = logging.getLogger("Device")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CHANNELS = 1  # global default, can be overridden by caller

# ===================== LIST DEVICES =====================
def list_input_devices():
    """
    List all available input (microphone) devices.
    """
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    log.info("Available input devices:")
    for i, dev in enumerate(input_devices):
        log.info(f"{i}: {dev['name']} - {dev['max_input_channels']} channels")
    return input_devices

def list_output_devices():
    """
    List all available output (speaker) devices.
    """
    devices = sd.query_devices()
    output_devices = [d for d in devices if d['max_output_channels'] > 0]
    log.info("Available output devices:")
    for i, dev in enumerate(output_devices):
        log.info(f"{i}: {dev['name']} - {dev['max_output_channels']} channels")
    return output_devices

# ===================== VALIDATE DEVICE =====================
def validate_device(device_id=None, input=True):
    """
    Validate that a device exists and can be used.
    Args:
        device_id: int or None (default: system default)
        input: True for input device, False for output device
    Returns:
        device info dict
    """
    if device_id is None:
        device_id = sd.default.device[0 if input else 1]

    try:
        device_info = sd.query_devices(device_id)
        if input and device_info['max_input_channels'] < 1:
            raise ValueError("Selected device has no input channels")
        if not input and device_info['max_output_channels'] < 1:
            raise ValueError("Selected device has no output channels")
        log.info(f"Using device {device_info['name']} (ID {device_id})")
        return device_info
    except Exception as e:
        log.error(f"Device validation failed: {e}")
        raise

# ===================== INPUT DEVICE PICKER =====================
def _pick_input_device(explicit_index: int = -1) -> tuple[int, int]:
    """
    Return (device_index, channels) for input.
    - If explicit_index is valid, use it.
    - Else use default input device.
    - Else auto-pick the first device with input channels.
    """
    try:
        devices = sd.query_devices()
        if explicit_index >= 0:
            info = sd.query_devices(explicit_index)
            max_ch = int(info.get("max_input_channels", 0))
            if max_ch <= 0:
                log.warning(f"Explicit device {explicit_index} has no input channels, falling back to default")
                explicit_index = -1  # fallback to default
            else:
                log.info(f"Using explicit input device {explicit_index} with {max_ch} channels")
                return explicit_index, min(CHANNELS, max_ch) or 1

        # check default input device
        cur_dev = sd.default.device
        default_in = cur_dev[0] if isinstance(cur_dev, (list, tuple)) else None

        if default_in is None or default_in < 0:
            # auto-pick first available input device
            for i, d in enumerate(devices):
                if int(d.get("max_input_channels", 0)) > 0:
                    out_dev = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
                    sd.default.device = (i, out_dev)
                    log.info(f"Auto-selected input device {i} with {d['max_input_channels']} channels")
                    return i, min(CHANNELS, int(d["max_input_channels"])) or 1

            raise RuntimeError("No input devices with channels > 0 were found.")

        # validate default input device
        info = sd.query_devices(default_in)
        max_ch = int(info.get("max_input_channels", 0))
        if max_ch <= 0:
            raise RuntimeError(f"Default input device {default_in} has no input channels")
        log.info(f"Using default input device {default_in} with {max_ch} channels")
        return default_in, min(CHANNELS, max_ch) or 1

    except Exception as e:
        log.error(f"Audio device selection failed: {e}")
        raise RuntimeError(f"Audio device selection failed: {e}")

# ===================== DEFAULT DEVICE =====================
def get_default_input_device():
    return validate_device(input=True)

def get_default_output_device():
    return validate_device(input=False)

# ===================== SELECT DEVICE =====================
def select_device(prompt="Select input device:"):
    devices = list_input_devices()
    while True:
        try:
            idx = int(input(f"{prompt} "))
            if 0 <= idx < len(devices):
                return devices[idx]
            else:
                print(f"Invalid selection, enter 0-{len(devices)-1}")
        except ValueError:
            print("Please enter a valid integer")
