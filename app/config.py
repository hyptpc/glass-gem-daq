from __future__ import annotations

from pathlib import Path
import os


# Required: oscilloscope IP address
OSC_IP = os.environ.get("OSC_IP")
if not OSC_IP:
    raise RuntimeError("OSC_IP environment variable is not set.")

# Optional: channel, record length, etc. use code defaults if not set
DEFAULT_CHANNEL = os.environ.get("OSC_CHANNEL", "CH1")
START_INDEX = int(os.environ.get("OSC_START_INDEX", "1"))
STOP_INDEX = int(os.environ.get("OSC_STOP_INDEX", "10000"))
TIMEOUT_MS = int(os.environ.get("OSC_TIMEOUT_MS", "30000"))

# Output directory for saved waveforms (default: outputs/)
OUTDIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

# Optional: MJPEG camera stream URL (e.g. Raspberry Pi mjpg-streamer). Unset = camera disabled.
CAMERA_STREAM_URL = os.environ.get("CAMERA_STREAM_URL") or None
