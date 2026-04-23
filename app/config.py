from __future__ import annotations

from pathlib import Path
import os
from typing import Literal


# Required: oscilloscope IP address
OSC_IP = os.environ.get("OSC_IP")
if not OSC_IP:
    raise RuntimeError("OSC_IP environment variable is not set.")

# Optional: channel, record length, etc. use code defaults if not set
DEFAULT_CHANNEL = os.environ.get("OSC_CHANNEL", "CH1")
START_INDEX = int(os.environ.get("OSC_START_INDEX", "1"))
STOP_INDEX = int(os.environ.get("OSC_STOP_INDEX", "10000"))
# fixed: DATA:START/STOP from OSC_* only. record: DATA:START=1, DATA:STOP=HORIZONTAL:RECORDLENGTH?
_dw = os.environ.get("OSC_DATA_WINDOW", "record").strip().lower()
DATA_WINDOW_MODE: Literal["fixed", "record"] = (
    _dw if _dw in ("fixed", "record") else "record"
)
TIMEOUT_MS = int(os.environ.get("OSC_TIMEOUT_MS", "30000"))
# Trigger-driven capture: max seconds to wait for one acquisition (trigger + record) per save.
TRIGGER_CAPTURE_ACQUISITION_TIMEOUT_S = float(
    os.environ.get("OSC_TRIGGER_CAPTURE_TIMEOUT_S", "30")
)

# Output directory for saved waveforms (default: outputs/)
OUTDIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

# Optional: MJPEG camera stream URL (e.g. Raspberry Pi mjpg-streamer). Unset = camera disabled.
CAMERA_STREAM_URL = os.environ.get("CAMERA_STREAM_URL") or None

# Optional: built-in oscilloscope web app port (default: 81).
OSC_WEBAPP_PORT = int(os.environ.get("OSC_WEBAPP_PORT", "81"))
