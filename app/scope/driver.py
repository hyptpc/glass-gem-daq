from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import json
import re
import threading

import matplotlib.pyplot as plt
import numpy as np
import pyvisa


DEFAULT_TIMEOUT_MS = 30000


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "output"


@dataclass
class ScopeConfig:
    ip: str
    channel: str = "CH1"
    start_index: int = 1
    stop_index: int = 10000
    timeout_ms: int = DEFAULT_TIMEOUT_MS


class ScopeDriver:
    def __init__(self, config: ScopeConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._rm: pyvisa.ResourceManager | None = None
        self._scope: pyvisa.resources.MessageBasedResource | None = None

    @property
    def scope(self) -> pyvisa.resources.MessageBasedResource:
        if self._scope is None:
            self._connect()
        assert self._scope is not None
        return self._scope

    def _connect(self) -> None:
        resource = f"TCPIP0::{self.config.ip}::INSTR"
        self._rm = pyvisa.ResourceManager("@py")
        scope = self._rm.open_resource(resource)
        scope.timeout = self.config.timeout_ms
        scope.encoding = "ascii"
        scope.read_termination = "\n"
        scope.write_termination = "\n"
        self._scope = scope

    def get_idn(self) -> str:
        """Return *IDN? result (thread-safe)."""
        with self._lock:
            return self.scope.query("*IDN?").strip()

    def close(self) -> None:
        if self._scope is not None:
            try:
                self._scope.close()
            except Exception:
                pass
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
        self._scope = None
        self._rm = None

    def setup_waveform_transfer(
        self,
        channel: str | None = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        with self._lock:
            ch = channel or self.config.channel
            start_idx = start if start is not None else self.config.start_index
            stop_idx = stop if stop is not None else self.config.stop_index

            scope = self.scope
            scope.write("HEADER 0")
            scope.write(f"DATA:SOURCE {ch}")
            scope.write(f"DATA:START {start_idx}")
            scope.write(f"DATA:STOP {stop_idx}")
            scope.write("DATA:ENCDG RIBINARY")
            scope.write("DATA:WIDTH 1")

    def acquire_waveform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            scope = self.scope

            raw = scope.query_binary_values(
                "CURVE?",
                datatype="b",
                container=np.array,
            )

            ymult = float(scope.query("WFMOUTPRE:YMULT?"))
            yzero = float(scope.query("WFMOUTPRE:YZERO?"))
            yoff = float(scope.query("WFMOUTPRE:YOFF?"))

            xincr = float(scope.query("WFMOUTPRE:XINCR?"))
            xzero = float(scope.query("WFMOUTPRE:XZERO?"))
            ptoff = float(scope.query("WFMOUTPRE:PT_OFF?"))

            volts = (raw - yoff) * ymult + yzero
            time_s = (np.arange(len(raw)) - ptoff) * xincr + xzero

            return time_s, volts, raw

    def get_basic_metadata(self, channel: str | None = None) -> Dict[str, Any]:
        with self._lock:
            scope = self.scope
            ch = channel or self.config.channel

            idn = scope.query("*IDN?").strip()
            try:
                record_length = scope.query("HORIZONTAL:RECORDLENGTH?").strip()
            except Exception:
                record_length = "unknown"

            try:
                vscale = scope.query(f"{ch}:SCALE?").strip()
            except Exception:
                vscale = "unknown"

            try:
                hscale = scope.query("HORIZONTAL:SCALE?").strip()
            except Exception:
                hscale = "unknown"

            return {
                "idn": idn,
                "channel": ch,
                "record_length": record_length,
                "vscale": vscale,
                "hscale": hscale,
            }

    def get_trigger_state(self) -> Dict[str, Any]:
        with self._lock:
            scope = self.scope
            state: Dict[str, Any] = {}
            try:
                state["mode"] = scope.query("TRIGGER:A:MODE?").strip()
            except Exception:
                state["mode"] = "unknown"
            try:
                state["source"] = scope.query("TRIGGER:A:EDGE:SOURCE?").strip()
            except Exception:
                state["source"] = "unknown"
            try:
                state["level"] = float(scope.query("TRIGGER:A:LEVEL?"))
            except Exception:
                state["level"] = None
            return state

    def set_trigger_source(self, source: str) -> None:
        with self._lock:
            self.scope.write(f"TRIGGER:A:EDGE:SOURCE {source}")

    def set_trigger_mode(self, mode: str) -> None:
        with self._lock:
            self.scope.write(f"TRIGGER:A:MODE {mode}")

    def set_trigger_level(self, level: float) -> None:
        with self._lock:
            self.scope.write(f"TRIGGER:A:LEVEL {level}")


def save_csv(path: Path, time_s: np.ndarray, volts: np.ndarray) -> None:
    data = np.column_stack([time_s, volts])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time_s,voltage_V",
        comments="",
    )


def save_plot(path: Path, time_s: np.ndarray, volts: np.ndarray, meta: Dict[str, Any]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(time_s, volts, linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"{meta['idn']}  {meta['channel']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_capture_paths(outdir: Path, channel: str) -> Dict[str, Path]:
    csv_dir = outdir / "csv"
    meta_dir = outdir / "meta"
    for d in (csv_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Use sub-second timestamp to avoid overwriting when saving
    # multiple captures within the same second.
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = sanitize_filename(f"{channel}_{now}")
    return {
        "csv": csv_dir / f"{base}.csv",
        "json": meta_dir / f"{base}.json",
    }

