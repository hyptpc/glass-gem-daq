from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict

import json
import re
import threading
import time

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
    data_window_mode: Literal["fixed", "record"] = "record"


class WfmOutPre(TypedDict):
    """Tek WFMOUTPRE fields for the waveform last transferred with CURVE?."""

    xincr: float
    xzero: float
    pt_off: float
    ymult: float
    yzero: float
    yoff: float
    data_start: int
    data_stop: int


class ScopeDriver:
    def __init__(self, config: ScopeConfig) -> None:
        self.config = config
        # Reentrant: acquire_waveform_multi_after_trigger holds the lock across RUN/wait/CURVE.
        self._lock = threading.RLock()
        self._rm: pyvisa.ResourceManager | None = None
        self._scope: pyvisa.resources.MessageBasedResource | None = None
        # Last DATA:START / DATA:STOP used by setup_waveform_transfer (for time axis).
        self._last_data_start: int = config.start_index
        self._last_data_stop: int = config.stop_index
        # Saved when starting triggered capture session (restore on end).
        self._saved_acquire_stopafter: str | None = None

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

    def _query_horizontal_record_length_unlocked(self) -> Optional[int]:
        """Parse HORIZONTAL:RECORDLENGTH? (caller must hold self._lock)."""
        try:
            raw = self.scope.query("HORIZONTAL:RECORDLENGTH?").strip()
            n = int(float(raw))
            if n > 0:
                return n
        except Exception:
            pass
        return None

    def setup_waveform_transfer(
        self,
        channel: str | None = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        with self._lock:
            ch = channel or self.config.channel
            explicit = start is not None or stop is not None
            if explicit:
                start_idx = (
                    start if start is not None else self.config.start_index
                )
                stop_idx = stop if stop is not None else self.config.stop_index
            elif self.config.data_window_mode == "record":
                start_idx = 1
                rl = self._query_horizontal_record_length_unlocked()
                stop_idx = rl if rl is not None else self.config.stop_index
            else:
                start_idx = self.config.start_index
                stop_idx = self.config.stop_index

            scope = self.scope
            scope.write("HEADER 0")
            scope.write(f"DATA:SOURCE {ch}")
            scope.write(f"DATA:START {start_idx}")
            scope.write(f"DATA:STOP {stop_idx}")
            scope.write("DATA:ENCDG RIBINARY")
            scope.write("DATA:WIDTH 1")
            self._last_data_start = start_idx
            self._last_data_stop = stop_idx

    def query_wfmoutpre_preamble(self) -> str:
        """
        Send ``WFMOUTPRE?`` and return the full comma-separated preamble string.

        Tek updates this for the waveform last transferred with ``CURVE?``; if no recent
        transfer, values may be stale. See programmer manual for field order / meaning.
        """
        with self._lock:
            return self.scope.query("WFMOUTPRE?").strip()

    def _read_wfmoutpre(self) -> Tuple[float, float, float, float, float, float]:
        """Read individual WFMOUTPRE fields after CURVE? (must be same lock / same transfer).

        For the entire preamble in one response, use ``query_wfmoutpre_preamble()`` (``WFMOUTPRE?``).
        """
        scope = self.scope
        ymult = float(scope.query("WFMOUTPRE:YMULT?"))
        yzero = float(scope.query("WFMOUTPRE:YZERO?"))
        yoff = float(scope.query("WFMOUTPRE:YOFF?"))
        xincr = float(scope.query("WFMOUTPRE:XINCR?"))
        xzero = float(scope.query("WFMOUTPRE:XZERO?"))
        ptoff = float(scope.query("WFMOUTPRE:PT_OFF?"))
        return ymult, yzero, yoff, xincr, xzero, ptoff

    def acquire_waveform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, WfmOutPre]:
        with self._lock:
            scope = self.scope
            start_idx = self._last_data_start
            stop_idx = self._last_data_stop

            raw = scope.query_binary_values(
                "CURVE?",
                datatype="b",
                container=np.array,
            )

            ymult, yzero, yoff, xincr, xzero, ptoff = self._read_wfmoutpre()

            volts = (raw - yoff) * ymult + yzero
            # Point i (0-based) in the transferred buffer is record point (start_idx + i)
            # in Tek's horizontal record; PT_OFF is the trigger index in that record.
            idx = np.arange(len(raw), dtype=np.float64)
            time_s = (idx + float(start_idx - 1) - ptoff) * xincr + xzero

            preamble: WfmOutPre = {
                "xincr": xincr,
                "xzero": xzero,
                "pt_off": ptoff,
                "ymult": ymult,
                "yzero": yzero,
                "yoff": yoff,
                "data_start": start_idx,
                "data_stop": stop_idx,
            }

            return time_s, volts, raw, preamble

    def begin_triggered_capture_session(self) -> None:
        """
        Configure single-sequence acquisition so each RUN completes after one trigger.

        Saves current ACQUIRE:STOPAFTER? and sets SEQUENCE; call
        end_triggered_capture_session() when done.
        """
        with self._lock:
            scope = self.scope
            self._saved_acquire_stopafter = None
            try:
                self._saved_acquire_stopafter = scope.query("ACQUIRE:STOPAFTER?").strip()
            except Exception:
                pass
            try:
                scope.write("ACQUIRE:STOPAFTER SEQUENCE")
            except Exception:
                pass

    def end_triggered_capture_session(self) -> None:
        """Restore ACQUIRE:STOPAFTER from before begin_triggered_capture_session()."""
        with self._lock:
            if self._saved_acquire_stopafter:
                try:
                    self.scope.write(f"ACQUIRE:STOPAFTER {self._saved_acquire_stopafter}")
                except Exception:
                    pass
            self._saved_acquire_stopafter = None

    def _wait_acquisition_complete_unlocked(self, timeout_s: float) -> None:
        """
        After ACQUIRE:STATE RUN, wait until one acquisition finishes (poll ACQUIRE:STATE?).

        Caller must hold self._lock. Raises TimeoutError if no completion before timeout
        (e.g. trigger never occurs).         If this always times out on your scope, set env ``OSC_TRIGGER_CAPTURE_STRICT=0``
        (default) so triggered capture reads the buffer without waiting for STOP.
        """
        scope = self.scope
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                st = scope.query("ACQUIRE:STATE?").strip().upper()
            except Exception:
                time.sleep(0.005)
                continue
            if st in ("STOP", "STOPPED"):
                return
            time.sleep(0.002)
        raise TimeoutError(
            f"acquisition did not complete within {timeout_s}s (no trigger or scope busy)"
        )

    def acquire_waveform_multi_after_trigger(
        self,
        channels: list[str],
        acquisition_timeout_s: float = 30.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, WfmOutPre]]:
        """
        Arm one acquisition, wait for it to complete (trigger + record), then CURVE? per channel.

        Use this for trigger-driven file saves so each file corresponds to a completed
        acquisition, not repeated reads of the same waveform buffer. RUN/wait/CURVE stay
        under one lock so the buffer cannot change between wait and transfer.
        """
        with self._lock:
            self.scope.write("ACQUIRE:STATE RUN")
            # Let RUN latch before polling so we do not treat pre-RUN idle STOP as "done".
            time.sleep(0.01)
            self._wait_acquisition_complete_unlocked(acquisition_timeout_s)
            results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, WfmOutPre]] = {}
            for ch in channels:
                self.setup_waveform_transfer(channel=ch)
                time_s, volts, raw, preamble = self.acquire_waveform()
                results[ch] = (time_s, volts, raw, preamble)
            return results

    def acquire_waveform_multi(
        self, channels: list[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, WfmOutPre]]:
        """
        Acquire waveforms for multiple channels in one triggered acquisition.

        For each channel in `channels`, this method:
        - configures DATA:SOURCE / DATA:START / DATA:STOP
        - reads CURVE? and waveform preamble

        The returned dict maps channel name to (time_s, volts, raw, wfmoutpre).

        The time axis is computed per channel from that channel's WFMOUTPRE; with a
        shared horizontal scale the arrays should match across channels.
        """
        results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, WfmOutPre]] = {}
        for ch in channels:
            # Reuse existing single-channel helpers to keep locking and
            # configuration consistent.
            self.setup_waveform_transfer(channel=ch)
            time_s, volts, raw, preamble = self.acquire_waveform()
            results[ch] = (time_s, volts, raw, preamble)
        return results

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

    def get_acquire_state(self) -> str:
        """Return ``ACQUIRE:STATE?`` (e.g. RUN/STOP/STOPPED)."""
        with self._lock:
            # Tek scopes sometimes use both ACQ:STATE and ACQUIRE:STATE aliases.
            for cmd in ("ACQ:STATE?", "ACQUIRE:STATE?"):
                try:
                    return self.scope.query(cmd).strip().upper()
                except Exception:
                    continue
            # Fallback (will raise upstream if actually used)
            return self.scope.query("ACQ:STATE?").strip().upper()

    def set_acquire_state(self, state: str) -> None:
        """Set acquisition state (e.g. RUN or STOP)."""
        st = str(state).strip().upper()
        with self._lock:
            for cmd in ("ACQ:STATE", "ACQUIRE:STATE"):
                try:
                    self.scope.write(f"{cmd} {st}")
                    return
                except Exception:
                    continue
            raise RuntimeError(f"failed to set acquire state to {st}")

    def set_stopa_runstop(self) -> None:
        """Set Tek Stop-A mode to RUNSTOP (debug.py pattern: ACQ:STOPA RUNSTOP)."""
        with self._lock:
            # best-effort; if unsupported, let caller proceed
            self.scope.write("ACQ:STOPA RUNSTOP")

    def get_horizontal_scale_s(self) -> float:
        """Horizontal scale in seconds per division (``HORIZONTAL:SCALE``)."""
        with self._lock:
            return float(self.scope.query("HORIZONTAL:SCALE?").strip())

    def set_horizontal_scale_s(self, scale_s_per_div: float) -> None:
        """Set horizontal scale in seconds per division."""
        with self._lock:
            self.scope.write(f"HORIZONTAL:SCALE {scale_s_per_div}")

    def get_horizontal_position_s(self) -> float:
        """Horizontal position in seconds (``HORIZONTAL:POSITION``).

        Tek scopes define it as the time between the Time Zero (first trigger point)
        and the Horizontal Reference Point.
        """
        with self._lock:
            return float(self.scope.query("HORIZONTAL:POSITION?").strip())

    def set_horizontal_position_s(self, position_s: float) -> None:
        """Set horizontal position in seconds (``HORIZONTAL:POSITION``)."""
        with self._lock:
            self.scope.write(f"HORIZONTAL:POSITION {position_s}")

    def get_horizontal_delay_s(self) -> float:
        """Horizontal delay in seconds (``HORIZONTAL:DELAY``)."""
        with self._lock:
            return float(self.scope.query("HORIZONTAL:DELAY?").strip())

    def set_horizontal_delay_s(self, delay_s: float) -> None:
        """Set horizontal delay in seconds (``HORIZONTAL:DELAY``)."""
        with self._lock:
            self.scope.write(f"HORIZONTAL:DELAY {delay_s}")

    def set_trigger_left_fraction(
        self,
        *,
        left_fraction: float,
        scale_s_per_div: float,
        reference_fraction: float = 0.5,
        total_divisions: float = 10.0,
    ) -> None:
        """Move trigger (Time Zero) to a target horizontal screen fraction.

        We map "left_fraction" (0..1) to Horizontal Position time value:
        - Horizontal Position = time between Time Zero (trigger) and Horizontal Reference Point.
        - Assume Horizontal Reference Point is at `reference_fraction` of the screen width.
        - Assume screen width spans `total_divisions` divisions (default 10).
        """
        lf = float(left_fraction)
        if not (0.0 <= lf <= 1.0):
            raise ValueError("left_fraction must be within [0, 1]")

        total_width_s = float(total_divisions) * float(scale_s_per_div)

        # Mapping:
        # - We want Time Zero (trigger) to be at `left_fraction` of the screen width from
        #   the left edge.
        # - Tek FAQ defines Horizontal Delay as the time between Time Zero and the first
        #   sample point.
        # - If Time Zero is at +x seconds from the left edge, then the first sample is at
        #   time -x relative to Time Zero, so delay = (first_sample - time_zero) ~= -x.
        # - With x = left_fraction * width, delay_s = -left_fraction * width.
        delay_s = -lf * total_width_s

        # Desired horizontal reference is at reference_fraction on the screen.
        # Horizontal Position is the time between Time Zero and the Horizontal Reference Point.
        # Use it as a secondary adjustment.
        diff_div = (reference_fraction - lf) * float(total_divisions)
        pos_s = diff_div * float(scale_s_per_div)

        # Primary: set delay (most likely to move trigger point on-screen)
        def is_close(a: float, b: float) -> bool:
            return abs(a - b) <= max(1e-12, abs(b) * 1e-6)

        # Tek front-panel style shorthand: HOR:POS <percent>
        # If the user's instrument expects HOR:POS in percent (e.g. HOR:POS 20),
        # then left_fraction=0.2 maps directly to 20.
        hor_pos_percent = lf * 100.0

        # Primary (user-proven): HOR:POS
        try:
            # Follow the same pattern as debug.py:
            # - disable delay mode explicitly
            # - ensure acquisition is safe while changing horizontal controls
            try:
                self.scope.write("HOR:DEL:MOD OFF")
            except Exception:
                pass

            self.scope.write(f"HOR:POS {hor_pos_percent}")
            # Best-effort readback for instruments supporting it.
            try:
                actual = float(self.scope.query("HOR:POS?").strip())
                if abs(actual - hor_pos_percent) <= max(1e-9, abs(hor_pos_percent) * 1e-6):
                    return
            except Exception:
                # If readback is unsupported, still assume the write worked.
                return
        except Exception:
            pass

        delay_candidates = [delay_s, -delay_s]
        delay_set = False
        for candidate in delay_candidates:
            try:
                self.set_horizontal_delay_s(candidate)
                actual = self.get_horizontal_delay_s()
                if is_close(actual, candidate):
                    delay_set = True
                    break
            except Exception:
                continue

        # Secondary: set horizontal position (for scopes that use that knob)
        try:
            self.set_horizontal_position_s(pos_s)
            return
        except Exception:
            pass

        # Fallback: some scopes expose trigger position directly.
        # Try both seconds and divisions-ish values.
        try:
            self.scope.write(f"TRIGGER:A:POSITION {pos_s}")
            return
        except Exception:
            pass
        try:
            self.scope.write(f"TRIGGER:A:POSITION {diff_div}")
            return
        except Exception:
            # If everything fails, let it pass silently (caller can read-back if desired).
            return

    def get_channel_vertical(self, channel: str) -> Dict[str, float]:
        """Per-channel volts/div and position in divisions (``CHx:SCALE``, ``CHx:POSITION``)."""
        with self._lock:
            scale_v = float(self.scope.query(f"{channel}:SCALE?").strip())
            pos_div = float(self.scope.query(f"{channel}:POSITION?").strip())
        return {"scale_v_per_div": scale_v, "position_div": pos_div}

    def set_channel_vertical(
        self,
        channel: str,
        *,
        scale_v_per_div: Optional[float] = None,
        position_div: Optional[float] = None,
    ) -> None:
        """Set channel vertical scale and/or position (only passed keys are written)."""
        with self._lock:
            scope = self.scope
            if scale_v_per_div is not None:
                scope.write(f"{channel}:SCALE {scale_v_per_div}")
            if position_div is not None:
                scope.write(f"{channel}:POSITION {position_div}")

    def get_scope_display_state(self, channels: Optional[list[str]] = None) -> Dict[str, Any]:
        """Current horizontal scale and vertical settings for each analog channel."""
        chs = channels or ["CH1", "CH2", "CH3", "CH4"]
        with self._lock:
            h = float(self.scope.query("HORIZONTAL:SCALE?").strip())
            out: Dict[str, Any] = {"horizontal_scale_s_per_div": h, "channels": {}}
            for ch in chs:
                try:
                    out["channels"][ch] = {
                        "scale_v_per_div": float(self.scope.query(f"{ch}:SCALE?").strip()),
                        "position_div": float(self.scope.query(f"{ch}:POSITION?").strip()),
                    }
                except Exception:
                    out["channels"][ch] = {
                        "scale_v_per_div": None,
                        "position_div": None,
                    }
        return out


def save_csv(path: Path, time_s: np.ndarray, volts: np.ndarray) -> None:
    data = np.column_stack([time_s, volts])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="time_s,voltage_V",
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
    """Return paths for one capture: outdir/CHx/csv/<timestamp>.csv and outdir/CHx/json/<timestamp>.json."""
    ch_dir = outdir / sanitize_filename(channel)
    csv_dir = ch_dir / "csv"
    json_dir = ch_dir / "json"
    for d in (csv_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = sanitize_filename(now)
    return {
        "csv": csv_dir / f"{base}.csv",
        "json": json_dir / f"{base}.json",
    }

