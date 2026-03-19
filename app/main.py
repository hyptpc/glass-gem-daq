from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    DEFAULT_CHANNEL,
    OSC_IP,
    OUTDIR,
    START_INDEX,
    STOP_INDEX,
    TIMEOUT_MS,
)
from app.scope import (
    ScopeConfig,
    ScopeDriver,
    build_capture_paths,
    save_csv,
    save_json,
)

ALL_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]

scope_config = ScopeConfig(
    ip=OSC_IP,
    channel=DEFAULT_CHANNEL,
    start_index=START_INDEX,
    stop_index=STOP_INDEX,
    timeout_ms=TIMEOUT_MS,
)

driver = ScopeDriver(scope_config)
driver_lock = asyncio.Lock()

baseline_common: Optional[float] = None
histogram_values: Dict[str, List[float]] = defaultdict(list)
trigger_capture_task: Optional[asyncio.Task] = None
trigger_capture_active_run: Optional[str] = None
trigger_capture_channels: List[str] = []
trigger_capture_saved_count: int = 0
trigger_capture_started_at: Optional[datetime] = None
trigger_capture_last_saved_at: Optional[datetime] = None
trigger_capture_stop_requested: bool = False

app = FastAPI()

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    index_path = static_dir / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    driver.close()


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        async with driver_lock:
            idn = driver.get_idn()
        return {"status": "ok", "idn": idn}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)},
        )


@app.get("/info")
async def info() -> Dict[str, Any]:
    async with driver_lock:
        meta = driver.get_basic_metadata()
        trig = driver.get_trigger_state()
    return {"meta": meta, "trigger": trig}


@app.get("/trigger")
async def get_trigger() -> Dict[str, Any]:
    async with driver_lock:
        state = driver.get_trigger_state()
    return state


@app.post("/trigger/source")
async def set_trigger_source(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = str(payload.get("source", "CH1"))
    async with driver_lock:
        driver.set_trigger_source(source)
        state = driver.get_trigger_state()
    return state


@app.post("/trigger/mode")
async def set_trigger_mode(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(payload.get("mode", "AUTO"))
    async with driver_lock:
        driver.set_trigger_mode(mode)
        state = driver.get_trigger_state()
    return state


@app.post("/trigger/level")
async def set_trigger_level(payload: Dict[str, Any]) -> Dict[str, Any]:
    level = float(payload["level"])
    async with driver_lock:
        driver.set_trigger_level(level)
        state = driver.get_trigger_state()
    return state


def _normalize_channels(channels: Any) -> List[str]:
    if isinstance(channels, list) and channels:
        return [str(ch) for ch in channels if str(ch) in ALL_CHANNELS]
    return [DEFAULT_CHANNEL]


@app.post("/api/capture_once")
async def capture_once(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    save = False
    channels = [DEFAULT_CHANNEL]
    run_id: Optional[str] = None
    if payload:
        save = bool(payload.get("save", False))
        if "channels" in payload:
            channels = _normalize_channels(payload["channels"])
        elif "channel" in payload:
            channels = [str(payload["channel"])]
        if payload.get("run_id"):
            run_id = str(payload["run_id"])

    async with driver_lock:
        multi = driver.acquire_waveform_multi(channels)
        metas = {ch: driver.get_basic_metadata(ch) for ch in channels}

    time_s = None
    voltage_dict: Dict[str, List[float]] = {}
    for ch in channels:
        t_s, volts, raw = multi[ch]
        if time_s is None:
            time_s = t_s
        meta = metas[ch]
        meta["n_points"] = int(len(raw))
        meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
        metas[ch] = meta
        voltage_dict[ch] = volts.tolist()

        if save:
            base_dir = OUTDIR / run_id if run_id else OUTDIR
            paths = build_capture_paths(base_dir, ch)
            meta["csv_file"] = paths["csv"].name
            save_csv(paths["csv"], time_s, volts)
            save_json(paths["json"], meta)

    time_list = time_s.tolist() if time_s is not None else []
    return {
        "time": time_list,
        "voltage": voltage_dict,
        "voltage_primary": voltage_dict.get(channels[0], []),
        "meta": metas,
    }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    channels = [DEFAULT_CHANNEL]
    first_wait = True
    try:
        while True:
            try:
                timeout = 1.0 if first_wait else 0.05
                msg = await asyncio.wait_for(ws.receive_text(), timeout=timeout)
                data = json.loads(msg)
                if "channels" in data and data["channels"]:
                    channels = _normalize_channels(data["channels"])
            except asyncio.TimeoutError:
                pass
            except (ValueError, KeyError):
                pass
            first_wait = False
            async with driver_lock:
                multi = driver.acquire_waveform_multi(channels)
                metas = {ch: driver.get_basic_metadata(ch) for ch in channels}
            time_s = None
            voltage_dict: Dict[str, List[float]] = {}
            for ch in channels:
                t_s, volts, raw = multi[ch]
                if time_s is None:
                    time_s = t_s
                meta = metas[ch]
                meta["n_points"] = int(len(raw))
                metas[ch] = meta
                voltage_dict[ch] = volts.tolist()
                if baseline_common is not None:
                    diff = np.abs(volts - baseline_common)
                    peak = float(np.max(diff))
                    histogram_values[ch].append(peak)
            payload = {
                "time": time_s.tolist() if time_s is not None else [],
                "voltage": voltage_dict,
                "voltage_primary": voltage_dict.get(channels[0], []),
                "meta": metas,
            }
            await ws.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close()


@app.get("/baseline")
async def get_baseline() -> Dict[str, Any]:
    return {"baseline": baseline_common}


@app.post("/baseline/set")
async def set_baseline(payload: Dict[str, Any]) -> Dict[str, Any]:
    global baseline_common
    value = float(payload["value"])
    baseline_common = value
    return {"baseline": value}


@app.post("/analysis/append")
async def analysis_append(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    channel = DEFAULT_CHANNEL
    if payload and "channel" in payload:
        channel = str(payload["channel"])
    if channel not in ALL_CHANNELS:
        channel = DEFAULT_CHANNEL
    if baseline_common is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "baseline not set"},
        )
    bl = baseline_common
    async with driver_lock:
        driver.setup_waveform_transfer(channel=channel)
        time_s, volts, raw = driver.acquire_waveform()
        meta = driver.get_basic_metadata(channel=channel)
    diff = np.abs(volts - bl)
    peak = float(np.max(diff))
    histogram_values[channel].append(peak)
    return {
        "baseline": bl,
        "peak": peak,
        "count": len(histogram_values[channel]),
        "channel": channel,
        "meta": meta,
    }


@app.get("/analysis/histogram")
async def analysis_histogram(channel: str = DEFAULT_CHANNEL, bins: int = 20) -> Dict[str, Any]:
    if channel not in ALL_CHANNELS:
        channel = DEFAULT_CHANNEL
    values = histogram_values.get(channel, [])
    if not values:
        return {"bins": [], "counts": [], "count": 0, "channel": channel}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "count": len(values),
        "channel": channel,
    }


@app.post("/analysis/clear")
async def analysis_clear(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    channel = None
    if payload and "channel" in payload:
        channel = str(payload["channel"])
    if channel and channel in ALL_CHANNELS:
        histogram_values[channel] = []
        return {"status": "cleared", "count": 0, "channel": channel}
    for ch in ALL_CHANNELS:
        histogram_values[ch] = []
    return {"status": "cleared", "count": 0}


def _save_one_channel_sync(
    csv_path: Path,
    json_path: Path,
    time_s: np.ndarray,
    volts: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """Save one channel's CSV and JSON. Safe to run in a thread."""
    save_csv(csv_path, time_s, volts)
    save_json(json_path, meta)


def _acquire_waveform_sync() -> tuple:
    """Blocking single-channel acquire. Run in thread pool."""
    driver.setup_waveform_transfer()
    time_s, volts, raw = driver.acquire_waveform()
    meta = driver.get_basic_metadata()
    return time_s, volts, raw, meta


def _acquire_waveform_multi_sync(channels: List[str]) -> tuple:
    """Blocking multi-channel acquire. Run in thread pool."""
    multi = driver.acquire_waveform_multi(channels)
    metas = {ch: driver.get_basic_metadata(ch) for ch in channels}
    return multi, metas


async def _trigger_capture_worker(run_id: str, channels: List[str]) -> None:
    """
    Trigger-driven capture loop (multi-channel).

    Runs blocking scope I/O in a thread pool. Saves each channel to CSV/meta.
    Appends peak to histogram per channel when baseline is set for that channel.
    """
    global trigger_capture_saved_count, trigger_capture_last_saved_at
    global baseline_common, histogram_values
    base_dir = OUTDIR / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    trigger_capture_saved_count = 0
    trigger_capture_last_saved_at = None
    loop = asyncio.get_event_loop()
    try:
        while True:
            if trigger_capture_stop_requested:
                break
            multi, metas = await loop.run_in_executor(
                None, lambda: _acquire_waveform_multi_sync(channels)
            )
            if trigger_capture_stop_requested:
                break
            save_tasks = []
            for ch in channels:
                time_s, volts, raw = multi[ch]
                meta = metas[ch].copy()
                meta["n_points"] = int(len(raw))
                meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
                paths = build_capture_paths(base_dir, ch)
                meta["csv_file"] = paths["csv"].name
                if baseline_common is not None:
                    diff = np.abs(volts - baseline_common)
                    peak = float(np.max(diff))
                    histogram_values[ch].append(peak)
                save_tasks.append(
                    (paths["csv"], paths["json"], time_s.copy(), volts.copy(), meta)
                )
            await asyncio.gather(
                *[
                    loop.run_in_executor(None, _save_one_channel_sync, *t)
                    for t in save_tasks
                ]
            )
            trigger_capture_saved_count += 1
            trigger_capture_last_saved_at = datetime.now()
    except asyncio.CancelledError:
        return


def _histogram_count_dict() -> Dict[str, int]:
    return {ch: len(histogram_values.get(ch, [])) for ch in ALL_CHANNELS}


@app.post("/trigger_capture/start")
async def start_trigger_capture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start background trigger-driven capture loop.
    """
    global trigger_capture_task
    global trigger_capture_active_run
    global trigger_capture_channels
    global trigger_capture_saved_count
    global trigger_capture_started_at
    global trigger_capture_last_saved_at
    global trigger_capture_stop_requested
    if trigger_capture_task is not None and not trigger_capture_task.done():
        return JSONResponse(
            status_code=400,
            content={"detail": "triggered capture already running"},
        )
    run_id = str(payload.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    channels = _normalize_channels(payload.get("channels", [DEFAULT_CHANNEL]))
    trigger_capture_stop_requested = False
    trigger_capture_started_at = datetime.now()
    trigger_capture_last_saved_at = None
    trigger_capture_saved_count = 0
    trigger_capture_channels = channels
    trigger_capture_task = asyncio.create_task(_trigger_capture_worker(run_id, channels))
    trigger_capture_active_run = run_id
    return {
        "status": "started",
        "run_id": run_id,
        "channels": channels,
        "saved_count": trigger_capture_saved_count,
        "histogram_count": _histogram_count_dict(),
    }


@app.post("/trigger_capture/stop")
async def stop_trigger_capture() -> Dict[str, Any]:
    """
    Stop background trigger-driven capture loop.
    """
    global trigger_capture_task, trigger_capture_active_run, trigger_capture_saved_count
    global trigger_capture_stop_requested
    if trigger_capture_task is None:
        trigger_capture_stop_requested = False
        return {
            "status": "stopped",
            "run_id": trigger_capture_active_run,
            "saved_count": trigger_capture_saved_count,
            "histogram_count": _histogram_count_dict(),
        }
    if trigger_capture_task.done():
        run_id = trigger_capture_active_run
        trigger_capture_task = None
        trigger_capture_active_run = None
        trigger_capture_stop_requested = False
        return {
            "status": "stopped",
            "run_id": run_id,
            "saved_count": trigger_capture_saved_count,
            "histogram_count": _histogram_count_dict(),
        }
    trigger_capture_stop_requested = True
    trigger_capture_task.cancel()
    try:
        await trigger_capture_task
    except asyncio.CancelledError:
        pass
    run_id = trigger_capture_active_run
    trigger_capture_task = None
    trigger_capture_active_run = None
    trigger_capture_stop_requested = False
    return {
        "status": "stopped",
        "run_id": run_id,
        "saved_count": trigger_capture_saved_count,
        "histogram_count": _histogram_count_dict(),
    }


@app.get("/trigger_capture/status")
async def trigger_capture_status() -> Dict[str, Any]:
    active = trigger_capture_task is not None and not trigger_capture_task.done()
    started_iso = (
        trigger_capture_started_at.isoformat(timespec="seconds")
        if trigger_capture_started_at
        else None
    )
    last_saved_iso = (
        trigger_capture_last_saved_at.isoformat(timespec="seconds")
        if trigger_capture_last_saved_at
        else None
    )
    duration_s: Optional[float] = None
    rate_per_min: Optional[float] = None
    if trigger_capture_started_at:
        duration_s = (datetime.now() - trigger_capture_started_at).total_seconds()
        if duration_s > 0 and trigger_capture_saved_count > 0:
            rate_per_min = trigger_capture_saved_count / (duration_s / 60.0)
    return {
        "active": active,
        "run_id": trigger_capture_active_run,
        "channels": trigger_capture_channels,
        "saved_count": trigger_capture_saved_count,
        "histogram_count": _histogram_count_dict(),
        "started_at": started_iso,
        "last_saved_at": last_saved_iso,
        "duration_s": duration_s,
        "rate_per_min": rate_per_min,
    }

