from __future__ import annotations

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


scope_config = ScopeConfig(
    ip=OSC_IP,
    channel=DEFAULT_CHANNEL,
    start_index=START_INDEX,
    stop_index=STOP_INDEX,
    timeout_ms=TIMEOUT_MS,
)

driver = ScopeDriver(scope_config)
driver_lock = asyncio.Lock()

baseline_value: Optional[float] = None
histogram_values: List[float] = []
trigger_capture_task: Optional[asyncio.Task] = None
trigger_capture_active_run: Optional[str] = None
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


@app.post("/api/capture_once")
async def capture_once(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    save = False
    channel = DEFAULT_CHANNEL
    run_id: Optional[str] = None
    if payload:
        save = bool(payload.get("save", False))
        if "channel" in payload:
            channel = str(payload["channel"])
        if payload.get("run_id"):
            run_id = str(payload["run_id"])

    async with driver_lock:
        driver.config.channel = channel
        driver.setup_waveform_transfer()
        time_s, volts, raw = driver.acquire_waveform()
        meta = driver.get_basic_metadata()

    meta["n_points"] = int(len(raw))
    meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")

    if save:
        base_dir = OUTDIR / run_id if run_id else OUTDIR
        paths = build_capture_paths(base_dir, channel)
        meta["csv_file"] = paths["csv"].name
        save_csv(paths["csv"], time_s, volts)
        save_json(paths["json"], meta)

    return {
        "time": time_s.tolist(),
        "voltage": volts.tolist(),
        "meta": meta,
    }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            async with driver_lock:
                driver.setup_waveform_transfer()
                time_s, volts, raw = driver.acquire_waveform()
                meta = driver.get_basic_metadata()
            payload = {
                "time": time_s.tolist(),
                "voltage": volts.tolist(),
                "meta": meta,
            }
            # During streaming, append peaks for histogram if baseline is set.
            if baseline_value is not None:
                diff = np.abs(volts - baseline_value)
                peak = float(np.max(diff))
                histogram_values.append(peak)
            await ws.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close()


@app.get("/baseline")
async def get_baseline() -> Dict[str, Any]:
    return {"baseline": baseline_value}


@app.post("/baseline/set")
async def set_baseline(payload: Dict[str, Any]) -> Dict[str, Any]:
    global baseline_value
    baseline_value = float(payload["value"])
    return {"baseline": baseline_value}


@app.post("/analysis/append")
async def analysis_append(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global baseline_value
    if baseline_value is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "baseline not set"},
        )

    async with driver_lock:
        driver.setup_waveform_transfer()
        time_s, volts, raw = driver.acquire_waveform()
        meta = driver.get_basic_metadata()

    diff = np.abs(volts - baseline_value)
    peak = float(np.max(diff))

    histogram_values.append(peak)

    return {
        "baseline": baseline_value,
        "peak": peak,
        "count": len(histogram_values),
        "meta": meta,
    }


@app.get("/analysis/histogram")
async def analysis_histogram(bins: int = 20) -> Dict[str, Any]:
    if not histogram_values:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(histogram_values, bins=bins)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "count": len(histogram_values),
    }


@app.post("/analysis/clear")
async def analysis_clear() -> Dict[str, Any]:
    histogram_values.clear()
    return {"status": "cleared", "count": 0}


def _acquire_waveform_sync() -> tuple:
    """Blocking acquire (setup + waveform + meta). Run in thread pool."""
    driver.setup_waveform_transfer()
    time_s, volts, raw = driver.acquire_waveform()
    meta = driver.get_basic_metadata()
    return time_s, volts, raw, meta


async def _trigger_capture_worker(run_id: str) -> None:
    """
    Trigger-driven capture loop.

    Runs blocking scope I/O in a thread pool so that task.cancel() and
    stop_requested can take effect between captures.
    Automatically appends peak (vs baseline) to histogram when baseline is set.
    """
    global trigger_capture_saved_count, trigger_capture_last_saved_at
    global baseline_value, histogram_values
    base_dir = OUTDIR / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    trigger_capture_saved_count = 0
    trigger_capture_last_saved_at = None
    loop = asyncio.get_event_loop()
    try:
        while True:
            if trigger_capture_stop_requested:
                break
            time_s, volts, raw, meta = await loop.run_in_executor(
                None, _acquire_waveform_sync
            )
            if trigger_capture_stop_requested:
                break
            meta["n_points"] = int(len(raw))
            meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
            paths = build_capture_paths(base_dir, driver.config.channel)
            meta["csv_file"] = paths["csv"].name
            save_csv(paths["csv"], time_s, volts)
            save_json(paths["json"], meta)
            trigger_capture_saved_count += 1
            trigger_capture_last_saved_at = datetime.now()
            if baseline_value is not None:
                diff = np.abs(volts - baseline_value)
                peak = float(np.max(diff))
                histogram_values.append(peak)
    except asyncio.CancelledError:
        return


@app.post("/trigger_capture/start")
async def start_trigger_capture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start background trigger-driven capture loop.
    """
    global trigger_capture_task
    global trigger_capture_active_run
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
    trigger_capture_stop_requested = False
    trigger_capture_started_at = datetime.now()
    trigger_capture_last_saved_at = None
    trigger_capture_saved_count = 0
    trigger_capture_task = asyncio.create_task(_trigger_capture_worker(run_id))
    trigger_capture_active_run = run_id
    return {
        "status": "started",
        "run_id": run_id,
        "saved_count": trigger_capture_saved_count,
        "histogram_count": len(histogram_values),
    }


@app.post("/trigger_capture/stop")
async def stop_trigger_capture() -> Dict[str, Any]:
    """
    Stop background trigger-driven capture loop.
    Sets stop_requested so the worker exits after the current capture (or immediately
    if it is between captures). Also cancels the task so we don't wait forever.
    """
    global trigger_capture_task, trigger_capture_active_run, trigger_capture_saved_count
    global trigger_capture_stop_requested
    if trigger_capture_task is None:
        trigger_capture_stop_requested = False
        return {
            "status": "stopped",
            "run_id": trigger_capture_active_run,
            "saved_count": trigger_capture_saved_count,
            "histogram_count": len(histogram_values),
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
            "histogram_count": len(histogram_values),
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
        "histogram_count": len(histogram_values),
    }


@app.get("/trigger_capture/status")
async def trigger_capture_status() -> Dict[str, Any]:
    # Compute basic stats for monitoring.
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
        "saved_count": trigger_capture_saved_count,
        "histogram_count": len(histogram_values),
        "started_at": started_iso,
        "last_saved_at": last_saved_iso,
        "duration_s": duration_s,
        "rate_per_min": rate_per_min,
    }

