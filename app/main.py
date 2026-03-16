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
    save_plot,
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
logging_task: Optional[asyncio.Task] = None
logging_active_run: Optional[str] = None

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
            idn = driver.scope.query("*IDN?").strip()
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
    if payload:
        save = bool(payload.get("save", False))
        if "channel" in payload:
            channel = str(payload["channel"])

    async with driver_lock:
        driver.config.channel = channel
        driver.setup_waveform_transfer()
        time_s, volts, raw = driver.acquire_waveform()
        meta = driver.get_basic_metadata()

    meta["n_points"] = int(len(raw))
    meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")

    if save:
        paths = build_capture_paths(OUTDIR, channel)
        meta["csv_file"] = paths["csv"].name
        meta["png_file"] = paths["png"].name
        save_csv(paths["csv"], time_s, volts)
        save_plot(paths["png"], time_s, volts, meta)
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


async def _logging_worker(interval_s: float, run_id: str) -> None:
    base_dir = OUTDIR / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    while True:
        async with driver_lock:
            driver.setup_waveform_transfer()
            time_s, volts, raw = driver.acquire_waveform()
            meta = driver.get_basic_metadata()
        meta["n_points"] = int(len(raw))
        meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
        paths = build_capture_paths(base_dir, driver.config.channel)
        meta["csv_file"] = paths["csv"].name
        meta["png_file"] = paths["png"].name
        save_csv(paths["csv"], time_s, volts)
        save_plot(paths["png"], time_s, volts, meta)
        save_json(paths["json"], meta)
        await asyncio.sleep(interval_s)


@app.post("/logging/start")
async def start_logging(payload: Dict[str, Any]) -> Dict[str, Any]:
    global logging_task, logging_active_run
    if logging_task is not None and not logging_task.done():
        return JSONResponse(
            status_code=400,
            content={"detail": "logging already running"},
        )
    run_id = str(payload.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    interval_s = float(payload.get("interval_s", 5.0))
    logging_active_run = run_id
    logging_task = asyncio.create_task(_logging_worker(interval_s, run_id))
    return {"run_id": run_id, "interval_s": interval_s}


@app.post("/logging/stop")
async def stop_logging() -> Dict[str, Any]:
    global logging_task, logging_active_run
    if logging_task is None or logging_task.done():
        return {"stopped": False}
    logging_task.cancel()
    try:
        await logging_task
    except Exception:
        pass
    logging_task = None
    run_id = logging_active_run
    logging_active_run = None
    return {"stopped": True, "run_id": run_id}

