from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import httpx

from app.config import (
    CAMERA_STREAM_URL,
    DATA_WINDOW_MODE,
    DEFAULT_CHANNEL,
    OSC_IP,
    OUTDIR,
    START_INDEX,
    STOP_INDEX,
    TIMEOUT_MS,
    TRIGGER_CAPTURE_ACQUISITION_TIMEOUT_S,
    TRIGGER_CAPTURE_STRICT_TRIGGER,
)
from app.scope import (
    ScopeConfig,
    ScopeDriver,
    build_capture_paths,
    save_csv,
    save_json,
)

ALL_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]

# Edge trigger source: analog channels + external (Tek SCPI names).
TRIGGER_EDGE_SOURCES = ["CH1", "CH2", "CH3", "CH4", "EXT"]

_RUN_DIR_RE = re.compile(r"^run(\d{4})$")


def next_run_id_from_outdir() -> str:
    best = 0
    if OUTDIR.exists():
        for p in OUTDIR.iterdir():
            if not p.is_dir():
                continue
            m = _RUN_DIR_RE.match(p.name)
            if m:
                best = max(best, int(m.group(1)))
    return f"run{best + 1:04d}"


scope_config = ScopeConfig(
    ip=OSC_IP,
    channel=DEFAULT_CHANNEL,
    start_index=START_INDEX,
    stop_index=STOP_INDEX,
    timeout_ms=TIMEOUT_MS,
    data_window_mode=DATA_WINDOW_MODE,
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


@app.get("/scope/display")
async def scope_display() -> Dict[str, Any]:
    """Current oscilloscope horizontal scale and per-channel vertical scale/position (SCPI)."""
    async with driver_lock:
        return driver.get_scope_display_state()


@app.post("/scope/horizontal")
async def scope_set_horizontal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``HORIZONTAL:SCALE`` (seconds per division)."""
    try:
        scale = float(payload["scale_s_per_div"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": "body must include numeric scale_s_per_div"},
        )
    if scale <= 0:
        return JSONResponse(
            status_code=400,
            content={"detail": "scale_s_per_div must be positive"},
        )
    try:
        async with driver_lock:
            prev_acquire_state = None
            acquire_state = None
            driver.set_horizontal_scale_s(scale)
            trigger_left_fraction = payload.get("trigger_left_fraction", None)
            if trigger_left_fraction is not None:
                # Some scopes ignore horizontal position changes while running.
                # Temporarily STOP and restore the previous state.
                try:
                    prev_acquire_state = driver.get_acquire_state()
                    acquire_state = prev_acquire_state
                except Exception:
                    prev_acquire_state = None
                    acquire_state = None
                if acquire_state in ("RUN", "RUNNING", "RUNSTOP"):
                    try:
                        driver.set_acquire_state("STOP")
                    except Exception:
                        pass
                    # Match debug.py: configure Stop-A behavior before changing HOR controls.
                    try:
                        driver.set_stopa_runstop()
                    except Exception:
                        pass
                driver.set_trigger_left_fraction(
                    left_fraction=float(trigger_left_fraction),
                    scale_s_per_div=scale,
                )
                if prev_acquire_state in ("RUN", "RUNNING", "RUNSTOP"):
                    try:
                        driver.set_acquire_state("RUN")
                    except Exception:
                        pass
            h = driver.get_horizontal_scale_s()
            # Optional read-back for debugging / UI verification.
            horizontal_delay_s = None
            horizontal_position_s = None
            horizontal_hor_pos_percent = None
            horizontal_hor_del_mod = None
            try:
                horizontal_delay_s = driver.get_horizontal_delay_s()
            except Exception:
                pass
            try:
                horizontal_position_s = driver.get_horizontal_position_s()
            except Exception:
                pass
            try:
                # Tek front-panel style shorthand; may not be supported by all models.
                horizontal_hor_pos_percent = float(driver.scope.query("HOR:POS?").strip())  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                horizontal_hor_del_mod = driver.scope.query("HOR:DEL:MOD?").strip()  # type: ignore[attr-defined]
            except Exception:
                pass
        resp: Dict[str, Any] = {"ok": True, "horizontal_scale_s_per_div": h}
        if trigger_left_fraction is not None:
            resp["trigger_left_fraction"] = trigger_left_fraction
            resp["acquire_state"] = acquire_state
            if horizontal_delay_s is not None:
                resp["horizontal_delay_s"] = horizontal_delay_s
            if horizontal_position_s is not None:
                resp["horizontal_position_s"] = horizontal_position_s
            if horizontal_hor_pos_percent is not None:
                resp["horizontal_hor_pos_percent"] = horizontal_hor_pos_percent
            if horizontal_hor_del_mod is not None:
                resp["horizontal_hor_del_mod"] = horizontal_hor_del_mod
        return resp
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.post("/scope/channel/{channel}/vertical")
async def scope_set_channel_vertical(channel: str, payload: Dict[str, Any]) -> Any:
    """Set ``CHx:SCALE`` (V/div) and/or ``CHx:POSITION`` (divisions)."""
    ch = channel.strip().upper()
    if ch not in ALL_CHANNELS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"invalid channel {channel!r}; use CH1–CH4"},
        )
    scale = payload.get("scale_v_per_div")
    pos = payload.get("position_div")
    kw: Dict[str, float] = {}
    if scale is not None:
        try:
            kw["scale_v_per_div"] = float(scale)
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=400,
                content={"detail": "scale_v_per_div must be a number"},
            )
        if kw["scale_v_per_div"] <= 0:
            return JSONResponse(
                status_code=400,
                content={"detail": "scale_v_per_div must be positive"},
            )
    if pos is not None:
        try:
            kw["position_div"] = float(pos)
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=400,
                content={"detail": "position_div must be a number"},
            )
    if not kw:
        return JSONResponse(
            status_code=400,
            content={"detail": "provide scale_v_per_div and/or position_div"},
        )
    try:
        async with driver_lock:
            driver.set_channel_vertical(ch, **kw)
            state = driver.get_channel_vertical(ch)
        return {"ok": True, "channel": ch, **state}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.get("/scope/wfmoutpre")
async def get_scope_wfmoutpre() -> Dict[str, Any]:
    """
    Raw ``WFMOUTPRE?`` response (comma-separated fields for the last ``CURVE?`` transfer).

    Equivalent to: ``preamble = scope.query("WFMOUTPRE?")``
    """
    async with driver_lock:
        raw = driver.query_wfmoutpre_preamble()
    return {
        "wfmoutpre": raw,
        "note": "Tek WFMOUTPRE comma-separated preamble; reflects the last CURVE? transfer.",
    }


@app.get("/storage")
async def storage_info() -> Dict[str, Any]:
    try:
        # Get disk usage for the output directory
        stat = shutil.disk_usage(OUTDIR.parent if OUTDIR.exists() else Path.cwd())
        
        # Calculate output directory size
        output_size = 0
        if OUTDIR.exists():
            for item in OUTDIR.rglob("*"):
                if item.is_file():
                    output_size += item.stat().st_size
        
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "output_size": output_size,
            "output_dir": str(OUTDIR),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )


@app.get("/camera/config")
async def camera_config() -> Dict[str, Any]:
    return {"enabled": CAMERA_STREAM_URL is not None and len(CAMERA_STREAM_URL) > 0}


@app.get("/camera/stream")
async def camera_stream():
    if not CAMERA_STREAM_URL:
        return JSONResponse(status_code=503, content={"detail": "camera stream not configured"})

    # mjpg-streamer returns 400 on HEAD, so we probe with GET to get Content-Type first
    media_type = "multipart/x-mixed-replace; boundary=boundary"
    async with httpx.AsyncClient(timeout=10.0) as probe_client:
        try:
            async with probe_client.stream("GET", CAMERA_STREAM_URL) as probe_response:
                probe_response.raise_for_status()
                ct = probe_response.headers.get("content-type")
                if ct and "multipart" in ct.lower():
                    media_type = ct
        except Exception:
            pass
    
    async def stream_chunks():
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("GET", CAMERA_STREAM_URL) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_chunks(), media_type=media_type)


@app.get("/trigger")
async def get_trigger() -> Dict[str, Any]:
    async with driver_lock:
        state = driver.get_trigger_state()
    return state


@app.post("/trigger/source")
async def set_trigger_source(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = str(payload.get("source", "CH1")).strip()
    if source not in TRIGGER_EDGE_SOURCES:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"invalid trigger source: {source!r}; "
                f"allowed: {', '.join(TRIGGER_EDGE_SOURCES)}",
            },
        )
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
    """
    Single acquisition (optionally saved under ``run_id``).

    **Acquisition discipline:** For a given measurement run (same ``run_id`` or a continuous
    ``trigger_capture`` session), keep oscilloscope conditions fixed—timebase, record length,
    trigger, input coupling/bandwidth, etc. Do not change the instrument mid-run; otherwise
    saved traces are not comparable.
    """
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
        t_s, volts, raw, wfm = multi[ch]
        if time_s is None:
            time_s = t_s
        meta = metas[ch]
        meta["n_points"] = int(len(raw))
        meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
        meta["wfmoutpre"] = dict(wfm)
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
                t_s, volts, raw, wfm = multi[ch]
                if time_s is None:
                    time_s = t_s
                meta = metas[ch]
                meta["n_points"] = int(len(raw))
                meta["wfmoutpre"] = dict(wfm)
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
        time_s, volts, raw, wfm = driver.acquire_waveform()
        meta = driver.get_basic_metadata(channel=channel)
        meta["wfmoutpre"] = dict(wfm)
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
async def analysis_histogram(
    channel: str = DEFAULT_CHANNEL,
    bins: int = Query(20, ge=1),
    range_min: Optional[float] = Query(None),
    range_max: Optional[float] = Query(None),
) -> Dict[str, Any]:
    if channel not in ALL_CHANNELS:
        channel = DEFAULT_CHANNEL
    values = histogram_values.get(channel, [])
    if not values:
        return {"bins": [], "counts": [], "count": 0, "channel": channel}
    if (range_min is None) ^ (range_max is None):
        return JSONResponse(
            status_code=400,
            content={"detail": "range_min and range_max must both be set or both omitted"},
        )
    hist_range: Optional[tuple[float, float]] = None
    if range_min is not None and range_max is not None:
        if range_min >= range_max:
            return JSONResponse(
                status_code=400,
                content={"detail": "range_min must be less than range_max"},
            )
        hist_range = (range_min, range_max)
    counts, edges = np.histogram(values, bins=bins, range=hist_range)
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
    time_s, volts, raw, wfm = driver.acquire_waveform()
    meta = driver.get_basic_metadata()
    meta["wfmoutpre"] = dict(wfm)
    return time_s, volts, raw, meta


def _acquire_waveform_multi_sync(channels: List[str]) -> tuple:
    """Blocking multi-channel acquire. Run in thread pool."""
    multi = driver.acquire_waveform_multi(channels)
    metas = {ch: driver.get_basic_metadata(ch) for ch in channels}
    return multi, metas


def _acquire_waveform_multi_sync_after_trigger(channels: List[str]) -> tuple:
    """
    One completed acquisition per save: ARM + wait for trigger/record, then CURVE per channel.
    """
    multi = driver.acquire_waveform_multi_after_trigger(
        channels,
        acquisition_timeout_s=TRIGGER_CAPTURE_ACQUISITION_TIMEOUT_S,
    )
    metas = {ch: driver.get_basic_metadata(ch) for ch in channels}
    return multi, metas


async def _trigger_capture_worker(run_id: str, channels: List[str]) -> None:
    """
    Trigger-driven capture loop (multi-channel).

    Runs blocking scope I/O in a thread pool. Saves each channel to CSV/meta.
    Appends peak to histogram per channel when baseline is set for that channel.

    **Acquisition discipline:** Until this loop is stopped, do not change oscilloscope
    settings (horizontal scale, record length, trigger, channel setup, etc.). One ``run_id``
    should mean one fixed acquisition profile.

    When ``TRIGGER_CAPTURE_STRICT_TRIGGER`` is true, each save follows a completed
    acquisition (``ACQUIRE:STOPAFTER SEQUENCE``, ``ACQUIRE:STATE RUN``, wait, then ``CURVE?``).
    When false (default), the current waveform buffer is read every iteration so files are
    saved reliably; duplicates are possible if the scope has not retriggered yet.
    """
    global trigger_capture_saved_count, trigger_capture_last_saved_at
    global baseline_common, histogram_values
    base_dir = OUTDIR / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    trigger_capture_saved_count = 0
    trigger_capture_last_saved_at = None
    loop = asyncio.get_event_loop()
    session_started = False
    try:
        if TRIGGER_CAPTURE_STRICT_TRIGGER:
            await loop.run_in_executor(None, driver.begin_triggered_capture_session)
            session_started = True
        while True:
            if trigger_capture_stop_requested:
                break
            try:
                if TRIGGER_CAPTURE_STRICT_TRIGGER:
                    multi, metas = await loop.run_in_executor(
                        None,
                        _acquire_waveform_multi_sync_after_trigger,
                        channels,
                    )
                else:
                    multi, metas = await loop.run_in_executor(
                        None,
                        _acquire_waveform_multi_sync,
                        channels,
                    )
            except TimeoutError:
                if not TRIGGER_CAPTURE_STRICT_TRIGGER:
                    raise
                if trigger_capture_stop_requested:
                    break
                continue
            if trigger_capture_stop_requested:
                break
            save_tasks = []
            for ch in channels:
                time_s, volts, raw, wfm = multi[ch]
                meta = metas[ch].copy()
                meta["wfmoutpre"] = dict(wfm)
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
    finally:
        if session_started:
            await loop.run_in_executor(None, driver.end_triggered_capture_session)


def _histogram_count_dict() -> Dict[str, int]:
    return {ch: len(histogram_values.get(ch, [])) for ch in ALL_CHANNELS}


@app.post("/trigger_capture/start")
async def start_trigger_capture(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start background trigger-driven capture loop.

    Operators should treat the returned ``run_id`` as a single experiment: keep scope
    acquisition parameters unchanged until ``/trigger_capture/stop`` (see module notes on
    ``capture_once`` / ``_trigger_capture_worker``).
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
    run_id = next_run_id_from_outdir()
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

