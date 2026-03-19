#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pyvisa
import requests

from app.scope.driver import (
    ScopeConfig,
    ScopeDriver,
    build_capture_paths,
    save_csv,
    save_json,
    save_plot,
)


# =========================
# User settings (env)
# =========================
OSC_IP = os.environ.get("OSC_IP")
if not OSC_IP:
    raise RuntimeError("OSC_IP environment variable is not set.")

WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

CHANNEL = os.environ.get("OSC_CHANNEL", "CH1")  # CH1, CH2, CH3, CH4
START_INDEX = int(os.environ.get("OSC_START_INDEX", "1"))
STOP_INDEX = int(os.environ.get("OSC_STOP_INDEX", "10000"))
TIMEOUT_MS = int(os.environ.get("OSC_TIMEOUT_MS", "30000"))
OUTDIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))


# =========================
# Utility
# =========================
def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "output"


def ensure_ok(resp: requests.Response) -> None:
    if not resp.ok:
        raise RuntimeError(
            f"Discord webhook failed: {resp.status_code} {resp.text}"
        )


OSC_CONFIG = ScopeConfig(
    ip=OSC_IP,
    channel=CHANNEL,
    start_index=START_INDEX,
    stop_index=STOP_INDEX,
    timeout_ms=TIMEOUT_MS,
)


# =========================
# Discord
# =========================
def post_to_discord(
    webhook_url: str,
    message: str,
    png_path: Path,
    csv_path: Path,
    json_path: Path
) -> None:
    payload = {
        "content": message,
        "allowed_mentions": {"parse": []},
    }

    with (
        png_path.open("rb") as f_png,
        csv_path.open("rb") as f_csv,
        json_path.open("rb") as f_json,
    ):
        files = {
            "files[0]": (png_path.name, f_png, "image/png"),
            "files[1]": (csv_path.name, f_csv, "text/csv"),
            "files[2]": (json_path.name, f_json, "application/json"),
        }
        data = {
            "payload_json": json.dumps(payload, ensure_ascii=False)
        }
        resp = requests.post(webhook_url, data=data, files=files, timeout=60)
        ensure_ok(resp)


# =========================
# Main
# =========================
def main():
    paths = build_capture_paths(OUTDIR, CHANNEL)

    print("Connecting to scope...")
    driver = ScopeDriver(OSC_CONFIG)

    try:
        scope = driver.scope
        print("IDN:", scope.query("*IDN?").strip())

        print("Configuring waveform transfer...")
        driver.setup_waveform_transfer()

        print("Reading waveform...")
        time_s, volts, raw = driver.acquire_waveform()

        print("Reading metadata...")
        meta = driver.get_basic_metadata()
        meta["n_points"] = int(len(raw))
        meta["timestamp_local"] = datetime.now().isoformat(timespec="seconds")
        meta["csv_file"] = paths["csv"].name
        meta["png_file"] = paths["png"].name

        print("Saving CSV...")
        save_csv(paths["csv"], time_s, volts)

        print("Saving PNG...")
        save_plot(paths["png"], time_s, volts, meta)

        print("Saving JSON metadata...")
        save_json(paths["json"], meta)

        peak_v = float(np.max(volts))
        min_v = float(np.min(volts))
        vpp = peak_v - min_v

        message = (
            f"Oscilloscope capture done\n"
            f"- Instrument: `{meta['idn']}`\n"
            f"- Channel: `{meta['channel']}`\n"
            f"- Points: `{meta['n_points']}`\n"
            f"- Vmax: `{peak_v:.6g} V`\n"
            f"- Vmin: `{min_v:.6g} V`\n"
            f"- Vpp: `{vpp:.6g} V`"
        )

        if WEBHOOK_URL:
            print("Posting to Discord...")
            post_to_discord(
                WEBHOOK_URL,
                message,
                paths["png"],
                paths["csv"],
                paths["json"],
            )

        print("Done.")
        print("Saved:", paths["png"])
        print("Saved:", paths["csv"])
        print("Saved:", paths["json"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
