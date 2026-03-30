#!/usr/bin/env python3
"""
Quickly review CSV waveforms (glass-gem save_csv format) and build two histograms:
  - max(|V - baseline|)  (same spirit as the web UI)
  - integral of (V - baseline) over [INTEGRAL_T_MIN, INTEGRAL_T_MAX]

Edit the constants below for view limits and integration window.

Per-file time span differs when the scope horizontal scale (or record length) changed between
captures; use None for time limits so each trace fills the axis. Saved JSON next to each CSV
includes ``hscale`` and ``wfmoutpre`` to compare acquisition settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# --- Edit these ---
BASELINE_V = 0.0

# Waveform plot limits (seconds, volts).
# Time: None, None = auto (min/max of each trace + small padding). Fixed window only if both set.
# Voltage: None = fit each trace (min/max + pad).
VIEW_T_MIN = None
VIEW_T_MAX = None
VIEW_V_MIN = None
VIEW_V_MAX = None

# Integration window (seconds); only samples with INTEGRAL_T_MIN <= t <= INTEGRAL_T_MAX contribute.
INTEGRAL_T_MIN = -5e-6
INTEGRAL_T_MAX = 5e-6

HIST_BINS = 50

# glass-gem save_csv writes one header line (e.g. "# time_s,voltage_V"); skip it by default.
SKIP_HEADER_ROWS = 1
# --- end constants ---


def read_waveform_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load time_s, voltage_V from glass-gem CSV (header line skipped via SKIP_HEADER_ROWS)."""
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=SKIP_HEADER_ROWS,
        comments="#",
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("expected at least 2 columns")
    return data[:, 0].copy(), data[:, 1].copy()


def peak_metric(voltage: np.ndarray, baseline: float) -> float:
    return float(np.max(np.abs(voltage - baseline)))


def integral_metric(
    time_s: np.ndarray,
    voltage: np.ndarray,
    baseline: float,
    t0: float,
    t1: float,
) -> float:
    if t1 <= t0:
        raise ValueError("INTEGRAL_T_MAX must be > INTEGRAL_T_MIN")
    mask = (time_s >= t0) & (time_s <= t1)
    tt = time_s[mask]
    vv = voltage[mask] - baseline
    n = tt.size
    if n < 2:
        return float("nan")
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(vv, tt))
    return float(np.trapz(vv, tt))


def apply_view_limits(ax, time_s: np.ndarray, voltage: np.ndarray) -> None:
    if VIEW_T_MIN is not None and VIEW_T_MAX is not None:
        ax.set_xlim(VIEW_T_MIN, VIEW_T_MAX)
    else:
        t0, t1 = float(np.min(time_s)), float(np.max(time_s))
        span = t1 - t0
        tpad = 0.02 * span if span > 0 else 1e-12
        ax.set_xlim(t0 - tpad, t1 + tpad)
    if VIEW_V_MIN is not None and VIEW_V_MAX is not None:
        ax.set_ylim(VIEW_V_MIN, VIEW_V_MAX)
    else:
        pad = 0.05 * (float(np.max(voltage)) - float(np.min(voltage)) + 1e-12)
        ax.set_ylim(float(np.min(voltage)) - pad, float(np.max(voltage)) + pad)


def _print_time_ranges_and_json_hint(csv_paths: list[Path]) -> None:
    """Summarize t_min/t_max per CSV and hscale from sibling json/ if present."""
    json_dir = csv_paths[0].parent.parent / "json"
    rows: list[tuple[str, float, float, float, str]] = []
    for fp in csv_paths:
        t, _ = read_waveform_csv(fp)
        t0, t1 = float(np.min(t)), float(np.max(t))
        span = t1 - t0
        hs = "?"
        jpath = json_dir / (fp.stem + ".json")
        if jpath.is_file():
            try:
                meta = json.loads(jpath.read_text(encoding="utf-8"))
                hs = str(meta.get("hscale", "?"))
            except (OSError, json.JSONDecodeError):
                hs = "(json read failed)"
        rows.append((fp.name, t0, t1, span, hs))

    spans = [r[3] for r in rows]
    hset = {r[4] for r in rows}
    print(
        f"Time axis summary ({len(rows)} file(s)): "
        f"span min={min(spans):.6e} s, max={max(spans):.6e} s",
        flush=True,
    )
    print(f"  distinct json hscale values: {len(hset)}  {sorted(hset)}", flush=True)
    n_show = min(8, len(rows))
    print(f"  first {n_show} file(s):", flush=True)
    for r in rows[:n_show]:
        print(
            f"    {r[0]}:  [{r[1]:.6e}, {r[2]:.6e}]  span={r[3]:.6e} s  hscale={r[4]}",
            flush=True,
        )
    if len(rows) > n_show:
        print(f"    ... ({len(rows) - n_show} more)", flush=True)

    if len(hset) > 1:
        print(
            "\nNote: multiple distinct hscale values — horizontal scale (or equivalent) "
            "changed between captures; time spans differ by design. Lock the scope timebase "
            "for comparable runs.",
            file=sys.stderr,
            flush=True,
        )


def collect_metrics(
    files: list[Path],
    baseline: float,
    t0: float,
    t1: float,
) -> tuple[list[str], list[float], list[float], list[tuple[np.ndarray, np.ndarray]]]:
    paths_ok: list[str] = []
    peaks: list[float] = []
    integrals: list[float] = []
    waveforms: list[tuple[np.ndarray, np.ndarray]] = []
    skipped: list[tuple[str, str]] = []

    for fp in files:
        try:
            t, v = read_waveform_csv(fp)
            if t.size == 0:
                raise ValueError("empty")
            peaks.append(peak_metric(v, baseline))
            integrals.append(integral_metric(t, v, baseline, t0, t1))
            paths_ok.append(str(fp))
            waveforms.append((t, v))
        except Exception as exc:
            skipped.append((str(fp), str(exc)))

    if skipped:
        print(f"Skipped {len(skipped)} file(s):", file=sys.stderr)
        for p, msg in skipped[:15]:
            print(f"  {p}: {msg}", file=sys.stderr)
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more", file=sys.stderr)

    return paths_ok, peaks, integrals, waveforms


def plot_histograms(
    peaks: list[float],
    integrals: list[float],
    bins: int,
    save: Optional[Path],
) -> None:
    peaks_a = np.asarray(peaks, dtype=float)
    int_a = np.asarray(integrals, dtype=float)
    int_valid = int_a[np.isfinite(int_a)]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(peaks_a, bins=bins)
    ax1.set_xlabel("max(|V - baseline|) [V]")
    ax1.set_ylabel("Counts")
    ax1.set_title("Peak amplitude (abs deviation from baseline)")
    fig1.tight_layout()
    if save:
        fig1.savefig(save / "hist_peak.png", dpi=200)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if int_valid.size > 0:
        ax2.hist(int_valid, bins=bins)
    else:
        ax2.text(0.5, 0.5, "No valid integrals", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel(r"$\int (V - b)\,dt$ over [INTEGRAL_T_MIN, INTEGRAL_T_MAX] [V·s]")
    ax2.set_ylabel("Counts")
    ax2.set_title("Integrated signal (baseline-subtracted)")
    fig2.tight_layout()
    if save:
        fig2.savefig(save / "hist_integral.png", dpi=200)


def run_interactive(
    paths: list[str],
    waveforms: list[tuple[np.ndarray, np.ndarray]],
    peaks: list[float],
    integrals: list[float],
    bins: int,
    save: Optional[Path],
) -> None:
    n = len(paths)
    if n == 0:
        print("No waveforms to show.")
        return

    idx = [0]

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        fig.canvas.manager.set_window_title("quick_review — n next, p prev, q quit")
    except Exception:
        pass

    (line,) = ax.plot([], [], lw=1.0, color="C0")

    def redraw():
        t, v = waveforms[idx[0]]
        line.set_data(t, v)
        # relim/autoscale is expensive on every keypress; skip when axes are fully fixed by constants
        view_fully_fixed = (
            VIEW_T_MIN is not None
            and VIEW_T_MAX is not None
            and VIEW_V_MIN is not None
            and VIEW_V_MAX is not None
        )
        if not view_fully_fixed:
            ax.relim()
            ax.autoscale_view()
        apply_view_limits(ax, t, v)
        p = peaks[idx[0]]
        ing = integrals[idx[0]]
        ing_s = f"{ing:.3e}" if np.isfinite(ing) else "nan"
        ax.set_title(
            f"[{idx[0] + 1}/{n}] {Path(paths[idx[0]]).name}\n"
            f"peak=max|V-b|={p:.3e} V   integral={ing_s} V·s"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "n":
            idx[0] = (idx[0] + 1) % n
            redraw()
        elif event.key == "p":
            idx[0] = (idx[0] - 1) % n
            redraw()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    plot_histograms(peaks, integrals, bins=bins, save=save)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review waveforms and histogram peak / integrated signal (no time-at-peak hist)."
    )
    parser.add_argument(
        "dir",
        type=Path,
        help="Directory containing *.csv files (e.g. outputs/run0001/CH1/csv)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not step through waveforms; only show histograms at the end.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=HIST_BINS,
        help=f"Histogram bins (default: {HIST_BINS})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save PNGs into this directory",
    )
    parser.add_argument(
        "--print-time-ranges",
        action="store_true",
        help="Print min/max time per CSV and hscale from matching json/ (if any), then continue.",
    )
    args = parser.parse_args()

    input_dir = args.dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(input_dir.glob("*.csv"))
    if not files:
        print(f"No *.csv in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} CSV file(s) in {input_dir}")

    if args.print_time_ranges:
        _print_time_ranges_and_json_hint(files)

    paths_ok, peaks, integrals, waveforms = collect_metrics(
        files,
        baseline=BASELINE_V,
        t0=INTEGRAL_T_MIN,
        t1=INTEGRAL_T_MAX,
    )

    if not paths_ok:
        print("No valid waveforms.", file=sys.stderr)
        sys.exit(1)

    save_dir = args.save
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.no_interactive:
        plot_histograms(peaks, integrals, bins=args.bins, save=save_dir)
        plt.show()
    else:
        run_interactive(paths_ok, waveforms, peaks, integrals, bins=args.bins, save=save_dir)


if __name__ == "__main__":
    main()
