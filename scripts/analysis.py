#!/usr/bin/env python3
"""
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 24,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'axes.axisbelow': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
})

threshold = 0.0 # V

def read_csv_2cols(filepath: Path, skiprows: int = 1):
    """
    Read a CSV file and return the first column as time
    and the second column as voltage.
    glass-gem CSV has a header line (e.g. # time_s,voltage_V); skiprows skips it.
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=skiprows, comments="#")

    if data.ndim == 1:
        if len(data) < 2:
            raise ValueError("CSV does not have at least 2 columns.")
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError("CSV does not have at least 2 columns.")

    time = data[:, 0]
    voltage = data[:, 1]
    return time, voltage


def main():
    parser = argparse.ArgumentParser(
        description="Make histograms from many CSV files (peak voltage; optional legacy time/2D)."
    )
    parser.add_argument("dir", help="Directory containing CSV files")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for histograms")
    parser.add_argument("--skiprows", type=int, default=1,
                        help="Leading rows to skip (default 1 for glass-gem # header; use 0 if no header)")
    parser.add_argument("--legacy-time", action="store_true",
                        help="Also plot time-at-max and 2D time vs voltage (old behavior)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PNG")
    parser.add_argument("--outdir", default="peak_hist_output",
                        help="Output directory for saved figures")
    args = parser.parse_args()

    input_dir = Path(args.dir).expanduser().resolve()

    print(f"Input directory : {input_dir}")
    print(f"Directory exists: {input_dir.is_dir()}")

    if not input_dir.is_dir():
        print("Error: input directory does not exist.")
        return

    files = sorted(input_dir.glob("*.csv"))

    print(f"Found {len(files)} CSV files")

    if not files:
        print(f"No CSV files found in: {input_dir}")
        return

    max_times = []
    max_voltages = []
    skipped_files = []

    for filepath in tqdm(files, desc="Processing CSVs"):
        try:
            time, voltage = read_csv_2cols(filepath, skiprows=args.skiprows)

            if len(time) == 0:
                raise ValueError("Empty data")

            idx_max = np.argmax(voltage)
            if voltage[idx_max] > threshold:
              max_times.append(time[idx_max])
              max_voltages.append(voltage[idx_max])

        except Exception as e:
            skipped_files.append((str(filepath), str(e)))

    if len(max_times) == 0:
        print("No valid CSV data found.")
        return

    max_times = np.asarray(max_times)
    max_voltages = np.asarray(max_voltages)

    print(f"Valid files  : {len(max_times)}")
    print(f"Skipped files: {len(skipped_files)}")

    if skipped_files:
        print("\n--- Skipped files ---")
        for filepath, reason in skipped_files[:20]:
            print(f"{filepath}: {reason}")
        if len(skipped_files) > 20:
            print(f"... and {len(skipped_files) - 20} more")

    plt.figure(figsize=(8, 5))
    plt.hist(max_voltages, bins=args.bins, histtype='step')
    plt.xlabel("Maximum voltage")
    plt.ylabel("Counts")
    plt.tight_layout()

    if args.save:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / "peak_voltage_hist.png", dpi=200)

    if args.legacy_time:
        plt.figure(figsize=(8, 5))
        plt.hist(max_times, bins=args.bins)
        plt.xlabel("Time at maximum voltage")
        plt.ylabel("Counts")
        plt.title("Peak time distribution")
        plt.tight_layout()

        if args.save:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / "peak_time_hist.png", dpi=200)

        plt.figure(figsize=(8, 6))
        plt.hist2d(max_times, max_voltages, bins=args.bins)
        plt.xlabel("Time at maximum voltage")
        plt.ylabel("Maximum voltage")
        plt.title("Peak time vs peak voltage")
        plt.colorbar(label="Counts")
        plt.tight_layout()

        if args.save:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / "peak_time_vs_voltage_2d.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
