"""
Plot total CO2 for Reference / WEM / WAM scenarios from results/emissions.

- Looks for CSVs named `co2.csv` under results/emissions.
- Prefers aggregated files under results/emissions/All_countries/<mix>/co2.csv.
- Plots absolute_baseline (dotted), absolute_upper_bound and absolute_lower_bound (solid),
  shades the area between upper and lower. Only years <= 2100 are shown.

Usage (from repo root, macOS terminal):
  python scripts/plot_co2_scenarios.py
  python scripts/plot_co2_scenarios.py --out plots/co2_scenarios.png
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

MIXES = ["Reference", "WEM", "WAM"]
DEFAULT_OUT = Path("plots/co2_scenarios.png")


def find_co2_csv(mix: str) -> Path | None:
    # Preferred location: aggregated results
    pref = Path("results/emissions/All_countries") / mix / "co2.csv"
    if pref.exists():
        return pref
    # Fallback: any co2.csv whose path contains the mix name as a directory
    for p in Path("results/emissions").rglob("co2.csv"):
        parts = [part.lower() for part in p.parts]
        if mix.lower() in parts:
            return p
    return None


def load_co2_df(path: Path) -> pd.DataFrame:
    # Read CSV, skip comment lines (unit comment)
    df = pd.read_csv(path, comment="#")
    if "year" not in df.columns:
        # assume first column is year
        df = df.rename(columns={df.columns[0]: "year"})
    df = df.set_index("year")
    # keep numeric and drop NaNs
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


def main(out_path: Path):
    found_any = False
    plt.figure(figsize=(10, 6))
    cmap = {"Reference": "#2a6fdb", "WEM": "#2aa05a", "WAM": "#db8b2a"}  # similar tone palette

    for mix in MIXES:
        csv_path = find_co2_csv(mix)
        if csv_path is None:
            print(f"[WARN] No co2.csv found for mix '{mix}' under results/emissions; skipping.", file=sys.stderr)
            continue

        df = load_co2_df(csv_path)
        # restrict to year <= 2100
        df = df[df.index <= 2100]
        if df.empty:
            print(f"[WARN] File {csv_path} contains no years <= 2100; skipping.", file=sys.stderr)
            continue

        found_any = True
        years = df.index.values
        color = cmap.get(mix, None) or None

        # columns expected: absolute_baseline, absolute_upper_bound, absolute_lower_bound
        baseline_col = "absolute_baseline"
        up_col = "absolute_upper_bound"
        low_col = "absolute_lower_bound"

        if baseline_col in df.columns:
            plt.plot(years, df[baseline_col], linestyle=":", color=color, label=f"{mix} baseline")
        else:
            print(f"[WARN] {baseline_col} not in {csv_path}; baseline not plotted.", file=sys.stderr)

        if up_col in df.columns and low_col in df.columns:
            plt.plot(years, df[up_col], linestyle="-", color=color, alpha=0.9, label=f"{mix} upper")
            plt.plot(years, df[low_col], linestyle="-", color=color, alpha=0.9, label=f"{mix} lower")
            # fill between lower and upper
            plt.fill_between(years, df[low_col], df[up_col], color=color, alpha=0.12)
        else:
            print(f"[WARN] upper/lower columns missing in {csv_path}; upper/lower not plotted.", file=sys.stderr)

    if not found_any:
        print("[ERROR] No matching co2.csv files found for any of the mixes.", file=sys.stderr)
        return

    plt.xlabel("Year")
    plt.ylabel("CO₂ emissions (Mt/year)")
    plt.title("Total CO₂ emissions — Reference / WEM / WAM (<= 2100)")
    # Build a single legend (avoid duplicate labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize="small")
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO2 scenarios from results/emissions")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT, help="Output image path")
    args = parser.parse_args()
    main(args.out)