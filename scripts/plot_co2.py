from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from calc_emissions import BASE_DEMAND_CASE

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

with CONFIG_PATH.open() as handle:
    ROOT_CONFIG = yaml.safe_load(handle) or {}

COUNTRIES_CFG = ROOT_CONFIG.get("calc_emissions", {}).get("countries", {}) or {}
MIX_CASES = COUNTRIES_CFG.get("mix_scenarios") or ["base_mix", "WEM", "WAM"]
DEMAND_CASES = COUNTRIES_CFG.get("demand_scenarios") or [
    "base_demand",
    "scen1_lower",
    "scen1_upper",
]
BASELINE_DEMAND_CASE = (
    str(COUNTRIES_CFG.get("baseline_demand_case", BASE_DEMAND_CASE)).strip() or BASE_DEMAND_CASE
)
SCENARIO_CASES = [case for case in DEMAND_CASES if case != BASELINE_DEMAND_CASE]

AGGREGATED_ROOT = Path("results/emissions/All_countries")
DEFAULT_OUT = Path("plots/co2_scenarios.png")


def _load_mix_dataframe(mix_case: str) -> pd.DataFrame | None:
    path = AGGREGATED_ROOT / mix_case / "co2.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    if "year" not in df.columns:
        return None
    df = df[df["year"] <= 2100]
    if df.empty:
        return None
    return df.set_index("year")


def main(out_path: Path):
    found_any = False
    plt.figure(figsize=(10, 6))
    cmap = {"base_mix": "#2a6fdb", "WEM": "#2aa05a", "WAM": "#db8b2a"}

    for mix in MIX_CASES:
        df = _load_mix_dataframe(mix)
        if df is None:
            print(f"[WARN] Mix '{mix}' has no usable co2.csv; skipping.", file=sys.stderr)
            continue

        baseline_col = f"absolute_{BASELINE_DEMAND_CASE}"
        if baseline_col not in df.columns:
            print(f"[WARN] Mix '{mix}' missing baseline column '{baseline_col}'.", file=sys.stderr)
            continue

        baseline_series = df[baseline_col]
        years = df.index.values
        color = cmap.get(mix) or None
        plt.plot(years, baseline_series.values, linestyle=":", color=color, label=f"{mix} base")
        found_any = True

        scenario_cols = [f"absolute_{case}" for case in SCENARIO_CASES]
        scenario_series = []
        for case, col in zip(SCENARIO_CASES, scenario_cols, strict=False):
            if col not in df.columns:
                print(f"[WARN] Mix '{mix}' missing column '{col}'.", file=sys.stderr)
                continue
            scenario_series.append((case, df[col]))

        if len(scenario_series) >= 2:
            lower_case, lower_series = scenario_series[0]
            upper_case, upper_series = scenario_series[1]
            lower_vals = (
                lower_series.reindex(years, method=None)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            upper_vals = (
                upper_series.reindex(years, method=None)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
            plt.plot(
                years,
                upper_vals.values,
                linestyle="-",
                color=color,
                alpha=0.9,
                label=f"{mix} {upper_case}",
            )
            plt.plot(
                years,
                lower_vals.values,
                linestyle="-",
                color=color,
                alpha=0.9,
                label=f"{mix} {lower_case}",
            )
            plt.fill_between(years, lower_vals.values, upper_vals.values, color=color, alpha=0.12)
        else:
            for case, series in scenario_series:
                aligned = (
                    series.reindex(years, method=None).fillna(method="ffill").fillna(method="bfill")
                )
                plt.plot(
                    years,
                    aligned.values,
                    linestyle="-",
                    color=color,
                    alpha=0.9,
                    label=f"{mix} {case}",
                )

    if not found_any:
        print(
            "[ERROR] No matching CO₂ mixes found under results/emissions/All_countries.",
            file=sys.stderr,
        )
        return

    plt.xlabel("Year")
    plt.ylabel("CO₂ emissions (Mt/year)")
    plt.title("Total CO₂ emissions (<= 2100)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    plt.legend(by_label.values(), by_label.keys(), loc="best", fontsize="small")
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO₂ scenarios from results/emissions")
    parser.add_argument("--out", "-o", type=Path, default=DEFAULT_OUT, help="Output image path")
    args = parser.parse_args()
    main(args.out)
