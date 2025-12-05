"""Generate additional emission-savings plots.

Outputs:
  - Bar chart of total CO₂ saved (All countries) at selected years with upper/lower error bars.
  - Bar chart of CO₂ saved per country at selected years with upper/lower error bars.

By default, the script uses the repository config to locate emission outputs and
plots to ``results/<run>/summary/additional_plots``. Positive values indicate
emissions avoided (deltas are sign-flipped).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from calc_emissions.constants import BASE_DEMAND_CASE, BASE_MIX_CASE
from config_paths import apply_results_run_directory, get_config_path, get_results_run_directory

LOGGER = logging.getLogger("additional_plots")
ROOT = Path(__file__).resolve().parents[1]


def _load_config(config_path: Path) -> dict:
    with config_path.open() as handle:
        return yaml.safe_load(handle) or {}


def _resolve_paths(config: Mapping[str, object], mix: str) -> tuple[Path, Path]:
    """Return (aggregate_root, per_country_root) with run directory applied."""
    run_directory = get_results_run_directory(config)
    calc_cfg = config.get("calc_emissions", {}) or {}
    countries_cfg = calc_cfg.get("countries", {}) or {}

    agg_cfg = countries_cfg.get("aggregate_output_directory", "results/emissions/All_countries")
    agg_root = Path(agg_cfg)
    if not agg_root.is_absolute():
        agg_root = (ROOT / agg_root).resolve()
    agg_root = apply_results_run_directory(agg_root, run_directory, repo_root=ROOT) / mix

    per_country_cfg = countries_cfg.get("resources_root", "results/emissions")
    per_country_root = Path(per_country_cfg)
    if not per_country_root.is_absolute():
        per_country_root = (ROOT / per_country_cfg).resolve()
    per_country_root = apply_results_run_directory(
        per_country_root, run_directory, repo_root=ROOT
    ) / mix

    return agg_root, per_country_root


def _load_deltas(co2_path: Path, demand_cases: Iterable[str], baseline_case: str) -> pd.DataFrame:
    if not co2_path.exists():
        raise FileNotFoundError(f"Missing CO₂ file: {co2_path}")
    df = pd.read_csv(co2_path, comment="#")
    if "year" not in df.columns:
        raise ValueError(f"'year' column missing in {co2_path}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    expected = [f"delta_{d}" for d in demand_cases if d != baseline_case]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise KeyError(f"Missing delta columns in {co2_path}: {missing}")
    return df[["year"] + expected]


def _savings_at_year(
    df: pd.DataFrame, year: int, mean_case: str, lower_case: str, upper_case: str
) -> tuple[float, float, float]:
    row = df[df["year"] == year]
    if row.empty:
        raise KeyError(f"Year {year} not found in CO₂ data.")
    mean = -float(row[f"delta_{mean_case}"].iloc[0])
    lower = -float(row[f"delta_{lower_case}"].iloc[0])
    upper = -float(row[f"delta_{upper_case}"].iloc[0])
    err_low = max(mean - lower, 0.0)
    err_high = max(upper - mean, 0.0)
    return mean, err_low, err_high


def _cumulative_savings(
    df: pd.DataFrame, up_to_year: int, mean_case: str, lower_case: str, upper_case: str
) -> tuple[float, float, float]:
    window = df[df["year"] <= up_to_year]
    if window.empty:
        raise KeyError(f"No data up to year {up_to_year} in CO₂ data.")
    mean = -float(window[f"delta_{mean_case}"].sum())
    lower = -float(window[f"delta_{lower_case}"].sum())
    upper = -float(window[f"delta_{upper_case}"].sum())
    err_low = max(mean - lower, 0.0)
    err_high = max(upper - mean, 0.0)
    return mean, err_low, err_high


def _plot_total_saved(
    agg_root: Path,
    demand_cases: Iterable[str],
    baseline_case: str,
    years: list[int],
    output_dir: Path,
    mix: str,
    cumulative_years: list[int] | None = None,
) -> None:
    mean_case = "scen1_mean"
    lower_case = "scen1_lower"
    upper_case = "scen1_upper"
    co2_path = agg_root / "co2.csv"
    df = _load_deltas(co2_path, demand_cases, baseline_case)

    means: list[float] = []
    err_lows: list[float] = []
    err_highs: list[float] = []
    for year in years:
        m, el, eh = _savings_at_year(df, year, mean_case, lower_case, upper_case)
        means.append(m)
        err_lows.append(el)
        err_highs.append(eh)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(years))
    ax.bar(x, means, yerr=[err_lows, err_highs], capsize=6, color="#2a6fdb")
    ax.set_xticks(x, [str(y) for y in years])
    ax.set_ylabel("Mt CO₂ saved")
    ax.set_title(f"Annual CO₂ Saved (All countries) — {mix}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{mix}_total_co2_saved.png", format="png")
    plt.close(fig)

    if cumulative_years:
        cum_means: list[float] = []
        cum_err_lows: list[float] = []
        cum_err_highs: list[float] = []
        for year in cumulative_years:
            m, el, eh = _cumulative_savings(df, year, mean_case, lower_case, upper_case)
            cum_means.append(m)
            cum_err_lows.append(el)
            cum_err_highs.append(eh)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(cumulative_years))
        ax.bar(x, cum_means, yerr=[cum_err_lows, cum_err_highs], capsize=6, color="#7b68ee")
        ax.set_xticks(x, [str(y) for y in cumulative_years])
        ax.set_ylabel("Mt CO₂ saved (cumulative)")
        ax.set_title(f"Cumulative CO₂ Saved (All countries) — {mix}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mix}_total_co2_saved_cumulative.png", format="png")
        plt.close(fig)


def _plot_country_saved(
    per_country_root: Path,
    demand_cases: Iterable[str],
    baseline_case: str,
    years: list[int],
    output_dir: Path,
    mix: str,
    cumulative_years: list[int] | None = None,
) -> None:
    mean_case = "scen1_mean"
    lower_case = "scen1_lower"
    upper_case = "scen1_upper"
    country_dirs = sorted([p for p in per_country_root.iterdir() if p.is_dir()])
    if not country_dirs:
        LOGGER.warning("No country directories found under %s", per_country_root)
        return

    for year in years:
        labels: list[str] = []
        means: list[float] = []
        err_lows: list[float] = []
        err_highs: list[float] = []
        for cdir in country_dirs:
            co2_path = cdir / "co2.csv"
            try:
                df = _load_deltas(co2_path, demand_cases, baseline_case)
                m, el, eh = _savings_at_year(df, year, mean_case, lower_case, upper_case)
            except Exception as exc:
                LOGGER.warning("Skipping %s: %s", co2_path, exc)
                continue
            labels.append(cdir.name)
            means.append(m)
            err_lows.append(el)
            err_highs.append(eh)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(labels))
        ax.bar(x, means, yerr=[err_lows, err_highs], capsize=5, color="#2aa05a")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Mt CO₂ saved")
        ax.set_title(f"Annual CO₂ Saved by Country — {year}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mix}_country_co2_saved_{year}.png", format="png")
        plt.close(fig)

    if cumulative_years:
        for year in cumulative_years:
            labels_c: list[str] = []
            means_c: list[float] = []
            err_lows_c: list[float] = []
            err_highs_c: list[float] = []
            for cdir in country_dirs:
                co2_path = cdir / "co2.csv"
                try:
                    df = _load_deltas(co2_path, demand_cases, baseline_case)
                    m, el, eh = _cumulative_savings(df, year, mean_case, lower_case, upper_case)
                except Exception as exc:
                    LOGGER.warning("Skipping cumulative %s: %s", co2_path, exc)
                    continue
                labels_c.append(cdir.name)
                means_c.append(m)
                err_lows_c.append(el)
                err_highs_c.append(eh)
            if not labels_c:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = range(len(labels_c))
            ax.bar(x, means_c, yerr=[err_lows_c, err_highs_c], capsize=5, color="#7b68ee")
            ax.set_xticks(x, labels_c, rotation=20, ha="right")
            ax.set_ylabel("Mt CO₂ saved (cumulative)")
            ax.set_title(f"CO₂ Saved by Country ({year}) — cumulative — {mix}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(output_dir / f"{mix}_country_co2_saved_cumulative_{year}.png", format="png")
            plt.close(fig)


def _plot_country_pairwise(
    per_country_root: Path,
    demand_cases: Iterable[str],
    baseline_case: str,
    years: list[int],
    output_dir: Path,
    mix: str,
) -> None:
    """One plot per country comparing the specified years (e.g., 2030 vs 2050)."""
    mean_case = "scen1_mean"
    lower_case = "scen1_lower"
    upper_case = "scen1_upper"
    target_years = sorted(years)
    country_dirs = sorted([p for p in per_country_root.iterdir() if p.is_dir()])
    if not country_dirs:
        return

    for cdir in country_dirs:
        co2_path = cdir / "co2.csv"
        try:
            df = _load_deltas(co2_path, demand_cases, baseline_case)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", co2_path, exc)
            continue

        means: list[float] = []
        err_lows: list[float] = []
        err_highs: list[float] = []
        years_found: list[int] = []
        for year in target_years:
            try:
                m, el, eh = _savings_at_year(df, year, mean_case, lower_case, upper_case)
            except Exception as exc:
                LOGGER.warning("Skipping %s %s: %s", cdir.name, year, exc)
                continue
            means.append(m)
            err_lows.append(el)
            err_highs.append(eh)
            years_found.append(year)

        if not years_found:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(years_found))
        ax.bar(x, means, yerr=[err_lows, err_highs], capsize=6, color="#db8b2a")
        ax.set_xticks(x, [str(y) for y in years_found])
        ax.set_ylabel("Mt CO₂ saved")
        ax.set_title(f"CO₂ Saved — {cdir.name}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fname = f"{cdir.name}_{mix}_co2_saved.png"
        fig.savefig(output_dir / fname, format="png")
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate additional emission-savings plots.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument(
        "--mix",
        nargs="*",
        default=None,
        help=(
            "Mix scenario(s) to plot (default: calc_emissions.countries.mix_scenarios "
            "or baseline mix)."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[2030, 2050],
        help="Years to plot (default: 2030 2050).",
    )
    args = parser.parse_args()

    config_path = get_config_path(Path(args.config))
    config = _load_config(config_path)

    calc_cfg = config.get("calc_emissions", {}) or {}
    countries_cfg = calc_cfg.get("countries", {}) or {}
    demand_cases = countries_cfg.get("demand_scenarios", []) or []
    if not demand_cases:
        raise ValueError("calc_emissions.countries.demand_scenarios must be configured.")
    baseline_case = (
        str(countries_cfg.get("baseline_demand_case", BASE_DEMAND_CASE)).strip() or BASE_DEMAND_CASE
    )
    mix_candidates: Sequence[str] = (
        args.mix
        if args.mix
        else countries_cfg.get("mix_scenarios")
        or calc_cfg.get("mix_scenarios")
        or [countries_cfg.get("baseline_mix_case") or BASE_MIX_CASE]
    )
    mixes = [str(m).strip() for m in mix_candidates if str(m).strip()]
    if not mixes:
        mixes = [BASE_MIX_CASE]

    base_output_dir = apply_results_run_directory(
        (ROOT / "results" / "summary" / "additional_plots").resolve(),
        get_results_run_directory(config),
        repo_root=ROOT,
    )

    years = sorted({int(y) for y in args.years})

    for mix in mixes:
        agg_root, per_country_root = _resolve_paths(config, mix)
        output_dir = base_output_dir / mix
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Using aggregate path: %s", agg_root)
        LOGGER.info("Using per-country path: %s", per_country_root)
        LOGGER.info("Writing plots to: %s", output_dir)

        _plot_total_saved(
            agg_root,
            demand_cases,
            baseline_case,
            years,
            output_dir,
            mix,
            cumulative_years=[2030, 2050],
        )
        _plot_country_saved(
            per_country_root,
            demand_cases,
            baseline_case,
            years,
            output_dir,
            mix,
            cumulative_years=[2030, 2050],
        )
        _plot_country_pairwise(
            per_country_root, demand_cases, baseline_case, years, output_dir / "per_country", mix
        )


if __name__ == "__main__":
    main()
