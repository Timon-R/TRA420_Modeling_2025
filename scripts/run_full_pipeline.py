"""Run the full TRA420 modelling pipeline using configuration defaults.

Steps executed:
1. Aggregate calc_emissions across all configured countries (writing resources and results outputs).
2. Run FaIR climate simulations.
3. Apply pattern scaling to produce country-level temperatures.
4. Evaluate air-pollution impacts using the aggregated emission results.
5. Compute economic outputs (SCC / damages) via the configured settings.

The script assumes it is executed from the project root (where ``config.yaml``
resides). Logging is configured globally so progress is visible when "hitting
play" from the IDE.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_summary  # noqa: E402
import run_calc_emissions_all  # noqa: E402
import run_fair_scenarios  # noqa: E402
import run_pattern_scaling  # noqa: E402
import run_scc  # noqa: E402

from air_pollution import run_from_config as run_air_pollution  # noqa: E402

LOGGER = logging.getLogger("pipeline")


def _load_scenario_filter() -> list[str]:
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        return []
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle) or {}
    countries_cfg = cfg.get("calc_emissions", {}).get("countries", {})
    scenarios = countries_cfg.get("scenarios", [])
    return [s for s in scenarios if s]


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")


def _run_economic_module() -> None:
    LOGGER.info("Running economic module (SCC/damages)")
    argv_backup = sys.argv.copy()
    try:
        sys.argv = ["run_scc.py"]
        run_scc.main()
    except FileNotFoundError as exc:
        LOGGER.warning("Skipping economic module due to missing input: %s", exc)
    finally:
        sys.argv = argv_backup


def _run_summary_outputs() -> None:
    LOGGER.info("Generating results summary")
    argv_backup = sys.argv.copy()
    try:
        sys.argv = ["generate_summary.py"]
        generate_summary.main()
    finally:
        sys.argv = argv_backup


def run_pipeline(countries: Iterable[str] | None = None) -> None:
    LOGGER.info("Starting full pipeline")
    scenario_filter = _load_scenario_filter()
    aggregated_results = run_calc_emissions_all.run_all_countries(
        countries=countries,
        scenarios=scenario_filter,
        mirror_to_root=False,
    )
    if scenario_filter:
        LOGGER.info("Scenario filter: %s", ", ".join(scenario_filter))
    LOGGER.info(
        "Aggregated emission scenarios: %s",
        ", ".join(name for name in aggregated_results if name != "baseline"),
    )

    LOGGER.info("Running climate module")
    run_fair_scenarios.main()

    LOGGER.info("Applying pattern scaling")
    run_pattern_scaling.main()

    LOGGER.info("Running air-pollution module")
    run_air_pollution(emission_results=aggregated_results)

    _run_economic_module()
    _run_summary_outputs()
    LOGGER.info("Pipeline complete")


def main() -> None:
    _configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
