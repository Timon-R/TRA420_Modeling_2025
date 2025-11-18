"""Run the full TRA420 modelling pipeline with optional scenario suites."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate_summary  # noqa: E402
import run_calc_emissions_all  # noqa: E402
import run_scc  # noqa: E402

from air_pollution import run_from_config as run_air_pollution  # noqa: E402
from config_paths import (  # noqa: E402
    CONFIG_ENV_VAR,
    apply_results_run_directory,
    get_config_path,
    get_results_run_directory,
    join_run_directory,
    sanitize_run_directory,
    set_config_root,
)

LOGGER = logging.getLogger("pipeline")

_CLIMATE_SUFFIX_PATTERN = re.compile(r"_(ssp[0-9]{2,3}[a-z0-9]*)$", re.IGNORECASE)


def _load_scenario_filter(config: Mapping[str, object]) -> list[str]:
    cfg = config or {}
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


def _cleanup_default_climate_dirs() -> None:
    for relative in ("results/climate", "results/climate_scaled"):
        directory = ROOT / relative
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            next(directory.iterdir())
        except StopIteration:
            try:
                directory.rmdir()
                LOGGER.debug("Removed unused default directory %s", directory.relative_to(ROOT))
            except OSError:
                LOGGER.debug("Unable to remove directory %s", directory, exc_info=True)
        except OSError:
            LOGGER.debug("Unable to inspect directory %s", directory, exc_info=True)


def run_pipeline(countries: Iterable[str] | None = None) -> None:
    LOGGER.info("Starting full pipeline")
    config = _read_root_config()
    scenario_filter = _load_scenario_filter(config)
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
    fair_module = importlib.import_module("run_fair_scenarios")
    fair_module = importlib.reload(fair_module)
    fair_module.main()

    LOGGER.info("Applying pattern scaling")
    pattern_module = importlib.import_module("run_pattern_scaling")
    pattern_module = importlib.reload(pattern_module)
    pattern_module.main()

    LOGGER.info("Running air-pollution module")
    run_air_pollution(emission_results=aggregated_results)

    _run_economic_module()
    _run_summary_outputs()
    _cleanup_default_climate_dirs()
    LOGGER.info("Pipeline complete")


def _read_root_config(path: Path | None = None) -> dict:
    config_path = path if path is not None else get_config_path(ROOT / "config.yaml")
    config = {}
    if config_path.exists():
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}
    set_config_root(config, config_path.parent)
    return config


def _deep_merge(target: MutableMapping[str, object], overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)  # type: ignore[index]
        else:
            target[key] = deepcopy(value)


def _apply_run_directory(config: MutableMapping[str, object], run_directory: str | None) -> None:
    if not run_directory:
        results_cfg = config.get("results")
        if isinstance(results_cfg, MutableMapping):
            results_cfg.pop("run_directory", None)
        return
    results_cfg = config.setdefault("results", {})
    if isinstance(results_cfg, MutableMapping):
        results_cfg["run_directory"] = run_directory
    _rewrite_prefixed_paths(config, "resources", run_directory)


def _rewrite_prefixed_paths(obj: object, prefix: str, run_dir: str) -> object:
    prefix_norm = prefix.strip().rstrip("/")
    if not prefix_norm:
        return obj
    root_prefix = f"{prefix_norm}/"

    def _rewrite(value: object) -> object:
        if isinstance(value, str) and value.startswith(root_prefix):
            remainder = value[len(root_prefix) :].lstrip("/")
            new_path = (Path(prefix_norm) / run_dir / Path(remainder)).as_posix()
            return new_path
        return value

    if isinstance(obj, MutableMapping):
        for key, val in list(obj.items()):
            obj[key] = _rewrite_prefixed_paths(val, prefix, run_dir)
        return obj
    if isinstance(obj, list):
        return [_rewrite_prefixed_paths(item, prefix, run_dir) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrite_prefixed_paths(item, prefix, run_dir) for item in obj)
    return _rewrite(obj)


def _split_climate_label(label: str) -> tuple[str, str | None]:
    match = _CLIMATE_SUFFIX_PATTERN.search(label)
    if not match:
        return label, None
    base = label[: match.start()].rstrip("_") or label
    return base, match.group(1)


def _value_for_year(data: object | None, year: int) -> float:
    if not isinstance(data, Mapping):
        return math.nan
    key_candidates = [str(year), year]
    for key in key_candidates:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return math.nan
    return math.nan


def _infer_metric_years(metrics: Mapping[str, object]) -> list[int]:
    keys: set[int] = set()
    for metric_key in (
        "emission_delta_mt",
        "temperature_delta_c",
        "mortality_delta",
        "mortality_value_delta",
    ):
        series = metrics.get(metric_key)
        if not isinstance(series, Mapping):
            continue
        for candidate in series:
            try:
                value = int(float(str(candidate)))
            except (TypeError, ValueError):
                continue
            keys.add(value)
    return sorted(keys)


def _determine_summary_fields(records: list[dict[str, object]]) -> list[str]:
    preferred_order = [
        "suite",
        "suite_scenario",
        "model_scenario",
        "policy_scenario",
        "climate_label",
        "year",
        "run_directory",
        "emission_delta_mt",
        "temperature_delta_c",
        "mortality_delta",
        "mortality_percent",
        "mortality_baseline",
        "mortality_value_delta",
    ]
    seen = set()
    ordered = []
    for field in preferred_order:
        if any(field in record for record in records):
            ordered.append(field)
            seen.add(field)
    dynamic = sorted(key for record in records for key in record if key not in seen)
    return ordered + dynamic


def _extract_summary_rows(
    summary_json_path: str,
    suite_name: str,
    suite_scenario: str,
    run_directory: str | None,
) -> list[dict[str, object]]:
    if not summary_json_path:
        return []
    path = Path(summary_json_path)
    if not path.exists():
        LOGGER.warning("Summary JSON missing for scenario '%s': %s", suite_scenario, path)
        return []
    data = json.loads(path.read_text())
    scenario_entries = data.get("scenarios") or {}
    base_methods = [str(method) for method in data.get("methods") or []]
    default_years = [int(year) for year in data.get("years") or []]
    rows: list[dict[str, object]] = []
    for model_scenario, metrics in scenario_entries.items():
        policy_label, climate_label = _split_climate_label(model_scenario)
        years = list(default_years) or _infer_metric_years(metrics)
        method_labels = list(base_methods)
        for year in years:
            row: dict[str, object] = {
                "suite": suite_name,
                "suite_scenario": suite_scenario,
                "model_scenario": model_scenario,
                "policy_scenario": policy_label,
                "year": year,
                "run_directory": run_directory or "",
            }
            if climate_label:
                row["climate_label"] = climate_label
            emission_delta = metrics.get("emission_delta_mt")
            temperature_delta = metrics.get("temperature_delta_c")
            mortality_delta = metrics.get("mortality_delta")
            mortality_percent = metrics.get("mortality_percent")
            mortality_baseline = metrics.get("mortality_baseline")
            mortality_value = metrics.get("mortality_value_delta")
            row["emission_delta_mt"] = _value_for_year(emission_delta, year)
            row["temperature_delta_c"] = _value_for_year(temperature_delta, year)
            row["mortality_delta"] = _value_for_year(mortality_delta, year)
            row["mortality_percent"] = _value_for_year(mortality_percent, year)
            row["mortality_baseline"] = _value_for_year(mortality_baseline, year)
            row["mortality_value_delta"] = _value_for_year(mortality_value, year)
            scc_map = metrics.get("scc_usd_per_tco2") or {}
            for key in scc_map:
                label = str(key)
                if label not in method_labels:
                    method_labels.append(label)
            damages_map = metrics.get("damages_usd") or {}
            for key in damages_map:
                label = str(key)
                if label not in method_labels:
                    method_labels.append(label)
            for method in method_labels:
                method_series = scc_map.get(method)
                column = f"scc_usd_per_tco2_{method}"
                row[column] = _value_for_year(method_series, year)
            for method in method_labels:
                method_series = damages_map.get(method)
                column = f"damages_usd_{method}"
                row[column] = _value_for_year(method_series, year)
            averages = metrics.get("scc_average") or {}
            for method, value in averages.items():
                row[f"scc_average_{method}"] = float(value)
            totals = metrics.get("damage_total_usd") or {}
            for method, value in totals.items():
                row[f"damage_total_usd_{method}"] = float(value)
            rows.append(row)
    return rows


@contextmanager
def _use_config_path(path: Path):
    previous = os.environ.get(CONFIG_ENV_VAR)
    os.environ[CONFIG_ENV_VAR] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CONFIG_ENV_VAR, None)
        else:
            os.environ[CONFIG_ENV_VAR] = previous


@contextmanager
def _temporary_config(config: Mapping[str, object], label: str, base_root: Path):
    with tempfile.TemporaryDirectory(prefix=f"tra420_run_{label}_") as tmp_dir:
        tmp_path = Path(tmp_dir) / "config.yaml"
        config_copy = deepcopy(config)
        set_config_root(config_copy, base_root)
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_copy, handle, sort_keys=False)
        with _use_config_path(tmp_path):
            yield tmp_path


def _execute_config_run(
    config: Mapping[str, object], label: str, base_root: Path
) -> dict[str, str]:
    LOGGER.info("Executing pipeline run '%s'", label)
    with _temporary_config(config, label, base_root):
        run_pipeline()
    run_directory = (
        config.get("results", {}).get("run_directory") if isinstance(config, Mapping) else None
    )
    summary_cfg = (
        config.get("results", {}).get("summary", {}) if isinstance(config, Mapping) else {}
    )
    summary_dir_cfg = summary_cfg.get("output_directory", "results/summary")
    summary_dir = Path(summary_dir_cfg)
    summary_dir = apply_results_run_directory(summary_dir, run_directory, repo_root=ROOT)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / "summary.json"
    summary_txt = summary_dir / "summary.txt"
    return {
        "run_directory": run_directory or "",
        "summary_json": str(summary_json) if summary_json.exists() else "",
        "summary_txt": str(summary_txt) if summary_txt.exists() else "",
    }


def _run_single_run(
    config: Mapping[str, object],
    output_subdir: str | None,
    base_root: Path,
) -> None:
    config_copy = deepcopy(config)
    config_copy.setdefault("run", {})["mode"] = "normal"
    base_run_dir = get_results_run_directory(config_copy)
    final_run_dir = join_run_directory(base_run_dir, output_subdir)
    if output_subdir:
        config_copy.setdefault("run", {})["output_subdir"] = output_subdir
    _apply_run_directory(config_copy, final_run_dir)
    if final_run_dir:
        LOGGER.info("Running single configuration (results under 'results/%s')", final_run_dir)
    else:
        LOGGER.info("Running single configuration (default results folder)")
    _execute_config_run(config_copy, output_subdir or "default", base_root)


def _resolve_suite_directory(
    base_config: Mapping[str, object],
    run_cfg: Mapping[str, object],
    suite_name: str,
) -> str | None:
    base_run_dir = get_results_run_directory(base_config)
    suite_base = sanitize_run_directory(run_cfg.get("output_subdir"))
    return join_run_directory(base_run_dir, suite_base, suite_name)


def _run_scenario_suite(
    base_config: Mapping[str, object],
    run_cfg: Mapping[str, object],
    scenario_path: Path,
    base_root: Path,
) -> None:
    with scenario_path.open() as handle:
        scenario_overrides = yaml.safe_load(handle) or {}
    if not isinstance(scenario_overrides, Mapping):
        raise ValueError("Scenario file must define a mapping of scenario names to overrides.")

    suite_name = sanitize_run_directory(scenario_path.stem) or "scenarios"
    suite_run_dir = _resolve_suite_directory(base_config, run_cfg, suite_name)
    aggregate_records: list[dict[str, object]] = []

    scenario_items = list(scenario_overrides.items())
    LOGGER.info(
        "Running scenario suite '%s' (%d cases) using %s",
        suite_name,
        len(scenario_items),
        scenario_path,
    )

    for scenario_name, overrides in scenario_items:
        overrides = overrides or {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Scenario '{scenario_name}' overrides must be a mapping.")
        config_copy = deepcopy(base_config)
        _deep_merge(config_copy, overrides)
        config_copy.setdefault("run", {})["mode"] = "normal"
        scenario_run_cfg = (
            overrides.get("run", {}) if isinstance(overrides.get("run"), Mapping) else {}
        )
        scenario_output = scenario_run_cfg.get("output_subdir") or scenario_name
        final_run_dir = join_run_directory(suite_run_dir, scenario_output)
        config_copy.setdefault("run", {})["output_subdir"] = scenario_output
        _apply_run_directory(config_copy, final_run_dir)
        LOGGER.info(
            "Running scenario '%s' (results under 'results/%s')",
            scenario_name,
            final_run_dir or "<default>",
        )
        summary_info = _execute_config_run(config_copy, f"{suite_name}_{scenario_name}", base_root)
        summary_json_path = summary_info.get("summary_json", "")
        rows = _extract_summary_rows(
            summary_json_path,
            suite_name,
            scenario_name,
            summary_info.get("run_directory", final_run_dir or ""),
        )
        if rows:
            aggregate_records.extend(rows)
        else:
            aggregate_records.append(
                {
                    "suite": suite_name,
                    "suite_scenario": scenario_name,
                    "model_scenario": "",
                    "policy_scenario": "",
                    "year": math.nan,
                    "run_directory": summary_info.get("run_directory", final_run_dir or ""),
                }
            )

    suite_rel = suite_run_dir or suite_name
    suite_dir = (ROOT / "results" / suite_rel) if suite_rel else (ROOT / "results")
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = suite_dir / "scenario_suite_summary.csv"
    fieldnames = _determine_summary_fields(aggregate_records)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_records)
    summary_json = suite_dir / "scenario_suite_summary.json"
    summary_json.write_text(json.dumps(aggregate_records, indent=2))
    dest_scenario_file = suite_dir / scenario_path.name
    if not dest_scenario_file.exists():
        shutil.copyfile(scenario_path, dest_scenario_file)
    LOGGER.info(
        "Scenario suite '%s' complete. Aggregate summary: %s",
        suite_name,
        summary_csv.relative_to(ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TRA420 pipeline or scenario suite.")
    parser.add_argument("--config", help="Path to the base configuration file.")
    parser.add_argument(
        "--run-subdir",
        help="Name of the subdirectory under results/ for this run (overrides run.output_subdir).",
    )
    args = parser.parse_args()

    _configure_logging()
    base_config_path = get_config_path(Path(args.config) if args.config else ROOT / "config.yaml")
    base_root = base_config_path.parent.resolve()
    base_config = _read_root_config(base_config_path)

    run_cfg = base_config.get("run", {}) or {}
    run_mode = str(run_cfg.get("mode", "normal")).strip().lower()

    if args.run_subdir:
        run_cfg = dict(run_cfg)
        run_cfg["output_subdir"] = args.run_subdir

    if run_mode == "scenarios":
        scenario_file = run_cfg.get("scenario_file")
        if not scenario_file:
            raise ValueError("run.scenario_file must be provided when run.mode == 'scenarios'.")
        scenario_path = Path(scenario_file)
        if not scenario_path.is_absolute():
            scenario_path = (base_config_path.parent / scenario_path).resolve()
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        _run_scenario_suite(base_config, run_cfg, scenario_path, base_root)
    else:
        output_subdir = args.run_subdir or run_cfg.get("output_subdir")
        _run_single_run(base_config, output_subdir, base_root)


if __name__ == "__main__":
    main()
