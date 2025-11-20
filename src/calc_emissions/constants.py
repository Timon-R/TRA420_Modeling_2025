from __future__ import annotations

BASE_DEMAND_CASE = "base_demand"

POLLUTANTS: dict[str, dict[str, object]] = {
    "co2": {
        "aliases": {
            "co2_mt_per_twh": 1.0,
            "co2_kt_per_twh": 1e-3,
            "co2_gwh": 1e-3,
            "co2_kg_per_mwh": 1.0,
            "co2_kg_per_kwh": 1.0,
            "co2_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "sox": {
        "aliases": {
            "sox_mt_per_twh": 1.0,
            "sox_kt_per_twh": 1e-3,
            "sox_kg_per_mwh": 1.0,
            "sox_kg_per_kwh": 1.0,
            "sox_g_per_kwh": 1e-6,
            "so2_kt_per_twh": 1e-3,
            "so2_kg_per_kwh": 1.0,
        },
        "unit": "Mt",
    },
    "nox": {
        "aliases": {
            "nox_mt_per_twh": 1.0,
            "nox_kt_per_twh": 1e-3,
            "nox_kg_per_mwh": 1.0,
            "nox_kg_per_kwh": 1.0,
            "nox_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "pm25": {
        "aliases": {
            "pm25_mt_per_twh": 1.0,
            "pm25_kt_per_twh": 1e-3,
            "pm25_kg_per_mwh": 1.0,
            "pm25_kg_per_kwh": 1.0,
            "pm25_g_per_kwh": 1e-6,
        },
        "unit": "Mt",
    },
    "gwp100": {
        "aliases": {
            "gwp100_mt_per_twh": 1.0,
            "gwp100_kt_per_twh": 1e-3,
            "gwp100_kg_per_mwh": 1.0,
            "gwp100_kg_per_kwh": 1.0,
            "gwp100_g_per_kwh": 1e-6,
        },
        "unit": "Mt CO2eq",
    },
}
