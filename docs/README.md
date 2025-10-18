# Project Documentation

This folder collects module-level documentation, equations, and configuration
references for the TRA420 modeling pipeline. Each document focuses on one
package and highlights the key inputs, outputs, and assumptions that drive the
calculations.

- `calc_emissions.md` – electricity demand and technology mix assumptions,
  emission factor conversions, and delta CSV generation.
- `climate_module.md` – FaIR wrappers, climate parameter presets, and emission
  adjustment mechanics.
- `air_pollution.md` – concentration-to-health impact linkage, configuration of
  risk coefficients, baseline mortality, and weighting schemes.
- `economic_module.md` – social cost of carbon workflow, damage functions, and
  discounting methods.
- `pattern_scaling.md` – pattern scaling pipeline, required inputs, and output
  structure for country-level climate trajectories.

Refer to the module documents when creating scenarios, extending the pipeline,
or validating scientific assumptions.
