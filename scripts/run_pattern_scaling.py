from pattern_scaling import DEFAULT_CONFIG_PATH, get_scaling_factors, load_config, scale_results

config = load_config(DEFAULT_CONFIG_PATH)
sf = get_scaling_factors(config)
scale_results(config, sf)
