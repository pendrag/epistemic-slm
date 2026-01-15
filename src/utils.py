def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must .yaml or .yml")
