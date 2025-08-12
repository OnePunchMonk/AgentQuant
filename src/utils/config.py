import yaml
from pathlib import Path

def load_config():
    """Loads the config.yaml file."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found at the project root.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load config once and make it available for import
config = load_config()