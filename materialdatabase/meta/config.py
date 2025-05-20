"""Handles loading and validating the configuration from config.toml."""

# python libraries
from pathlib import Path
import logging

# 3rd party libraries
import toml

# own libraries
from materialdatabase.meta.toml_checker import Config, UserPaths, UserColors

logger = logging.getLogger(__name__)

DUMMY_CONFIG = """\
[paths]
comsol_results = "N:/example_path/"
material_data = "C:/example_path/material/"
graphics = "C:/example_path/plots/"

[colors]
red = "tab:red"
blue = "tab:blue"
"""


def get_config_path() -> Path:
    """Return the absolute path to the config.toml file."""
    return Path(__file__).resolve().parent / "config.toml"


def ensure_config_exists() -> None:
    """Ensure config.toml exists. If not, generate one with dummy values."""
    config_path = get_config_path()
    if not config_path.exists():
        config_path.write_text(DUMMY_CONFIG, encoding="utf-8")
        raise FileNotFoundError(
            f"'config.toml' was missing.\n\n"
            f"A default file was created at: {config_path.resolve()}\n"
            f"Please update it with your local paths and preferences."
        )


def check_paths_in_toml() -> None:
    """Ensure config.toml exists and check if all paths under [paths] exist. Raises FileNotFoundError if any path is missing."""
    ensure_config_exists()

    config_path = get_config_path()
    config = toml.load(config_path)
    paths_section = config.get("paths", {})

    if not paths_section:
        raise ValueError("No [paths] section found in the config.toml file.")

    missing = []
    for key, path_str in paths_section.items():
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            missing.append(f"{key} → {path}")

    if missing:
        raise FileNotFoundError(
            "The following paths do not exist:\n" + "\n".join(f" - {entry}" for entry in missing)
        )


def load_config() -> Config:
    """Load and parse the config.toml file into a validated Config object."""
    ensure_config_exists()
    config_path = get_config_path()
    with config_path.open("r", encoding="utf-8") as file:
        config_dict = toml.load(file)
    return Config(**config_dict)


def get_user_paths() -> UserPaths:
    """Retrieve the validated user paths section from the configuration."""
    return load_config().paths


def get_user_colors() -> UserColors:
    """Retrieve the validated user colors section from the configuration."""
    return load_config().colors
