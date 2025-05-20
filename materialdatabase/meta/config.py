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
        raise FileNotFoundError((f"config.toml"
                                 f"\n\n A default file was created at: {config_path}\n"
                                 f"Please update it with your local paths and preferences."))


def ensure_config_path_exists(path: Path) -> None:
    """Ensure config.toml exists. If not, generate one with dummy values."""
    config_path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"\n\n The path '{path}', specified in '{config_path}' does not exist.\n"
                                f"Please update it with your local paths.")

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
