"""Toml checker."""
from pydantic import BaseModel

class UserPaths(BaseModel):
    """Paths to the local directories."""

    material_data: str
    graphics: str
    external_material_data: str

class UserColors(BaseModel):
    """Customized pyplot colors."""

    gtruth: str
    compare1: str
    compare2: str
    compare3: str
    compare4: str
    # green: str
    # olive: str
    # red: str
    # orange: str

class Config(BaseModel):
    """Paths to the local directories."""

    paths: UserPaths
    colors: UserColors
