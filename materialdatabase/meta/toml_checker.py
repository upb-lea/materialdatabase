"""Toml checker."""
from pydantic import BaseModel

class UserPaths(BaseModel):
    """Paths to the local directories."""

    comsol_results: str
    material_data: str
    graphics: str

class UserColors(BaseModel):
    """Customized pyplot colors."""

    red: str
    blue: str

class Config(BaseModel):
    """Paths to the local directories."""

    paths: UserPaths
    colors: UserColors
