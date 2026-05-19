"""Configuration loader.

Loads config/settings.toml and validates it against typed pydantic schemas.
All relative paths in the config are resolved to the project root.
"""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path


class RbiPsiConfig(BaseModel):
    file_pattern: str
    header_rows: list[int]
    data_start_row: int
    columns: dict[str, list[str]]


class NpciUpiConfig(BaseModel):
    file_pattern: str


class ValidationConfig(BaseModel):
    max_null_pct: float = 20.0
    max_date_gap_days: int = 35
    min_rows: int = 60


class StructuralEvent(BaseModel):
    date: date
    name: str
    direction: str
    notes: str = ""


class Settings(BaseModel):
    paths: PathsConfig
    rbi_psi: RbiPsiConfig
    npci_upi: NpciUpiConfig
    validation: ValidationConfig
    issuers: dict[str, list[str]] = Field(default_factory=dict)
    structural_events: list[StructuralEvent] = Field(default_factory=list)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate config/settings.toml.

    Resolves relative paths to project-root absolute paths and ensures
    data directories exist on disk.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    settings = Settings(**raw)

    # Resolve relative paths against project root
    if not settings.paths.raw_dir.is_absolute():
        settings.paths.raw_dir = PROJECT_ROOT / settings.paths.raw_dir
    if not settings.paths.processed_dir.is_absolute():
        settings.paths.processed_dir = PROJECT_ROOT / settings.paths.processed_dir

    settings.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings
