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
    processed_dir: Path
    rbi_psi_dir: Path
    rbi_bankwise_dir: Path
    rbi_repo_dir: Path
    npci_upi_dir: Path
    npci_p2m_dir: Path
    mospi_cpi_dir: Path
    pmjdy_dir: Path


class PsiColumnSpec(BaseModel):
    """How to locate one PSI column: by parent section, own label, and unit."""
    section: list[str] = Field(default_factory=list)
    section_not: list[str] = Field(default_factory=list)
    column: list[str]
    unit: list[str]


class PsiFormatConfig(BaseModel):
    """Layout for one PSI format variant (old or new)."""
    sheet_match: str
    label_row: int
    unit_row: int
    data_start_row: int
    date_col: int
    columns: dict[str, PsiColumnSpec]


class RbiPsiConfig(BaseModel):
    file_pattern: str
    date_format: str
    formats: dict[str, PsiFormatConfig]


class NpciUpiConfig(BaseModel):
    file_pattern: str

class NpciP2mConfig(BaseModel):
    file_pattern: str

class RbiBankwiseConfig(BaseModel):
    file_pattern: str

class RbiRepoConfig(BaseModel):
    file_pattern: str

class MospiCpiConfig(BaseModel):
    file_pattern: str

class PmjdyConfig(BaseModel):
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
    npci_p2m: NpciP2mConfig
    rbi_bankwise: RbiBankwiseConfig
    rbi_repo: RbiRepoConfig
    mospi_cpi: MospiCpiConfig
    pmjdy: PmjdyConfig
    validation: ValidationConfig
    issuers: dict[str, list[str]] = Field(default_factory=dict)
    structural_events: list[StructuralEvent] = Field(default_factory=list)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate config/settings.toml.

    Resolves relative paths to project-root absolute paths and ensures all
    data directories exist on disk (raw sub-dirs + processed).
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    settings = Settings(**raw)

    # Resolve every path field relative to project root and mkdir
    path_fields = [
        "processed_dir",
        "rbi_psi_dir",
        "rbi_bankwise_dir",
        "rbi_repo_dir",
        "npci_upi_dir",
        "npci_p2m_dir",
        "mospi_cpi_dir",
        "pmjdy_dir",
    ]
    for field in path_fields:
        p = getattr(settings.paths, field)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
            setattr(settings.paths, field, p)
        p.mkdir(parents=True, exist_ok=True)

    return settings
