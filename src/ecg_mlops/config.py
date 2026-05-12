from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config" / "config.yaml"
    if not path.is_absolute():
        path = root / path

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root() / path

