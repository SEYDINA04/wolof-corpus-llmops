# src/pipeline/config.py
"""
Chargement centralisé de la configuration du pipeline.

- lit `pipeline.yaml` (config fonctionnelle, versionnée dans git)
- lit les secrets depuis l'environnement / `.env` (jamais versionnés) :
    * HF_TOKEN         : token HuggingFace (write)
    * HF_DATASET_REPO  : surcharge éventuelle du repo cible

Tous les chemins sont résolus en absolu, relatifs au dossier `src/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent.parent          # .../wolof_scraper/src
REPO_ROOT = SRC_DIR.parent                                  # .../wolof_scraper
DEFAULT_CONFIG = SRC_DIR / "pipeline.yaml"


def _load_dotenv() -> None:
    """Charge .env (cherché depuis src/ vers la racine) si python-dotenv dispo."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True) or str(REPO_ROOT / ".env")
    if Path(path).exists():
        load_dotenv(path)


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # ---- secrets (env) ----
    @property
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN")

    # ---- HF ----
    @property
    def hf_repo(self) -> str:
        return os.environ.get("HF_DATASET_REPO") or self.raw["hf"]["repo"]

    @property
    def hf_repo_type(self) -> str:
        return self.raw["hf"]["repo_type"]

    @property
    def hf_filename(self) -> str:
        return self.raw["hf"]["filename"]

    # ---- chemins (absolus) ----
    def p(self, key: str) -> Path:
        """Résout un chemin de la section `paths` en absolu (relatif à src/)."""
        return (SRC_DIR / self.raw["paths"][key]).resolve()

    # ---- sous-sections ----
    @property
    def ingest(self) -> dict[str, Any]:
        return self.raw["ingest"]

    @property
    def stats(self) -> dict[str, Any]:
        return self.raw["stats"]

    @property
    def gates(self) -> dict[str, Any]:
        return self.raw["quality_gates"]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    _load_dotenv()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(raw=raw, path=path)
