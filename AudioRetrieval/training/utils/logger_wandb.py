"""Thin Weights & Biases wrapper to simplify experiment logging."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "colaf_training"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: Sequence[str] = field(default_factory=list)
    dir: Optional[str] = None
    mode: Optional[str] = None
    notes: Optional[str] = None
    log_best_checkpoint: bool = True
    log_last_checkpoint: bool = False


class WandbLogger(AbstractContextManager):
    def __init__(self, config: WandbConfig):
        if config.enabled and wandb is None:
            raise RuntimeError("wandb package is required but not installed.")
        self.config = config
        self.active = False
        self.run = None

    def __enter__(self):
        if not self.config.enabled:
            return self

        init_kwargs = {
            "project": self.config.project,
            "entity": self.config.entity,
            "name": self.config.run_name,
            "group": self.config.group,
            "job_type": self.config.job_type,
            "dir": self.config.dir,
            "mode": self.config.mode,
            "notes": self.config.notes,
            "tags": list(self.config.tags) if self.config.tags else None,
        }
        clean_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, "", [])}
        self.run = wandb.init(**clean_kwargs)
        self.active = True
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self.active and self.run is not None:
            self.run.finish()
        self.active = False
        self.run = None
        return False

    def _sanitize(self, name: str) -> str:
        return name.replace(" ", "_")

    def log_params(self, params: Dict[str, str]) -> None:
        if not self.active or self.run is None:
            return
        safe = {self._sanitize(k): v for k, v in params.items()}
        self.run.config.update(safe, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.active or self.run is None:
            return
        safe = {self._sanitize(k): v for k, v in metrics.items()}
        self.run.log(safe, step=step)

    def log_artifact(self, path: str) -> None:
        if not self.active or self.run is None:
            return
        file_path = Path(path)
        if not file_path.exists():
            return
        artifact = wandb.Artifact(
            name=f"{file_path.stem}-{wandb.util.generate_id()}",
            type="checkpoint",
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)
