"""LA-ef: load configs/datasets.yaml into DatasetSpec objects (IMPLEMENTATION_PLAN.md §7, P1.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"


@dataclass
class DatasetSpec:
    id: str
    role: str = "analysis"
    source_type: str = "ann_hdf5"        # ann_hdf5 | hf_parquet | synthetic
    metric: str = "cd"                   # cd | ipd (the only metrics Ada-ef supports)
    dim: int = 0
    n_base: int = 0                      # rows before the holdout split
    n_query: int = 0
    query_source: str = "provided"       # provided | holdout | synthetic_sample
    gt_provided_valid: bool = False
    seed: int = 42
    urls: list[str] = field(default_factory=list)       # ann_hdf5 / fasttext_vec: download URL(s)
    hf_repo: str = ""                    # hf_parquet / hf_text_encode
    hf_subset: str = ""                  # hf_text_encode config/subset
    hf_split: str = "train"              # hf_text_encode split
    hf_tree_api: str = ""                # hf_parquet: JSON listing of the shards
    hf_resolve_prefix: str = ""          # hf_parquet: prefix + shard path = download URL
    column: str = ""                     # hf_parquet: list-valued embedding column
    text_column: str = "text"            # hf_text_encode: text column to encode
    encode_model: str = ""               # hf_text_encode: sentence-transformers model
    synthetic: dict[str, Any] = field(default_factory=dict)
    holdout_seed: int | None = None      # defaults to seed

    def __post_init__(self) -> None:
        if self.metric not in ("cd", "ipd"):
            raise ValueError(f"{self.id}: metric must be 'cd' or 'ipd', got {self.metric!r}")
        if self.holdout_seed is None:
            self.holdout_seed = self.seed


def load_registry(path: str | Path | None = None) -> dict[str, DatasetSpec]:
    path = Path(path) if path else DEFAULT_REGISTRY
    if not path.exists():
        raise FileNotFoundError(f"dataset registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = raw.get("datasets") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level 'datasets:' list")
    specs: dict[str, DatasetSpec] = {}
    for e in entries:
        spec = DatasetSpec(
            id=e["id"],
            role=e.get("role", "analysis"),
            source_type=e.get("source_type", "ann_hdf5"),
            metric=e.get("metric", "cd"),
            dim=int(e.get("dim", 0)),
            n_base=int(e.get("n_base", 0)),
            n_query=int(e.get("n_query", 0)),
            query_source=e.get("query_source", "provided"),
            gt_provided_valid=bool(e.get("gt_provided_valid", False)),
            seed=int(e.get("seed", 42)),
            urls=list(e.get("urls") or []),
            hf_repo=e.get("hf_repo", ""),
            hf_subset=e.get("hf_subset", ""),
            hf_split=e.get("hf_split", "train"),
            hf_tree_api=e.get("hf_tree_api", ""),
            hf_resolve_prefix=e.get("hf_resolve_prefix", ""),
            column=e.get("column", ""),
            text_column=e.get("text_column", "text"),
            encode_model=e.get("encode_model", ""),
            synthetic=dict(e.get("synthetic") or {}),
            holdout_seed=e.get("holdout_seed"),
        )
        specs[spec.id] = spec
    return specs


def get_dataset(name: str, path: str | Path | None = None) -> DatasetSpec:
    reg = load_registry(path)
    if name not in reg:
        raise KeyError(f"dataset {name!r} is not in the registry; available: {sorted(reg)}")
    return reg[name]
