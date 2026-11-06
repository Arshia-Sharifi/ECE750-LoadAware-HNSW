"""LA-ef: resumable, verified downloads (IMPLEMENTATION_PLAN.md §7, P1.2).

Raw files are never deleted or re-fetched once present; they are immutable inputs.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .registry import DatasetSpec

USER_AGENT = "laef-preprocess/1.1"


def raw_file(spec: DatasetSpec, raw_root: Path) -> Path:
    """Location of the single raw file of an ann_hdf5 dataset: raw_root/<id>/<id>.hdf5."""
    return raw_root / spec.id / f"{spec.id}.hdf5"


def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def list_hf_shards(spec: DatasetSpec) -> list[str]:
    """Download URLs of the parquet shards in shard order (shard order is row order)."""
    api = spec.hf_tree_api or f"https://huggingface.co/api/datasets/{spec.hf_repo}/tree/main/data"
    prefix = (spec.hf_resolve_prefix or f"https://huggingface.co/datasets/{spec.hf_repo}/resolve/main/").rstrip("/")
    with _open(api, timeout=30) as r:
        entries = json.loads(r.read().decode("utf-8"))
    urls = [f"{prefix}/{e['path']}" for e in entries
            if e.get("type") == "file" and e.get("path", "").endswith(".parquet")]
    if not urls:
        raise RuntimeError(f"{spec.id}: no parquet shards listed at {api}")
    return sorted(urls)


def download_file(url: str, out: Path, retries: int = 5, chunk: int = 1 << 20) -> Path:
    """Fetch `url` into `out`. Resumes from out.part with a Range request and checks the byte count."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            pos = part.stat().st_size if part.exists() else 0
            with _open(url, {"Range": f"bytes={pos}-"} if pos else None, timeout=120) as r:
                resumed = pos > 0 and getattr(r, "status", 200) == 206
                if pos and not resumed:
                    pos = 0  # the server ignored Range: start the file again
                clen = r.headers.get("Content-Length")
                expected = pos + int(clen) if clen is not None else None
                with open(part, "ab" if resumed else "wb") as f:
                    while block := r.read(chunk):
                        f.write(block)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise IOError(f"{out.name}: got {size} bytes, expected {expected}")
            part.replace(out)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # range starts past the end: the part is complete
                part.replace(out)
                return out
            err = e
        except (urllib.error.URLError, OSError) as e:  # URLError, timeouts, size mismatch
            err = e
        if attempt < retries:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {err}") from err


def download_raw(spec: DatasetSpec, raw_root: Path) -> Path:
    """Fetch the raw files of one dataset into raw_root/<id>/. Idempotent."""
    ds_dir = raw_root / spec.id
    ds_dir.mkdir(parents=True, exist_ok=True)
    if spec.source_type == "ann_hdf5":
        if len(spec.urls) != 1:
            raise ValueError(f"{spec.id}: ann_hdf5 needs exactly one URL in 'urls'")
        download_file(spec.urls[0], raw_file(spec, raw_root))
    elif spec.source_type == "hf_parquet":
        if not (spec.hf_repo or spec.hf_tree_api):
            raise ValueError(f"{spec.id}: hf_parquet needs hf_repo or hf_tree_api")
        urls = list_hf_shards(spec)
        for i, url in enumerate(urls, 1):
            name = url.rsplit("/", 1)[-1]
            print(f"  [{i}/{len(urls)}] {name}", flush=True)
            download_file(url, ds_dir / name)
    elif spec.source_type != "synthetic":
        raise ValueError(f"{spec.id}: unknown source_type {spec.source_type!r}")
    return ds_dir
