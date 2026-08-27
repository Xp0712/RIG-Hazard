from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_bundle_manifest(paths: Iterable[str | Path], root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    files = []
    for raw in sorted((Path(path).resolve() for path in paths), key=str):
        try:
            display = str(raw.relative_to(base)).replace("\\", "/")
        except ValueError:
            display = str(raw).replace("\\", "/")
        files.append(
            {
                "path": display,
                "bytes": int(raw.stat().st_size),
                "sha256": sha256_file(raw),
            }
        )
    return {"files": files, "bundle_sha256": canonical_sha256(files)}
