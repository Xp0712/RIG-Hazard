"""Read frozen tabular artifacts through the canonical naming boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .naming import (
    canonical_column_renames,
    canonicalize_artifact_name,
    resolve_artifact_name,
)


def canonicalize_frame(
    frame: pd.DataFrame,
    *,
    identifier_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a frame whose legacy columns and identifier values are canonical."""

    result = frame.rename(columns=canonical_column_renames(list(frame.columns)))
    for column in identifier_columns:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: canonicalize_artifact_name(str(value))
            )
    return result


def read_artifact_csv(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    identifier_columns: tuple[str, ...] = (),
    **kwargs: Any,
) -> pd.DataFrame:
    """Read canonical columns from either a current or a frozen CSV schema."""

    source_path = Path(path)
    read_kwargs = dict(kwargs)
    read_kwargs.pop("usecols", None)
    if columns is None:
        frame = pd.read_csv(source_path, **read_kwargs)
    else:
        header = pd.read_csv(source_path, nrows=0, encoding=read_kwargs.get("encoding"))
        source_columns = [
            resolve_artifact_name(list(header.columns), column) for column in columns
        ]
        frame = pd.read_csv(source_path, usecols=source_columns, **read_kwargs)
    return canonicalize_frame(frame, identifier_columns=identifier_columns)
