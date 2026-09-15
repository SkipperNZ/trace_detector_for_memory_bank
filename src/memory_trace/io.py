"""Portable UTF-8 artifacts, canonical identities, and strict JSON input."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def uniform(seed: int, namespace: str, key: object) -> float:
    # 53 random bits, so the conversion cannot round up to 1.0.
    return (int(digest([seed, namespace, key])[:16], 16) >> 11) / 2**53


def _constant(value: str):
    raise ValueError(f"Non-finite JSON value: {value}")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_constant)
                if not isinstance(row, dict):
                    raise ValueError("Each row must be a JSON object")
                rows.append(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Close the temporary file before replacement: required on Windows.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: str | Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    write_text(path, "".join(canonical(row) + "\n" for row in rows))


def index_rows(rows: list[dict], field: str = "input_id") -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Missing/non-string {field}")
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def probability(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1 or (positive and value == 0):
        raise ValueError(f"{name} must be {'(0, 1]' if positive else '[0, 1]'}")
    return value
