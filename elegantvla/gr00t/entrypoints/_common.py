"""Shared entrypoint-side experiment logging and artifact IO helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def parse_csv_items(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n")


def append_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
        for row in rows
    )
    if not payload:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_tsv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def sanitize_table_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            sanitized[key] = value.item()
        elif isinstance(value, (list, dict)):
            sanitized[key] = json.dumps(value, sort_keys=True)
        else:
            sanitized[key] = value
    return sanitized


def make_wandb_table(rows: list[dict[str, Any]]) -> Any | None:
    if not rows:
        return None

    import wandb

    sanitized_rows = [sanitize_table_row(row) for row in rows]
    columns = list(sanitized_rows[0].keys())
    data = [[row.get(column) for column in columns] for row in sanitized_rows]
    return wandb.Table(columns=columns, data=data)


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    job_type: str | None = None,
    tags: str | list[str] | None = None,
    config: Any | None = None,
    dir: str | None = None,
    mode: str = "online",
    sync_tensorboard: bool | None = None,
    run_id: str | None = None,
    resume: str | None = None,
    reinit: bool | None = None,
) -> Any | None:
    if not enabled:
        return None

    import wandb

    resolved_tags = tags if isinstance(tags, list) else parse_csv_items(tags)
    kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": group,
        "name": name,
        "job_type": job_type,
        "tags": resolved_tags,
        "config": config,
        "dir": dir,
        "mode": mode,
    }
    if sync_tensorboard is not None:
        kwargs["sync_tensorboard"] = bool(sync_tensorboard)
    if run_id is not None:
        kwargs["id"] = run_id
    if resume is not None:
        kwargs["resume"] = resume
    if reinit is not None:
        kwargs["reinit"] = reinit
    return wandb.init(**kwargs)


def log_wandb(payload: Mapping[str, Any], *, enabled: bool, step: int | None = None) -> None:
    if not enabled:
        return

    import wandb

    if step is None:
        wandb.log(dict(payload))
    else:
        wandb.log(dict(payload), step=int(step))


def update_wandb_summary(run: Any | None, payload: Mapping[str, Any]) -> None:
    if run is None:
        return
    for key, value in payload.items():
        run.summary[key] = value


def maybe_wandb_save(path: str | Path | None, *, enabled: bool) -> None:
    if not enabled or path is None:
        return
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists() or not resolved_path.is_file():
        return

    import wandb

    wandb.save(str(resolved_path), base_path=str(resolved_path.parent))


def maybe_wandb_save_many(raw_paths: str | None, *, enabled: bool) -> None:
    for raw_path in parse_csv_items(raw_paths):
        maybe_wandb_save(raw_path, enabled=enabled)


def finish_wandb_run(run: Any | None) -> None:
    if run is None:
        return

    import wandb

    wandb.finish()
