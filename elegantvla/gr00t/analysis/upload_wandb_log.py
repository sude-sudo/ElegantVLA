#!/usr/bin/env python3

"""Upload an existing elegantvla_gr00t log file to Weights & Biases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from elegantvla.gr00t.entrypoints._common import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb,
    maybe_wandb_save,
    parse_csv_items,
    update_wandb_summary,
)


@dataclass
class UploadConfig:
    log_path: str
    project: str = "temporal-skip"
    run_name: str | None = None
    entity: str | None = None
    group: str | None = None
    mode: str = "online"
    job_type: str = "log-import"
    tags: str = "elegantvla_gr00t,log-import"
    wandb_dir: str | None = None
    extra_paths: str = ""
    run_id: str | None = None
    resume: str = "allow"


def _parse_header(log_path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if "=" not in line:
            if header:
                break
            continue
        key, value = line.split("=", 1)
        if not key or " " in key:
            break
        header[key] = value
    return header


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return value
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_sb3_history(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_block: list[str] = []
    in_block = False

    def flush_block(lines: list[str]) -> None:
        parsed = _parse_sb3_block(lines)
        if parsed:
            rows.append(parsed)

    for raw_line in log_path.read_text(errors="replace").splitlines():
        stripped = raw_line.rstrip()
        if set(stripped) == {"-"} and len(stripped) >= 10:
            if not in_block:
                current_block = []
                in_block = True
            else:
                flush_block(current_block)
                current_block = []
                in_block = False
            continue
        if in_block:
            current_block.append(stripped)

    if in_block and current_block:
        flush_block(current_block)
    return rows


def _parse_sb3_block(lines: list[str]) -> dict[str, Any]:
    section: str | None = None
    parsed: dict[str, Any] = {}
    saw_metric = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.rstrip() for cell in stripped.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        left = cells[0].strip()
        right = cells[1].strip()
        if not left:
            continue
        if left.endswith("/") and right == "":
            section = left[:-1]
            continue
        key = left
        if section is not None:
            key = f"{section}/{left}"
        parsed[key] = _coerce_value(right)
        saw_metric = True
    if not saw_metric or "time/total_timesteps" not in parsed:
        return {}
    return parsed


def main(cfg: UploadConfig) -> None:
    log_path = Path(cfg.log_path).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Log path {log_path} does not exist.")

    header = _parse_header(log_path)
    history_rows = _parse_sb3_history(log_path)

    run = init_wandb_run(
        enabled=True,
        project=cfg.project,
        entity=cfg.entity,
        group=cfg.group,
        name=cfg.run_name or log_path.stem,
        job_type=cfg.job_type,
        tags=cfg.tags,
        dir=cfg.wandb_dir,
        mode=cfg.mode,
        config={
            "source_log_path": str(log_path),
            "source_log_name": log_path.name,
            "source_log_size_bytes": log_path.stat().st_size,
            "parsed_header": header,
            "parsed_history_rows": len(history_rows),
        },
        run_id=cfg.run_id,
        resume=cfg.resume if cfg.run_id is not None else None,
        reinit=True,
    )
    try:
        update_wandb_summary(
            run,
            {
                "source_log_path": str(log_path),
                "source_log_size_bytes": log_path.stat().st_size,
                "parsed_history_rows": len(history_rows),
                **{f"header/{key}": value for key, value in header.items()},
            },
        )

        for row in history_rows:
            step = row.get("time/total_timesteps")
            log_wandb(
                row,
                enabled=True,
                step=int(step) if isinstance(step, (int, float)) else None,
            )

        maybe_wandb_save(log_path, enabled=True)
        for extra_path in parse_csv_items(cfg.extra_paths):
            maybe_wandb_save(extra_path, enabled=True)

        print(
            {
                "wandb_project": cfg.project,
                "wandb_run_name": run.name,
                "source_log_path": str(log_path),
                "parsed_history_rows": len(history_rows),
            }
        )
    finally:
        finish_wandb_run(run)


def cli_main() -> None:
    main(tyro.cli(UploadConfig))


if __name__ == "__main__":
    cli_main()
