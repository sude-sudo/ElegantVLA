#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from elegantvla.gr00t.entrypoints._common import read_jsonl, read_tsv, write_jsonl, write_tsv


@dataclass
class Config:
    base_run_dir: str
    extra_run_dir: str
    output_dir: str


def _read_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_tsv(path):
        rows.append(
            {
                "episode": int(row["episode"]),
                "return": float(row["return"]),
                "decision_steps": int(row["decision_steps"]),
                "success": row["success"] == "True",
            }
        )
    return rows


def main(cfg: Config) -> None:
    base_root = Path(cfg.base_run_dir)
    extra_root = Path(cfg.extra_run_dir)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    task_names = sorted({p.name for p in base_root.glob("*") if p.is_dir()} | {p.name for p in extra_root.glob("*") if p.is_dir()})

    for task in task_names:
        base_dir = base_root / task
        extra_dir = extra_root / task
        out_dir = out_root / task
        out_dir.mkdir(parents=True, exist_ok=True)

        base_summary = _read_summary(base_dir / "summary.tsv")
        extra_summary = _read_summary(extra_dir / "summary.tsv")
        base_trace = read_jsonl(base_dir / "decision_trace.jsonl")
        extra_trace = read_jsonl(extra_dir / "decision_trace.jsonl")

        episode_offset = len(base_summary)
        merged_summary = list(base_summary)
        for row in extra_summary:
            row = dict(row)
            row["episode"] = int(row["episode"]) + episode_offset
            merged_summary.append(row)

        merged_trace = list(base_trace)
        for row in extra_trace:
            row = dict(row)
            row["episode"] = int(row["episode"]) + episode_offset
            merged_trace.append(row)

        write_tsv(
            out_dir / "summary.tsv",
            ["episode", "return", "decision_steps", "success"],
            merged_summary,
        )
        write_jsonl(out_dir / "decision_trace.jsonl", merged_trace)

    print(out_root)


if __name__ == "__main__":
    main(tyro.cli(Config))
