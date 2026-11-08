import json
from pathlib import Path
import tempfile

from elegantvla.gr00t.entrypoints.evaluate import (
    RunConfig,
    _completed_episode_prefix,
    _read_jsonl_prefix,
    _validate_resume_config,
)
from elegantvla.gr00t.analysis.summarize_paired_suite import (
    _is_latency_warmup,
)


def _config(**overrides) -> RunConfig:
    values = {
        "model_path": "/tmp/model",
        "episodes": 4,
        "seed": 10,
        "output_dir": "/tmp/output",
    }
    values.update(overrides)
    return RunConfig(**values)


def test_completed_episode_prefix_stops_at_first_gap() -> None:
    rows = [
        {"episode": 0, "seed": 10},
        {"episode": 1, "seed": 11},
        {"episode": 3, "seed": 13},
    ]

    assert _completed_episode_prefix(rows, _config()) == rows[:2]


def test_jsonl_prefix_ignores_truncated_tail() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "episodes.jsonl"
        path.write_text('{"episode": 0}\n{"episode":', encoding="utf-8")

        assert _read_jsonl_prefix(path) == [{"episode": 0}]


def test_resume_config_accepts_legacy_config_without_resume_field() -> None:
    cfg = _config()
    previous = dict(vars(cfg))
    previous.pop("resume")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "run_config.json"
        path.write_text(json.dumps(previous), encoding="utf-8")

        _validate_resume_config(path, cfg)


def test_explicit_resume_warmup_overrides_episode_index() -> None:
    assert _is_latency_warmup({"episode": 5, "latency_warmup": True}, 1)
    assert not _is_latency_warmup({"episode": 0, "latency_warmup": False}, 1)
    assert _is_latency_warmup({"episode": 0}, 1)


if __name__ == "__main__":
    test_completed_episode_prefix_stops_at_first_gap()
    test_jsonl_prefix_ignores_truncated_tail()
    test_resume_config_accepts_legacy_config_without_resume_field()
    test_explicit_resume_warmup_overrides_episode_index()
