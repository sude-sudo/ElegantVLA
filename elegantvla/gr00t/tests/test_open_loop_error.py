from __future__ import annotations

import numpy as np

from elegantvla.gr00t.analysis.open_loop_error.metrics import action_dict_error, mean_metrics


def test_action_dict_error_reports_chunk_and_prefix() -> None:
    reference = {
        "arm": np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
        "gripper": np.asarray([[[0.5], [1.0]]], dtype=np.float32),
    }
    candidate = {
        "arm": np.asarray([[[1.0, 1.0], [4.0, 3.0]]], dtype=np.float32),
        "gripper": np.asarray([[[0.5], [0.0]]], dtype=np.float32),
    }

    chunk = action_dict_error(reference, candidate)
    prefix = action_dict_error(reference, candidate, prefix_steps=1)

    assert chunk["overall"]["count"] == 6
    assert np.isclose(chunk["overall"]["mae"], 4.0 / 6.0)
    assert prefix["overall"]["count"] == 3
    assert np.isclose(prefix["overall"]["mae"], 1.0 / 3.0)
    assert set(chunk["per_dimension"]) == {"arm[0]", "arm[1]", "gripper[0]"}


def test_mean_metrics_keeps_step_count() -> None:
    reference = {"action": np.zeros((1, 2, 1), dtype=np.float32)}
    first = action_dict_error(reference, {"action": np.ones((1, 2, 1), dtype=np.float32)})
    second = action_dict_error(reference, {"action": np.full((1, 2, 1), 3.0, dtype=np.float32)})

    summary = mean_metrics(
        [{"chunk_error": first}, {"chunk_error": second}],
        metric_key="chunk_error",
    )

    assert summary["steps"] == 2
    assert np.isclose(summary["mae"], 2.0)
    assert np.isclose(summary["mse"], 5.0)
