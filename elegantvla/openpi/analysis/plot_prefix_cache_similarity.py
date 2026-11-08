import argparse
import json
import math
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prefix-cache similarity observations.")
    parser.add_argument(
        "--input",
        default="output_libero_shadow_prefix_sparse_count2_cka09940_mc1_tasks0_9_3trials_20260610.json",
    )
    parser.add_argument("--output-dir", default="output/prefix_cache_similarity_20260610")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.9940)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _episode_stats(episode: dict[str, Any]) -> dict[str, Any]:
    queries = episode.get("queries", [])
    query_count = len(queries)
    shadow_hits = sum(1 for query in queries if query.get("prefix_vlm_cache_shadow_hit"))
    shadow_cached = sum(int(query.get("prefix_vlm_cache_shadow_reused_blocks") or 0) for query in queries)
    total_layers = query_count * 18
    return {
        "task_id": int(episode.get("task_id", -1)),
        "episode_idx": int(episode.get("episode_idx", -1)),
        "success": bool(episode.get("success")),
        "queries": query_count,
        "shadow_hits": shadow_hits,
        "shadow_cached": shadow_cached,
        "shadow_cache_fraction": shadow_cached / total_layers if total_layers else 0.0,
    }


def _anchor_similarity_matrix(episode: dict[str, Any]) -> tuple[np.ndarray, list[int | None], list[float | None]]:
    queries = episode.get("queries", [])
    size = len(queries)
    matrix = np.full((size, size), np.nan, dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    anchors: list[int | None] = []
    ckas: list[float | None] = []
    anchor: int | None = None

    for idx, query in enumerate(queries):
        cka = _finite(query.get("prefix_vlm_cache_first_layer_cka"))
        anchors.append(anchor)
        ckas.append(cka)
        if anchor is not None and cka is not None:
            matrix[idx, anchor] = cka
            matrix[anchor, idx] = cka

        # In observe-only, a shadow hit keeps the previous cache anchor. Any
        # non-hit runs a full prefix refresh and becomes the next anchor.
        if not bool(query.get("prefix_vlm_cache_shadow_hit")):
            anchor = idx
    return matrix, anchors, ckas


def _pairwise_similarity_matrix(episode: dict[str, Any]) -> np.ndarray | None:
    queries = episode.get("queries", [])
    if not any(query.get("prefix_vlm_cache_pairwise_cka_row") for query in queries):
        return None
    size = len(queries)
    matrix = np.full((size, size), np.nan, dtype=np.float32)
    for idx, query in enumerate(queries):
        row = query.get("prefix_vlm_cache_pairwise_cka_row")
        if not row:
            continue
        row_values = [_finite(value) for value in row]
        for col, value in enumerate(row_values[: idx + 1]):
            if value is None:
                continue
            matrix[idx, col] = value
            matrix[col, idx] = value
    return matrix


def _plot_episode_matrix(
    episode: dict[str, Any],
    output_path: pathlib.Path,
    threshold: float,
) -> None:
    stats = _episode_stats(episode)
    pairwise_matrix = _pairwise_similarity_matrix(episode)
    matrix, anchors, ckas = _anchor_similarity_matrix(episode)
    is_pairwise = pairwise_matrix is not None
    if is_pairwise:
        matrix = pairwise_matrix
    queries = episode.get("queries", [])
    size = len(queries)
    hit_indices = [idx for idx, query in enumerate(queries) if query.get("prefix_vlm_cache_shadow_hit")]
    refresh_indices = [idx for idx, query in enumerate(queries) if not query.get("prefix_vlm_cache_shadow_hit")]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, size * 0.42), max(7, size * 0.34)),
        gridspec_kw={"height_ratios": [4, 1.25]},
        constrained_layout=True,
    )
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad(color="#eeeeee")
    sns.heatmap(
        matrix,
        ax=axes[0],
        cmap=cmap,
        vmin=0.985,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "first-layer CKA" if is_pairwise else "first-layer CKA to active cache anchor"},
        linewidths=0.2,
        linecolor="#d5d5d5",
    )
    axes[0].set_title(
        ("Pairwise CKA matrix: " if is_pairwise else "Anchor CKA matrix: ")
        +
        f"task {stats['task_id']} / ep {stats['episode_idx']} / "
        f"shadow {stats['shadow_cached']}/{stats['queries'] * 18} layers "
        f"({stats['shadow_cache_fraction'] * 100:.2f}%)"
    )
    axes[0].set_xlabel("policy query index")
    axes[0].set_ylabel("policy query index")
    for idx in hit_indices:
        anchor = anchors[idx]
        if anchor is not None:
            axes[0].scatter(anchor + 0.5, idx + 0.5, marker="o", s=55, facecolors="none", edgecolors="white")
            axes[0].scatter(idx + 0.5, anchor + 0.5, marker="o", s=55, facecolors="none", edgecolors="white")

    x = np.arange(size)
    y = np.array([np.nan if value is None else value for value in ckas], dtype=np.float32)
    axes[1].plot(x, y, color="#255f85", linewidth=1.8, marker=".", label="CKA to active anchor")
    axes[1].axhline(threshold, color="#c9412f", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.5f}")
    if hit_indices:
        axes[1].scatter(hit_indices, y[hit_indices], color="#0f8b4c", s=42, label="shadow hit")
    if refresh_indices:
        finite_refresh = [idx for idx in refresh_indices if idx < len(y) and np.isfinite(y[idx])]
        axes[1].scatter(finite_refresh, y[finite_refresh], color="#d99021", s=26, label="refresh")
    axes[1].set_ylim(0.982, 1.0005)
    axes[1].set_xlabel("policy query index")
    axes[1].set_ylabel("CKA")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_task_summary(episodes: list[dict[str, Any]], output_path: pathlib.Path) -> None:
    rows: dict[int, dict[str, float]] = {}
    for episode in episodes:
        stats = _episode_stats(episode)
        task = stats["task_id"]
        bucket = rows.setdefault(task, {"queries": 0, "hits": 0, "cached": 0, "layers": 0, "episodes": 0})
        bucket["queries"] += stats["queries"]
        bucket["hits"] += stats["shadow_hits"]
        bucket["cached"] += stats["shadow_cached"]
        bucket["layers"] += stats["queries"] * 18
        bucket["episodes"] += 1

    tasks = sorted(rows)
    cache_frac = [rows[task]["cached"] / rows[task]["layers"] if rows[task]["layers"] else 0 for task in tasks]
    hit_rate = [rows[task]["hits"] / rows[task]["queries"] if rows[task]["queries"] else 0 for task in tasks]

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    x = np.arange(len(tasks))
    width = 0.38
    ax1.bar(x - width / 2, cache_frac, width, label="shadow cache layer fraction", color="#2a6f97")
    ax1.bar(x + width / 2, hit_rate, width, label="shadow hit rate", color="#b5651d")
    ax1.set_xticks(x, [str(task) for task in tasks])
    ax1.set_xlabel("LIBERO task id")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(0, max(max(cache_frac), max(hit_rate), 0.01) * 1.25)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right")
    ax1.set_title("Shadow-cache opportunity by task (count=2, CKA=0.9940, mc=1)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(path: pathlib.Path, selected: list[dict[str, Any]], input_path: pathlib.Path) -> None:
    lines = [
        "# Prefix Cache Similarity Plots",
        "",
        f"Input: `{input_path}`",
        "",
        "The matrices are anchor-CKA matrices from observe-only runs. Gray cells are unobserved pairwise values; colored off-diagonal cells show the CKA between the current query and the active cache anchor.",
        "",
        "| rank | task | episode | queries | shadow hits | shadow cached layers | shadow fraction | image |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(selected, start=1):
        image = f"anchor_similarity_task{row['task_id']}_ep{row['episode_idx']}.png"
        lines.append(
            f"| {rank} | {row['task_id']} | {row['episode_idx']} | {row['queries']} | "
            f"{row['shadow_hits']} | {row['shadow_cached']} | {row['shadow_cache_fraction'] * 100:.2f}% | "
            f"`{image}` |"
        )
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    payload = json.loads(input_path.read_text())
    episodes = payload.get("episodes", [])
    ranked = sorted((_episode_stats(ep) | {"episode": ep} for ep in episodes), key=lambda r: r["shadow_cache_fraction"], reverse=True)
    selected = ranked[: args.top_k]

    for row in selected:
        _plot_episode_matrix(
            row["episode"],
            output_dir / f"anchor_similarity_task{row['task_id']}_ep{row['episode_idx']}.png",
            args.threshold,
        )
    _plot_task_summary(episodes, output_dir / "shadow_cache_task_summary.png")
    _write_summary(output_dir / "README.md", selected, input_path)


if __name__ == "__main__":
    main()
