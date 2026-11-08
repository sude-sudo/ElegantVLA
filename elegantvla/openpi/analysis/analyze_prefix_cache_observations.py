import argparse
import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any


PREFIX_LAYERS = 18
DEFAULT_THRESHOLDS = (0.9900, 0.9930, 0.9940, 0.99425, 0.9945, 0.9950, 0.9970)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LIBERO prefix-cache rollout observations.")
    parser.add_argument("--glob", default="output_libero_*20260610.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--date", default="20260610")
    parser.add_argument("--include-partial", action="store_true")
    return parser.parse_args()


def _task_ids(spec: Any) -> list[int] | None:
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if not text or text in {"all", "*"}:
        return None
    ids: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(chunk))
    return ids


def _expected_episodes(path: pathlib.Path, payload: dict[str, Any]) -> int | None:
    args = payload.get("args") or {}
    task_ids = _task_ids(args.get("task_ids"))
    trials = args.get("num_trials_per_task")
    if task_ids is not None and trials is not None:
        return len(task_ids) * int(trials)

    name = path.name
    match = re.search(r"tasks(\d+)_(\d+)_(\d+)trials", name)
    if match:
        start, end, trial_count = (int(x) for x in match.groups())
        return (end - start + 1) * trial_count
    match = re.search(r"task(\d+)_(\d+)trials", name)
    if match:
        return int(match.group(2))
    return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def _pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _threshold_from_name(name: str) -> float | None:
    match = re.search(r"cka(\d+)", name)
    if not match:
        return None
    digits = match.group(1)
    if digits.startswith("0") and len(digits) > 1:
        return float("0." + digits[1:])
    return float(digits)


def _config_tags(path: pathlib.Path, payload: dict[str, Any], query_records: list[dict[str, Any]]) -> dict[str, Any]:
    name = path.name
    count_match = re.search(r"count(\d+)", name)
    mc_match = re.search(r"_mc(\d+)", name)
    hit_masks = Counter(_mask_string(q.get("prefix_vlm_cache_compute_layer_mask")) for q in query_records if q.get("prefix_vlm_cache_hit"))
    hit_masks.pop("", None)
    scheduler_actions = (payload.get("summary") or {}).get("threshold_scheduler_ah_actions") or {}
    has_action_signal = bool(scheduler_actions) or any(
        q.get("threshold_scheduler_ah_action") is not None
        or q.get("intra_action_cached_steps") is not None
        or q.get("intra_action_refresh_steps") is not None
        for q in query_records
    )
    prefix_only_name = "prefix_sparse" in name or "prefix_only" in name
    return {
        "count": int(count_match.group(1)) if count_match else None,
        "max_continuous": int(mc_match.group(1)) if mc_match else None,
        "cka_threshold": _threshold_from_name(name),
        "prefix_only": prefix_only_name and not has_action_signal,
        "action_cache_enabled": has_action_signal or (
            not prefix_only_name
            and any(token in name for token in ("recommended", "balanced", "firstlast", "action_only", "noimage"))
        ),
        "hit_masks": "; ".join(f"{mask}:{count}" for mask, count in hit_masks.most_common()),
    }


def _mask_string(mask: Any) -> str:
    if mask is None:
        return ""
    if isinstance(mask, str):
        return mask
    if isinstance(mask, list):
        return ",".join(str(x) for x in mask)
    return str(mask)


def _layer_counts(query: dict[str, Any]) -> tuple[int, int]:
    computed = query.get("prefix_vlm_cache_computed_blocks")
    reused = query.get("prefix_vlm_cache_reused_blocks", query.get("prefix_vlm_cache_skipped_blocks"))
    computed_int = int(computed) if computed is not None else PREFIX_LAYERS
    reused_int = int(reused) if reused is not None else 0
    computed_int = max(0, min(PREFIX_LAYERS, computed_int))
    reused_int = max(0, min(PREFIX_LAYERS, reused_int))
    return computed_int, reused_int


def _read_file(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _collect(paths: list[pathlib.Path], include_partial: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    configs: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    episodes_out: list[dict[str, Any]] = []

    for path in paths:
        payload = _read_file(path)
        if not payload:
            continue
        episodes = payload.get("episodes") or []
        expected = _expected_episodes(path, payload)
        completed = expected is None or len(episodes) >= expected
        if not include_partial and not completed:
            continue

        query_records = [query for episode in episodes for query in episode.get("queries", [])]
        tags = _config_tags(path, payload, query_records)
        total_layers = len(query_records) * PREFIX_LAYERS
        computed_layers = 0
        cached_layers = 0
        cache_hits = 0
        no_image_hits = 0
        shadow_observed = False
        shadow_hits = 0
        shadow_cached_layers = 0
        cka_values: list[float] = []
        hit_cka: list[float] = []
        miss_cka: list[float] = []
        reasons = Counter()
        policy_ms: list[float] = []

        for query in query_records:
            computed, cached = _layer_counts(query)
            computed_layers += computed
            cached_layers += cached
            hit = bool(query.get("prefix_vlm_cache_hit"))
            cache_hits += int(hit)
            no_image_hits += int(bool(query.get("prefix_vlm_cache_skip_v4_no_image")))
            if query.get("prefix_vlm_cache_observe_only") or query.get("prefix_vlm_cache_shadow_hit") is not None:
                shadow_observed = True
                shadow_hit = bool(query.get("prefix_vlm_cache_shadow_hit"))
                shadow_hits += int(shadow_hit)
                shadow_reused = query.get(
                    "prefix_vlm_cache_shadow_reused_blocks",
                    query.get("prefix_vlm_cache_shadow_skipped_blocks"),
                )
                shadow_cached_layers += int(shadow_reused) if shadow_reused is not None else 0
            reason = query.get("prefix_vlm_cache_reason")
            if reason:
                reasons[str(reason)] += 1
            cka = _float(query.get("prefix_vlm_cache_first_layer_cka"))
            if cka is not None:
                cka_values.append(cka)
                (hit_cka if hit else miss_cka).append(cka)
            ms = _float(query.get("policy_infer_ms"))
            if ms is not None:
                policy_ms.append(ms)

        successes = sum(1 for episode in episodes if episode.get("success"))
        config_row = {
            "file": path.name,
            "completed": completed,
            "expected_episodes": expected,
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "policy_queries": len(query_records),
            "total_prefix_layers": total_layers,
            "computed_layers": computed_layers,
            "cached_layers": cached_layers,
            "cache_layer_fraction": cached_layers / total_layers if total_layers else None,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(query_records) if query_records else None,
            "no_image_hits": no_image_hits,
            "shadow_observed": shadow_observed,
            "shadow_hits": shadow_hits,
            "shadow_hit_rate": shadow_hits / len(query_records) if shadow_observed and query_records else None,
            "shadow_cached_layers": shadow_cached_layers,
            "shadow_cache_layer_fraction": shadow_cached_layers / total_layers
            if shadow_observed and total_layers
            else None,
            "cached_layers_per_query": cached_layers / len(query_records) if query_records else None,
            "policy_infer_mean_ms": _mean(policy_ms),
            "cka_mean": _mean(cka_values),
            "cka_p10": _quantile(cka_values, 0.10),
            "cka_p50": _quantile(cka_values, 0.50),
            "cka_p90": _quantile(cka_values, 0.90),
            "hit_cka_mean": _mean(hit_cka),
            "miss_cka_mean": _mean(miss_cka),
            "top_reasons": "; ".join(f"{reason}:{count}" for reason, count in reasons.most_common(5)),
            **tags,
        }
        configs.append(config_row)

        by_task: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "episodes": 0,
                "successes": 0,
                "steps": 0,
                "queries": 0,
            "computed_layers": 0,
            "cached_layers": 0,
            "cache_hits": 0,
            "shadow_hits": 0,
            "shadow_cached_layers": 0,
            "cka": [],
            }
        )
        for episode in episodes:
            task_id = int(episode.get("task_id", -1))
            bucket = by_task[task_id]
            bucket["episodes"] += 1
            bucket["successes"] += int(bool(episode.get("success")))
            bucket["steps"] += int(episode.get("steps") or 0)
            bucket["queries"] += len(episode.get("queries", []))
            ep_computed = 0
            ep_cached = 0
            ep_hits = 0
            for query in episode.get("queries", []):
                computed, cached = _layer_counts(query)
                ep_computed += computed
                ep_cached += cached
                ep_hits += int(bool(query.get("prefix_vlm_cache_hit")))
                if query.get("prefix_vlm_cache_observe_only") or query.get("prefix_vlm_cache_shadow_hit") is not None:
                    bucket["shadow_hits"] += int(bool(query.get("prefix_vlm_cache_shadow_hit")))
                    shadow_reused = query.get(
                        "prefix_vlm_cache_shadow_reused_blocks",
                        query.get("prefix_vlm_cache_shadow_skipped_blocks"),
                    )
                    bucket["shadow_cached_layers"] += int(shadow_reused) if shadow_reused is not None else 0
                cka = _float(query.get("prefix_vlm_cache_first_layer_cka"))
                if cka is not None:
                    bucket["cka"].append(cka)
            bucket["computed_layers"] += ep_computed
            bucket["cached_layers"] += ep_cached
            bucket["cache_hits"] += ep_hits

            ep_queries = len(episode.get("queries", []))
            episodes_out.append(
                {
                    "file": path.name,
                    "task_id": task_id,
                    "episode_idx": int(episode.get("episode_idx", -1)),
                    "success": bool(episode.get("success")),
                    "steps": int(episode.get("steps") or 0),
                    "policy_queries": ep_queries,
                    "cached_layers": ep_cached,
                    "total_prefix_layers": ep_queries * PREFIX_LAYERS,
                    "cache_layer_fraction": ep_cached / (ep_queries * PREFIX_LAYERS) if ep_queries else None,
                "cache_hits": ep_hits,
                "cache_hit_rate": ep_hits / ep_queries if ep_queries else None,
                "shadow_hits": sum(
                    1 for query in episode.get("queries", []) if query.get("prefix_vlm_cache_shadow_hit")
                ),
                "shadow_cached_layers": sum(
                    int(
                        query.get(
                            "prefix_vlm_cache_shadow_reused_blocks",
                            query.get("prefix_vlm_cache_shadow_skipped_blocks") or 0,
                        )
                        or 0
                    )
                    for query in episode.get("queries", [])
                ),
                "prefix_only": tags["prefix_only"],
                    "completed_config": completed,
                }
            )

        for task_id, bucket in sorted(by_task.items()):
            total = bucket["queries"] * PREFIX_LAYERS
            tasks.append(
                {
                    "file": path.name,
                    "task_id": task_id,
                    "episodes": bucket["episodes"],
                    "successes": bucket["successes"],
                    "success_rate": bucket["successes"] / bucket["episodes"] if bucket["episodes"] else None,
                    "steps": bucket["steps"],
                    "policy_queries": bucket["queries"],
                    "cached_layers": bucket["cached_layers"],
                    "total_prefix_layers": total,
                    "cache_layer_fraction": bucket["cached_layers"] / total if total else None,
                    "cache_hits": bucket["cache_hits"],
                    "cache_hit_rate": bucket["cache_hits"] / bucket["queries"] if bucket["queries"] else None,
                    "shadow_hits": bucket["shadow_hits"],
                    "shadow_hit_rate": bucket["shadow_hits"] / bucket["queries"] if bucket["queries"] else None,
                    "shadow_cached_layers": bucket["shadow_cached_layers"],
                    "shadow_cache_layer_fraction": bucket["shadow_cached_layers"] / total if total else None,
                    "cka_mean": _mean(bucket["cka"]),
                    "prefix_only": tags["prefix_only"],
                    "action_cache_enabled": tags["action_cache_enabled"],
                    "completed_config": completed,
                }
            )

    configs.sort(key=lambda row: (row["cache_layer_fraction"] or -1), reverse=True)
    tasks.sort(key=lambda row: (row["cache_layer_fraction"] or -1), reverse=True)
    episodes_out.sort(key=lambda row: (row["cache_layer_fraction"] or -1), reverse=True)
    return configs, tasks, episodes_out


def _threshold_opportunities(configs: list[dict[str, Any]], paths_by_name: dict[str, pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_names = [
        row["file"]
        for row in configs
        if row.get("prefix_only")
        and row.get("completed")
        and row.get("success_rate") == 1.0
        and row.get("count") == 2
        and row.get("cka_threshold") is not None
    ]
    for file_name in target_names:
        payload = _read_file(paths_by_name[file_name])
        if not payload:
            continue
        queries = [query for episode in payload.get("episodes", []) for query in episode.get("queries", [])]
        base_cached = 0
        current_hits = 0
        total_layers = len(queries) * PREFIX_LAYERS
        for query in queries:
            _computed, cached = _layer_counts(query)
            base_cached += cached
            current_hits += int(bool(query.get("prefix_vlm_cache_hit")))
        for threshold in DEFAULT_THRESHOLDS:
            candidate_extra = 0
            blocked_by_continuous_or_warmup = 0
            for query in queries:
                if query.get("prefix_vlm_cache_hit"):
                    continue
                cka = _float(query.get("prefix_vlm_cache_first_layer_cka"))
                if cka is None or cka < threshold:
                    continue
                reason = str(query.get("prefix_vlm_cache_reason") or "")
                if reason in {"max_continuous_cached_queries", "warmup", "empty"}:
                    blocked_by_continuous_or_warmup += 1
                else:
                    candidate_extra += 1
            rows.append(
                {
                    "file": file_name,
                    "threshold": threshold,
                    "current_hits": current_hits,
                    "candidate_extra_hits": candidate_extra,
                    "blocked_extra_hits": blocked_by_continuous_or_warmup,
                    "optimistic_cached_layers": base_cached + candidate_extra * 16,
                    "optimistic_cache_layer_fraction": (base_cached + candidate_extra * 16) / total_layers
                    if total_layers
                    else None,
                }
            )
    rows.sort(key=lambda row: (row["file"], row["threshold"]))
    return rows


def _phase_rows(paths_by_name: dict[str, pathlib.Path], file_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in file_names:
        path = paths_by_name.get(file_name)
        if path is None:
            continue
        payload = _read_file(path)
        if not payload:
            continue
        buckets: dict[str, Counter] = {phase: Counter() for phase in ("early", "middle", "late")}
        for episode in payload.get("episodes", []):
            queries = episode.get("queries", [])
            query_count = len(queries)
            if query_count == 0:
                continue
            for index, query in enumerate(queries):
                frac = index / max(1, query_count - 1)
                if frac < 1 / 3:
                    phase = "early"
                elif frac < 2 / 3:
                    phase = "middle"
                else:
                    phase = "late"
                computed, cached = _layer_counts(query)
                buckets[phase]["queries"] += 1
                buckets[phase]["computed"] += computed
                buckets[phase]["cached"] += cached
                buckets[phase]["hits"] += int(bool(query.get("prefix_vlm_cache_hit")))
        for phase, counter in buckets.items():
            total = counter["queries"] * PREFIX_LAYERS
            rows.append(
                {
                    "file": file_name,
                    "phase": phase,
                    "queries": counter["queries"],
                    "hits": counter["hits"],
                    "hit_rate": counter["hits"] / counter["queries"] if counter["queries"] else None,
                    "cached_layers": counter["cached"],
                    "total_prefix_layers": total,
                    "cache_layer_fraction": counter["cached"] / total if total else None,
                }
            )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(header for header, _key in columns) + " |",
        "| " + " | ".join("---" for _header, _key in columns) + " |",
    ]
    for row in selected:
        values = []
        for _header, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                if key == "threshold" or key == "cka_threshold":
                    values.append(_num(value, 5))
                elif key.endswith("rate") or key.endswith("fraction") or key == "success_rate":
                    values.append(_pct(value))
                elif "cka" in key:
                    values.append(_num(value, 5))
                else:
                    values.append(_num(value, 2))
            elif isinstance(value, bool):
                values.append("yes" if value else "no")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: pathlib.Path,
    configs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    partial_configs: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    csv_paths: dict[str, pathlib.Path],
) -> None:
    prefix_full = [
        row
        for row in configs
        if row.get("prefix_only")
        and row.get("completed")
        and row.get("success_rate") == 1.0
        and not row.get("shadow_observed")
    ]
    broad_prefix = [
        row
        for row in prefix_full
        if row.get("episodes") == 30 and "tasks0_9_3trials" in row["file"]
    ]
    partial_prefix = [
        row for row in partial_configs if row.get("prefix_only") and not row.get("completed")
    ]
    high_cache_failures = [
        row
        for row in configs
        if row.get("completed") and (row.get("success_rate") or 0.0) < 1.0 and (row.get("cache_layer_fraction") or 0.0) > 0.10
    ]
    shadow_configs = [
        row for row in configs if row.get("shadow_observed") and row.get("completed")
    ]
    shadow_configs.sort(key=lambda row: row.get("shadow_cache_layer_fraction") or -1.0, reverse=True)

    best_prefix = prefix_full[0] if prefix_full else None
    best_broad = broad_prefix[0] if broad_prefix else None

    lines: list[str] = []
    lines.append("# OpenPI prefix cache observation report")
    lines.append("")
    lines.append("日期：2026-06-10")
    lines.append("")
    lines.append(
        "口径：每次 policy inference 的 prefix stack 固定按 18 层计数；hit 时按日志里的 `computed_blocks` / `reused_blocks` 分摊，miss 算 18 层计算、0 层 cache。默认排除未完成 rollout。"
    )
    lines.append("")
    lines.append("输出文件：")
    for label, csv_path in csv_paths.items():
        lines.append(f"- {label}: `{csv_path}`")
    lines.append("")
    lines.append("## 直接观察结论")
    lines.append("")
    if best_broad:
        lines.append(
            f"- 在 broad task0-9 x3 且 prefix-only 30/30 的约束下，当前最合理的是 `{best_broad['file']}`：cache layer `{best_broad['cached_layers']}/{best_broad['total_prefix_layers']} = {_pct(best_broad['cache_layer_fraction'])}`，hit `{best_broad['cache_hits']}/{best_broad['policy_queries']} = {_pct(best_broad['cache_hit_rate'])}`。"
        )
    if best_prefix:
        lines.append(
            f"- 如果只看所有 prefix-only 且完整成功的配置，最高 cache layer 是 `{best_prefix['file']}`：`{best_prefix['cached_layers']}/{best_prefix['total_prefix_layers']} = {_pct(best_prefix['cache_layer_fraction'])}`。"
        )
    if shadow_configs:
        shadow_best = shadow_configs[0]
        lines.append(
            f"- observe-only 观测里，`{shadow_best['file']}` 在不实际 skip 的情况下 shadow cache layer 为 `{shadow_best['shadow_cached_layers']}/{shadow_best['total_prefix_layers']} = {_pct(shadow_best['shadow_cache_layer_fraction'])}`，候选 hit `{shadow_best['shadow_hits']}/{shadow_best['policy_queries']} = {_pct(shadow_best['shadow_hit_rate'])}`。"
        )
    lines.append(
        "- `count=2` 比 `count=3` 更适合 broad：两者 broad 都是 30/30，但 `count=2` hit 时复用 16/18 层，而 `count=3` 只复用 15/18；实际 broad hit 数没有因为多算一层而变多。"
    )
    if partial_prefix:
        count1 = [row for row in partial_prefix if row.get("count") == 1]
        if count1:
            row = count1[0]
            lines.append(
                f"- `count=1` 只完成了 partial 观测，不纳入正式排行；它当前 `{row['successes']}/{row['episodes']}` 成功，但 cache layer 只有 `{row['cached_layers']}/{row['total_prefix_layers']} = {_pct(row['cache_layer_fraction'])}`，说明只算 1 层的命中太少，不值得优先 broad。"
            )
    lines.append(
        "- 降 CKA 阈值能明显增加 cache layer，但已完成结果里 `0.993/0.990` 档会掉成功率；所以更合理的下一步不是直接大幅放松阈值，而是先做 shadow/near-threshold 观测。"
    )
    lines.append("")

    lines.append("## Prefix-only 完整成功配置")
    lines.append("")
    lines.append(
        _md_table(
            prefix_full,
            [
                ("file", "file"),
                ("episodes", "episodes"),
                ("success", "success_rate"),
                ("queries", "policy_queries"),
                ("cached", "cached_layers"),
                ("cache frac", "cache_layer_fraction"),
                ("hit rate", "cache_hit_rate"),
                ("count", "count"),
                ("mc", "max_continuous"),
                ("cka", "cka_threshold"),
                ("mask", "hit_masks"),
            ],
            limit=12,
        )
    )
    lines.append("")

    if high_cache_failures:
        lines.append("## 高 cache 但掉成功率的对照")
        lines.append("")
        lines.append(
            _md_table(
                high_cache_failures,
                [
                    ("file", "file"),
                    ("success", "success_rate"),
                    ("episodes", "episodes"),
                    ("cached", "cached_layers"),
                    ("cache frac", "cache_layer_fraction"),
                    ("hit rate", "cache_hit_rate"),
                    ("cka", "cka_threshold"),
                    ("action cache", "action_cache_enabled"),
                ],
                limit=10,
            )
        )
        lines.append("")

    if partial_prefix:
        lines.append("## Partial / 中断观测")
        lines.append("")
        lines.append("这些文件没有完成预期 episode 数，只用于排查方向，不进入正式推荐。")
        lines.append("")
        lines.append(
            _md_table(
                partial_prefix,
                [
                    ("file", "file"),
                    ("episodes", "episodes"),
                    ("expected", "expected_episodes"),
                    ("success", "success_rate"),
                    ("queries", "policy_queries"),
                    ("cached", "cached_layers"),
                    ("cache frac", "cache_layer_fraction"),
                    ("hit rate", "cache_hit_rate"),
                    ("count", "count"),
                    ("mc", "max_continuous"),
                    ("cka", "cka_threshold"),
                ],
            )
        )
        lines.append("")

    if shadow_configs:
        lines.append("## Shadow-cache 观测")
        lines.append("")
        lines.append(
            "这些配置完整计算 prefix，不实际复用 cache；`shadow cached` 表示如果按同一规则开 cache，理论会复用多少 prefix layer。"
        )
        lines.append("")
        lines.append(
            _md_table(
                shadow_configs,
                [
                    ("file", "file"),
                    ("success", "success_rate"),
                    ("queries", "policy_queries"),
                    ("actual cached", "cached_layers"),
                    ("shadow cached", "shadow_cached_layers"),
                    ("shadow frac", "shadow_cache_layer_fraction"),
                    ("shadow hit rate", "shadow_hit_rate"),
                    ("count", "count"),
                    ("mc", "max_continuous"),
                    ("cka", "cka_threshold"),
                ],
            )
        )
        lines.append("")

    if best_broad:
        task_rows = [row for row in tasks if row["file"] == best_broad["file"]]
        task_rows.sort(key=lambda row: row["cache_layer_fraction"] or -1, reverse=True)
        lines.append("## 当前 broad 推荐配置的 task 分布")
        lines.append("")
        lines.append(
            _md_table(
                task_rows,
                [
                    ("task", "task_id"),
                    ("success", "success_rate"),
                    ("queries", "policy_queries"),
                    ("cached", "cached_layers"),
                    ("cache frac", "cache_layer_fraction"),
                    ("hit rate", "cache_hit_rate"),
                    ("cka mean", "cka_mean"),
                ],
            )
        )
        lines.append("")

    if opportunities:
        lines.append("## Near-threshold 观测")
        lines.append("")
        lines.append(
            "下面是对 prefix-only `count=2` 完整成功配置的离线估计：只看已有 query 的 first-layer CKA，假设更多 near-threshold miss 也能被安全 cache。这个表只能指导下一轮实验，不能当作成功率验证。"
        )
        lines.append("")
        lines.append(
            _md_table(
                opportunities,
                [
                    ("file", "file"),
                    ("threshold", "threshold"),
                    ("current hits", "current_hits"),
                    ("extra hits", "candidate_extra_hits"),
                    ("blocked", "blocked_extra_hits"),
                    ("optimistic cached", "optimistic_cached_layers"),
                    ("optimistic frac", "optimistic_cache_layer_fraction"),
                ],
                limit=28,
            )
        )
        lines.append("")

    if phases:
        lines.append("## Step phase 分布")
        lines.append("")
        lines.append(
            _md_table(
                phases,
                [
                    ("file", "file"),
                    ("phase", "phase"),
                    ("queries", "queries"),
                    ("hits", "hits"),
                    ("hit rate", "hit_rate"),
                    ("cached", "cached_layers"),
                    ("cache frac", "cache_layer_fraction"),
                ],
            )
        )
        lines.append("")

    lines.append("## 下一步建议")
    lines.append("")
    lines.append(
        "1. 把当前推荐参数固定为观测基线：`uniform_sparse count=2, CKA=0.9945, max_continuous=1`。"
    )
    lines.append(
        "2. 加一个 shadow-cache / observe-only 模式：每次仍完整计算 18 层，只记录不同 CKA 阈值和不同 layer mask 会不会命中，不真正复用 cache。这样可以在不扰动成功率的情况下定位安全窗口。"
    )
    lines.append(
        "3. 真正上线 sweep 时，只优先验证 near-threshold 表里收益最大的窄区间，例如 `0.9940-0.9945`，不要直接降到 `0.993` 或 `0.990`。"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    root = pathlib.Path.cwd()
    paths = sorted(root.glob(args.glob))
    configs, tasks, episodes = _collect(paths, include_partial=args.include_partial)
    all_configs, _all_tasks, _all_episodes = _collect(paths, include_partial=True)
    partial_configs = [row for row in all_configs if not row.get("completed")]
    paths_by_name = {path.name: path for path in paths}
    opportunities = _threshold_opportunities(configs, paths_by_name)
    phase_files = [
        "output_libero_broad_prefix_sparse_count2_cka09945_mc1_tasks0_9_3trials_20260610.json",
        "output_libero_broad_prefix_sparse_count3_cka09945_mc1_tasks0_9_3trials_20260610.json",
        "output_libero_prefix_sparse_count2_cka09945_mc2_tasks0_2_5trials_20260610.json",
    ]
    phases = _phase_rows(paths_by_name, phase_files)

    output_dir = pathlib.Path(args.output_dir)
    csv_paths = {
        "config": output_dir / f"prefix_cache_observation_configs_{args.date}.csv",
        "task": output_dir / f"prefix_cache_observation_tasks_{args.date}.csv",
        "episode": output_dir / f"prefix_cache_observation_episodes_{args.date}.csv",
        "near_threshold": output_dir / f"prefix_cache_observation_near_threshold_{args.date}.csv",
        "phase": output_dir / f"prefix_cache_observation_phases_{args.date}.csv",
    }
    _write_csv(csv_paths["config"], configs)
    _write_csv(csv_paths["task"], tasks)
    _write_csv(csv_paths["episode"], episodes)
    _write_csv(csv_paths["near_threshold"], opportunities)
    _write_csv(csv_paths["phase"], phases)
    _write_report(
        output_dir / f"prefix_cache_observation_report_{args.date}_zh.md",
        configs,
        tasks,
        episodes,
        partial_configs,
        opportunities,
        phases,
        csv_paths,
    )


if __name__ == "__main__":
    main()
