"""Cache and threshold scheduler configuration for OpenPI."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class PrefixVLMCacheConfig:
    """Training-free cache for the image/language prefix between environment steps."""

    enabled: bool = False
    # Observe candidate cache hits without actually reusing cached prefix states.
    # This keeps actions on the exact full-compute path while reporting shadow
    # hit/cache-layer counters for threshold and mask studies.
    observe_only: bool = False
    # In observe_only mode, also run the candidate cached prefix path on shadow
    # hits and compare its final output-space actions with the full actions.
    # This gives an offline safety label for gate fitting while the environment
    # still executes the full-compute action.
    action_deviation_label_enabled: bool = False
    # Which shadow candidates receive full-vs-cached action labels:
    # "accepted" follows the current cache policy exactly; "pre_guard" also
    # labels candidates accepted by cheap prompt/image/state checks before the
    # first-layer CKA/L2 and motion-tier guards; "available" labels any same-
    # prompt cached prefix state. The environment still executes full actions.
    action_deviation_candidate_mode: str = "accepted"
    action_deviation_l2_threshold: float = -1.0
    action_deviation_translation_l2_threshold: float = -1.0
    action_deviation_rotation_l2_threshold: float = -1.0
    action_deviation_eef_position_l2_threshold: float = -1.0
    action_deviation_eef_rotation_l2_threshold: float = -1.0
    action_deviation_eef_pose_l2_threshold: float = -1.0
    action_deviation_gripper_abs_threshold: float = -1.0
    action_deviation_allow_gripper_flip: bool = True
    action_deviation_gripper_flip_eps: float = 1e-3
    observe_similarity_matrix: bool = False
    observe_similarity_max_queries: int = 64
    observe_layer_similarity: bool = False
    observe_layer_similarity_matrix: bool = False
    observe_layer_similarity_matrix_max_queries: int = 64
    observe_sublayer_similarity: bool = False
    observe_adjacent_layer_similarity: bool = False
    observe_lagged_adjacent_layer_similarity: bool = False
    observe_layer_cka_threshold: float = 0.994
    sublayer_cache_enabled: bool = False
    sublayer_cache_attn_layer_mask: str | None = None
    sublayer_cache_mlp_layer_mask: str | None = None
    sublayer_cache_min_first_layer_cka: float = -1.0
    cross_layer_residual_cache_enabled: bool = False
    cross_layer_residual_layer_mask: str | None = None
    cache_bank_enabled: bool = False
    cache_bank_size: int = 4
    fn_compute_blocks: int = 2
    bn_compute_blocks: int = 2
    # Optional explicit comma-separated layer ids/ranges to recompute on a prefix-cache hit.
    # Examples: "0,1,4,7,16-17" or "0-2,5,8". When set, this overrides fn/bn blocks.
    compute_layer_mask: str | None = None
    # Strategy used when compute_layer_count is positive and compute_layer_mask
    # is unset. "first_last" preserves the fn/bn block behavior, while
    # "uniform_sparse" spreads recomputed anchor layers across the prefix stack.
    compute_layer_strategy: str = "first_last"
    compute_layer_count: int = 0
    max_warmup_queries: int = 1
    max_continuous_cached_queries: int = 1
    image_signature_l2_threshold: float = 0.05
    # TeaCache-style online drift budget. When non-negative, consecutive
    # accepted cache hits accumulate image-signature deltas; exceeding the
    # budget forces a full prefix refresh and resets the accumulator.
    image_signature_accum_l2_threshold: float = -1.0
    # Negative disables the first-layer guard. When enabled, PyTorch prefix cache
    # compares the current first prefix block output against the cached refresh.
    first_layer_l2_threshold: float = -1.0
    # Guard metric for the first prefix block. "l2" uses relative L2 distance;
    # "cka" uses linear CKA over valid prefix tokens and accepts when similarity
    # is above first_layer_cka_threshold.
    first_layer_metric: str = "l2"
    first_layer_cka_threshold: float = 0.995
    # Multi-tier prefix cache controller. It uses online end-effector/state
    # motion and first-layer prefix CKA to choose how many prefix layers to
    # recompute. The intended behavior is aggressive cache while the arm is
    # moving quickly, and more exact compute during slow alignment/contact.
    motion_tier_cache_enabled: bool = False
    motion_tier_slow_speed_threshold: float = 0.025
    motion_tier_fast_speed_threshold: float = 0.055
    # Optional episode-local override for the fast boundary. This can increase
    # fast-tier reuse during early motion while preserving the default boundary
    # for later alignment/contact queries.
    motion_tier_early_fast_speed_threshold: float = -1.0
    motion_tier_early_fast_max_episode_query: int = 0
    # Optional top-speed tier. When accepted, "none" recomputes no prefix
    # transformer layers and reuses the cached prefix residuals for all blocks.
    motion_tier_very_fast_speed_threshold: float = -1.0
    motion_tier_very_fast_max_episode_query: int = 0
    motion_tier_very_fast_compute_layer_mask: str = "none"
    motion_tier_very_fast_fallback_compute_layer_mask: str | None = None
    motion_tier_slow_compute_layer_mask: str = "0-13"
    motion_tier_medium_compute_layer_mask: str = "0,1,16,17"
    motion_tier_fast_compute_layer_mask: str = "0,17"
    motion_tier_medium_high_cka_compute_layer_mask: str | None = None
    motion_tier_medium_high_cka_threshold: float = -1.0
    motion_tier_fast_high_cka_compute_layer_mask: str | None = None
    motion_tier_fast_high_cka_threshold: float = -1.0
    motion_tier_fast_fallback_compute_layer_mask: str | None = None
    # Optional early-rollout fastburst override. This lets the controller use a
    # more aggressive fast-tier mask only for the first N policy queries, then
    # fall back to the normal fast/fallback masks for alignment and placement.
    motion_tier_fast_aggressive_max_episode_query: int = 0
    motion_tier_fast_aggressive_compute_layer_mask: str | None = None
    motion_tier_fast_aggressive_cka_threshold: float = -1.0
    motion_tier_fast_aggressive_cached_state_delta_threshold: float = -1.0
    motion_tier_fast_aggressive_fallback_compute_layer_mask: str | None = None
    motion_tier_fast_aggressive_fallback_cka_threshold: float = -1.0
    motion_tier_fast_aggressive_disable_prompt_substrings: str | None = None
    motion_tier_fast_aggressive_conservative_prompt_substrings: str | None = None
    motion_tier_fast_aggressive_conservative_max_episode_query: int = 0
    motion_tier_fast_aggressive_conservative_compute_layer_mask: str | None = None
    motion_tier_fast_aggressive_conservative_cka_threshold: float = -1.0
    motion_tier_fast_aggressive_conservative_cached_state_delta_threshold: float = -1.0
    motion_tier_fast_aggressive_conservative_fallback_compute_layer_mask: str | None = None
    motion_tier_fast_aggressive_conservative_fallback_cka_threshold: float = -1.0
    # Slow-tier reuse was unsafe during fine alignment; keep it effectively disabled.
    motion_tier_slow_cka_threshold: float = 2.0
    motion_tier_medium_cka_threshold: float = 0.9945
    motion_tier_fast_cka_threshold: float = 0.9935
    motion_tier_very_fast_cka_threshold: float = 0.995
    motion_tier_very_fast_fallback_cka_threshold: float = -1.0
    motion_tier_fast_fallback_cka_threshold: float = -1.0
    motion_tier_slow_cached_state_delta_threshold: float = -1.0
    motion_tier_medium_cached_state_delta_threshold: float = -1.0
    motion_tier_fast_cached_state_delta_threshold: float = -1.0
    motion_tier_very_fast_cached_state_delta_threshold: float = -1.0
    motion_tier_slow_no_image_budget: int = 0
    motion_tier_medium_no_image_budget: int = 0
    motion_tier_fast_no_image_budget: int = 0
    motion_tier_slow_no_image_cka_threshold: float = -1.0
    motion_tier_medium_no_image_cka_threshold: float = -1.0
    motion_tier_fast_no_image_cka_threshold: float = -1.0
    motion_tier_no_image_min_speed: float = -1.0
    # Guard whole-prefix no-image reuse by drift from the current cache anchor.
    # This is stricter than the one-step motion tier: no-image reuse does not
    # recompute a first-layer probe, so a stale anchor needs a cheap state brake.
    motion_tier_no_image_cached_state_delta_threshold: float = -1.0
    # Per-reset guard against using cached prefix states too densely in one rollout.
    # A value >= 1.0 disables the budget.
    max_episode_hit_ratio: float = 0.25
    prompt_max_episode_hit_ratio_substrings: str | None = None
    prompt_max_episode_hit_ratio: float = 0.0
    min_episode_queries_before_cache: int = 0
    prompt_disable_cache_substrings: str | None = None
    # Optional softer alternative to prompt_disable_cache_substrings. Matching
    # prompts are allowed to cache, but with extra CKA/budget checks and,
    # optionally, a more exact recompute mask.
    prompt_conservative_cache_substrings: str | None = None
    prompt_conservative_first_layer_cka_threshold: float = -1.0
    prompt_conservative_compute_layer_mask: str | None = None
    prompt_conservative_high_first_layer_cka_threshold: float = -1.0
    prompt_conservative_high_compute_layer_mask: str | None = None
    prompt_conservative_max_episode_hit_ratio: float = -1.0
    prompt_conservative_max_continuous_cached_queries: int = 0
    # If positive, only allow prefix-cache hits up to this 1-based policy query
    # index within the current reset/episode. This is a simple online phase
    # guard learned from offline cache-safety labels.
    max_cache_episode_query: int = 30
    prompt_max_cache_episode_query_substrings: str | None = None
    prompt_max_cache_episode_query: int = 0
    prompt2_max_cache_episode_query_substrings: str | None = None
    prompt2_max_cache_episode_query: int = 0
    prompt3_max_cache_episode_query_substrings: str | None = None
    prompt3_max_cache_episode_query: int = 0
    prompt4_max_cache_episode_query_substrings: str | None = None
    prompt4_max_cache_episode_query: int = 0
    # When the phase guard says no future cache hits are allowed in this reset,
    # bypass the split cache path and use the native fused sample_actions path.
    adaptive_bypass_after_cache_window: bool = True
    # Optional JSON produced by scripts/train_prefix_cache_scheduler.py. When
    # set, the scheduler chooses a per-query recompute mask from online features.
    adaptive_scheduler_model_path: str | None = None
    # Optional JSON produced by scripts/fit_action_deviation_gate.py. When set,
    # this learned gate replaces the legacy first-layer/motion-tier accept
    # guards. CKA/L2 remain available as input features, but they are not the
    # hard-coded final admission criterion.
    action_deviation_gate_model_path: str | None = None
    action_deviation_gate_disable_no_image: bool = True
    # When a learned action-deviation gate is loaded, skip legacy phase/image/
    # state/adaptive-scheduler rejection rules so those signals become gate
    # features instead of hard-coded final admission criteria. Structural guards
    # such as empty cache, warmup, prompt mismatch, and explicit prompt disables
    # still apply.
    action_deviation_gate_replace_legacy_guards: bool = True
    # Optional observation gate that uses informative online observations rather
    # than hard phase rules. These raw-image thresholds came from the rich
    # observation sweep: safe cache hits often had small base-camera drift but
    # nontrivial wrist-camera drift.
    adaptive_observation_gate_enabled: bool = False
    observation_base_image_mean_abs_max: float = -1.0
    observation_wrist_image_rel_l2_min: float = -1.0
    # Online phase guard based on raw proprio/state deltas between policy queries.
    # Negative thresholds disable the corresponding check. These checks are cheap
    # and are intended to reject contact/rotation/gripper transition phases before
    # attempting a more expensive prefix-cache hit.
    state_velocity_guard_enabled: bool = False
    state_delta_threshold: float = -1.0
    state_translation_delta_threshold: float = -1.0
    state_rotation_delta_threshold: float = -1.0
    state_gripper_delta_threshold: float = -1.0
    cached_state_delta_threshold: float = -1.0
    min_stable_state_deltas_for_cache: int = 1
    pytorch_compile_hit_path: bool = False
    pytorch_compile_mode: str = "reduce-overhead"
    # Experimental cache for the action expert during the diffusion loop.
    # When enabled, a prefix-cache hit can reuse previous-environment-step
    # action-expert residuals for the middle suffix layers at each diffusion step.
    action_suffix_cache_enabled: bool = False
    action_suffix_fn_compute_blocks: int = 1
    action_suffix_bn_compute_blocks: int = 0
    # DiT-style cache inside one action denoising trajectory. This keeps the
    # current prefix/state exact, fully refreshes the action expert every
    # intra_action_cache_refresh_interval diffusion steps, and reuses residuals
    # from the previous diffusion step for middle action-expert layers.
    intra_action_cache_enabled: bool = False
    intra_action_cache_fn_compute_blocks: int = 2
    intra_action_cache_bn_compute_blocks: int = 2
    intra_action_cache_schedule: str = "refresh_interval"
    intra_action_cache_refresh_interval: int = 3
    # If enabled, only use intra-denoise action cache when the prefix cache was
    # also accepted for this policy query. This ties action-head approximation to
    # the same CKA/state safety gate used by the prefix cache.
    intra_action_cache_requires_prefix_hit: bool = False
    intra_action_cache_compile: bool = False
    intra_action_cache_compile_mode: str = "reduce-overhead"
    # ElegantVLA image-free prefix reuse. After an accepted probed
    # prefix-cache hit, future queries can skip image/prompt embedding and prefix
    # prefill entirely, reusing the cached prefix KV and prefix masks.
    skip_v4_no_image_budget_after_hit: int = 0
    # Optional prompt deny-list for no-image reuse. This lets risky prompts keep
    # ordinary probed prefix cache while preventing whole-prefix KV reuse.
    skip_v4_no_image_disable_prompt_substrings: str | None = None
    # Rule scheduler shared with the GR00T method. The first-layer CKA probe
    # controls how much future no-image prefix reuse is allowed, while state
    # motion controls how aggressive the intra-denoise action cache should be.
    threshold_scheduler_enabled: bool = False
    threshold_scheduler_max_no_image_budget: int = 1
    threshold_scheduler_free1_cka_threshold: float = 0.999
    threshold_scheduler_free2_cka_threshold: float = 0.9995
    threshold_scheduler_free3_cka_threshold: float = 0.9998
    threshold_scheduler_action_head_enabled: bool = False
    threshold_scheduler_action_head_selection_mode: str = "motion_speed_inner"
    threshold_scheduler_ah_speed_threshold: float = 0.005
    threshold_scheduler_ah_secondary_threshold: float = 0.05
    threshold_scheduler_ah_reuse_middle_refresh_interval: int = 0
    threshold_scheduler_ah_reuse_middle_fn_compute_blocks: int = -1
    threshold_scheduler_ah_reuse_middle_bn_compute_blocks: int = -1
    threshold_scheduler_ah_reuse_all_refresh_interval: int = 0
    threshold_scheduler_ah_reuse_all_fn_compute_blocks: int = 1
    threshold_scheduler_ah_reuse_all_bn_compute_blocks: int = 0
    threshold_scheduler_eef_rotation_speed_weight: float = 1.0
    threshold_scheduler_gripper_speed_weight: float = 1.0
    # Smooth online motion over recent policy queries. The scheduler currently
    # uses translation only; rotation/gripper deltas are logged but not mixed
    # into the motion signal.
    threshold_scheduler_motion_buffer_size: int = 5
