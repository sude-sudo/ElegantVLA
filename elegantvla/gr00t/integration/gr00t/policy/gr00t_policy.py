"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from pathlib import Path
from typing import Any
import time
import os

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        num_inference_steps: int | None = None,
        strict: bool = True,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            num_inference_steps: Optional override for the number of inference steps
            strict: Whether to enforce strict input validation (default: True)
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        self.realtime_eagle = os.environ.get(
            "GROOT_REALTIME_EAGLE", "0"
        ).lower() in ("1", "true", "yes", "on")

        # GR00T consumes Eagle hidden states, not vocabulary logits.  Keeping
        # this exact, opt-out optimization separate from the experimental
        # realtime kernels makes baseline/A-B comparisons straightforward.
        self.skip_unused_lm_head = os.environ.get(
            "GROOT_SKIP_UNUSED_LM_HEAD",
            "1" if self.realtime_eagle else "0",
        ).lower() in ("1", "true", "yes", "on")
        if self.skip_unused_lm_head:
            language_model = self.model.backbone.model.language_model
            if hasattr(language_model, "lm_head"):
                language_model.lm_head = torch.nn.Identity()
                print("[Gr00tPolicy] 跳过未使用的 Eagle 词表 lm_head")

        # RealtimeGR00T optimizations are inference-only and opt-in.  The
        # CUDA-Graph pieces are installed independently from Triton so a
        # machine without Triton can still use fixed-shape graph replay.
        self.realtime_eagle_kernel_info: dict[str, Any] = {}
        self.qwen_cuda_graph = False
        self.qwen_boundary_cuda_graphs = 0
        self.vision_cuda_graph = False

        use_cuda = self.model.device.type == "cuda" and torch.cuda.is_available()
        static_qwen_attention = os.environ.get(
            "GROOT_STATIC_EAGLE_FORWARD",
            "1" if self.realtime_eagle else "0",
        ).lower() in ("1", "true", "yes", "on") and use_cuda
        cuda_graph_qwen = os.environ.get(
            "GROOT_CUDA_GRAPH_EAGLE", "0"
        ).lower() in ("1", "true", "yes", "on") and use_cuda
        # Temporal head/tail recomputation bypasses QwenModel.forward and
        # calls layer boundaries directly. In realtime mode, use boundary
        # graphs by default unless the full-decoder graph was explicitly
        # selected. The two graph granularities are mutually exclusive.
        cuda_graph_qwen_boundaries = os.environ.get(
            "GROOT_CUDA_GRAPH_QWEN_BOUNDARIES",
            "1" if self.realtime_eagle and not cuda_graph_qwen else "0",
        ).lower() in ("1", "true", "yes", "on") and use_cuda
        cuda_graph_vision = os.environ.get(
            "GROOT_CUDA_GRAPH_VISION", "0"
        ).lower() in ("1", "true", "yes", "on") and use_cuda

        # Triton launchers need a compiler-visible Python.h. Keep them
        # explicitly opt-in because many shared environments expose Triton
        # without the matching Python development headers.
        triton_requested = os.environ.get(
            "GROOT_TRITON_EAGLE_KERNELS",
            os.environ.get("GROOT_TRITON_EAGLE", "0"),
        )
        self.triton_eagle_kernels = os.environ.get(
            "GROOT_TRITON_EAGLE_KERNELS", triton_requested
        ).lower() in ("1", "true", "yes", "on") and use_cuda
        if self.triton_eagle_kernels:
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_realtime_eagle_kernels,
            )

            kernel_info = install_realtime_eagle_kernels(
                self.model,
                fuse_gate_up=os.environ.get(
                    "GROOT_FUSED_EAGLE_GATE_UP",
                    "1" if self.realtime_eagle else "0",
                ).lower() in ("1", "true", "yes", "on"),
                fuse_qkv=os.environ.get(
                    "GROOT_FUSED_EAGLE_QKV",
                    "1" if self.realtime_eagle else "0",
                ).lower() in ("1", "true", "yes", "on"),
                fuse_siglip_fc2_residual=os.environ.get(
                    "GROOT_FUSED_SIGLIP_FC2_RESIDUAL",
                    "1" if self.realtime_eagle else "0",
                ).lower() in ("1", "true", "yes", "on"),
                static_qwen_attention=static_qwen_attention,
            )
            self.realtime_eagle_kernel_info = kernel_info
            if not kernel_info.get("triton_available", False):
                self.triton_eagle_kernels = False
                print("[Gr00tPolicy] Triton 不可用，保留 Eagle 原始算子")
            else:
                print(
                    "[Gr00tPolicy] Eagle Realtime 算子已安装: "
                    f"RMSNorm={kernel_info['rms_norms']}, "
                    f"SwiGLU={kernel_info['mlps']}, "
                    f"packed_QKV={kernel_info['qkv_projections']}, "
                    f"Siglip_FC2残差={kernel_info['siglip_fc2_residuals']}, "
                    f"Static_Qwen={kernel_info['static_qwen_attentions']}, "
                    f"fused_gate_up={kernel_info['fused_gate_up']}"
                )

        # Static attention is useful even when Triton is unavailable.  This
        # fallback is only enabled for the fixed batch-1 deployment path.
        if static_qwen_attention and not self.realtime_eagle_kernel_info.get(
            "static_qwen_attentions", 0
        ):
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_static_qwen_attention,
            )

            self.realtime_eagle_kernel_info["static_qwen_attentions"] = (
                install_static_qwen_attention(self.model)
            )

        if cuda_graph_qwen:
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_qwen_cuda_graph,
            )

            self.qwen_cuda_graph = bool(install_qwen_cuda_graph(self.model))
            print(f"[Gr00tPolicy] Eagle Qwen CUDA Graph={self.qwen_cuda_graph}")

        if cuda_graph_qwen_boundaries and not cuda_graph_qwen:
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_qwen_boundary_cuda_graphs,
            )

            self.qwen_boundary_cuda_graphs = install_qwen_boundary_cuda_graphs(
                self.model
            )
            print(
                "[Gr00tPolicy] Eagle Qwen boundary CUDA Graphs="
                f"{self.qwen_boundary_cuda_graphs} (temporal head/tail)"
            )

        if cuda_graph_vision:
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_siglip_encoder_cuda_graph,
            )

            self.vision_cuda_graph = bool(
                install_siglip_encoder_cuda_graph(self.model)
            )
            print(
                f"[Gr00tPolicy] Eagle Vision CUDA Graph={self.vision_cuda_graph}"
            )

        self.action_dit_cuda_graph = os.environ.get(
            "GROOT_CUDA_GRAPH_ACTION_DIT", "1" if self.realtime_eagle else "0"
        ).lower() in ("1", "true", "yes", "on") and use_cuda
        if self.action_dit_cuda_graph:
            from gr00t.model.modules.realtime_eagle_kernels import (
                install_action_dit_cuda_graph,
            )

            installed = install_action_dit_cuda_graph(self.model)
            self.action_dit_cuda_graph = bool(installed)
            print(f"[Gr00tPolicy] Action DiT CUDA Graph={self.action_dit_cuda_graph}")

        self.num_inference_steps = num_inference_steps

        # Load the processor for input/output transformation
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            # Check that the expected video key exists in the observation
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            # Check that the expected state key exists in the observation
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            # Check that the expected language key exists in the observation
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        preprocess_start_ns = time.perf_counter_ns()

        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        unbatch_end_ns = time.perf_counter_ns()
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))
        processor_end_ns = time.perf_counter_ns()

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collate_end_ns = time.perf_counter_ns()
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
        dtype_end_ns = time.perf_counter_ns()

        # Step 4: Run model inference to predict actions
        with torch.inference_mode():
            action_head = getattr(self.model, "action_head", None)
            original_num_inference_timesteps = None
            if action_head is not None and self.num_inference_steps is not None:
                original_num_inference_timesteps = action_head.num_inference_timesteps
                action_head.num_inference_timesteps = self.num_inference_steps
            try:
                model_pred = self.model.get_action(**collated_inputs)
            finally:
                if action_head is not None and original_num_inference_timesteps is not None:
                    action_head.num_inference_timesteps = original_num_inference_timesteps
        normalized_action = model_pred["action_pred"].float()
        model_timing = dict(model_pred.get("timing", {}))
        model_end_ns = time.perf_counter_ns()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        decode_end_ns = time.perf_counter_ns()
        model_timing.update(
            {
                "policy_preprocess_latency_ms": (dtype_end_ns - preprocess_start_ns) / 1e6,
                "policy_preprocess_unbatch_latency_ms": (unbatch_end_ns - preprocess_start_ns) / 1e6,
                "policy_preprocess_processor_latency_ms": (processor_end_ns - unbatch_end_ns) / 1e6,
                "policy_preprocess_collate_latency_ms": (collate_end_ns - processor_end_ns) / 1e6,
                "policy_preprocess_to_dtype_latency_ms": (dtype_end_ns - collate_end_ns) / 1e6,
                "policy_model_wrapper_latency_ms": (model_end_ns - dtype_end_ns) / 1e6,
                "policy_decode_latency_ms": (decode_end_ns - model_end_ns) / 1e6,
            }
        )
        return casted_action, {"timing": model_timing}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
