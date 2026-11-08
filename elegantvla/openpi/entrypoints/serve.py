import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Release path, matching the GR00T schedule and cache lifecycle.
    temporal_skip: _policy.TemporalSkipConfig = dataclasses.field(default_factory=_policy.TemporalSkipConfig)

    # Compatibility with earlier prefix-cache experiments; do not enable both.
    prefix_vlm_cache: _policy.PrefixVLMCacheConfig = dataclasses.field(default_factory=_policy.PrefixVLMCacheConfig)

    # Optional override for PyTorch Pi policies. If unset, use the training config default.
    pytorch_compile_mode: str | None = None
    # Force-disable torch.compile for PyTorch Pi policies. Useful for eval sweeps
    # where first-query compilation cost would dominate small pilots.
    disable_pytorch_compile: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    prefix_vlm_cache: _policy.PrefixVLMCacheConfig | None = None,
    temporal_skip: _policy.TemporalSkipConfig | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            prefix_vlm_cache=prefix_vlm_cache,
            temporal_skip=temporal_skip,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.disable_pytorch_compile:
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
                )
            elif args.pytorch_compile_mode is not None:
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(train_config.model, pytorch_compile_mode=args.pytorch_compile_mode),
                )
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                prefix_vlm_cache=args.prefix_vlm_cache,
                temporal_skip=args.temporal_skip,
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt,
                prefix_vlm_cache=args.prefix_vlm_cache, temporal_skip=args.temporal_skip,
            )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
