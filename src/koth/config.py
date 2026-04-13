"""Configuration for the King of the Hill system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Parameters for the sequential sign test evaluation."""

    N: int = 10_000
    alpha: float = 0.01
    sequence_length: int = 2048
    batch_size: int = 1
    use_amp: bool = True
    amp_dtype: str = "bfloat16"


@dataclass
class BoundingBoxConfig:
    """Constraints on the delta between challenger and king."""

    max_linf: float = 0.5
    max_l2_per_tensor: float | None = None
    max_l2_global: float | None = None
    max_sparsity_frac: float | None = None
    frozen_param_prefixes: list[str] = field(default_factory=list)


@dataclass
class R2Config:
    """Cloudflare R2 bucket configuration."""

    endpoint_url: str = ""
    bucket_name: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "auto"

    @property
    def configured(self) -> bool:
        return bool(self.endpoint_url and self.bucket_name and self.access_key_id)


@dataclass
class KingConfig:
    """Configuration for the king model lifecycle."""

    hf_repo: str = ""
    hf_token: str = ""
    local_cache_dir: str = "/tmp/koth/king"


@dataclass
class ChainConfig:
    """Bittensor chain configuration."""

    netuid: int = 3
    network: str = "finney"
    wallet_name: str = "default"
    wallet_hotkey: str = "default"


@dataclass
class PodConfig:
    """Lium/Celium pod configuration for ephemeral eval."""

    gpu_type: str = "rtx4090"
    ssh_key_path: str = "~/.ssh/id_rsa"
    startup_timeout_s: int = 300
    eval_timeout_s: int = 1800


@dataclass
class KOTHConfig:
    """Top-level configuration for the entire system."""

    eval: EvalConfig = field(default_factory=EvalConfig)
    bounding_box: BoundingBoxConfig = field(default_factory=BoundingBoxConfig)
    r2: R2Config = field(default_factory=R2Config)
    king: KingConfig = field(default_factory=KingConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    pod: PodConfig = field(default_factory=PodConfig)
    poll_interval_s: int = 12
