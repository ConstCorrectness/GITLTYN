"""Core ablation primitives for Grammar is the Last Thing You Need (GITLTYN)."""

from .config import (
    DEFAULT_CONFIG_PATH,
    LatentTeeConfig,
    RegimeExperimentConfig,
    load_regime_configs,
)
from .latent_loop import (
    ContinuousLatentLoop,
    LatentStepRecord,
    LatentTeeRingBuffer,
    LogitLensPrediction,
    RingBufferLatentTee,
    SLMArchitectureSpec,
    SidecarProbe,
    TrackedLatentTensor,
    estimate_regime_compute,
    extract_detached_array,
)
from .prune_syntax import (
    ClassifiedToken,
    PrunedTraceSummary,
    TokenCategory,
    classify_cot_tokens,
    prune_cot_trace,
)

__all__ = [
    "ClassifiedToken",
    "ContinuousLatentLoop",
    "DEFAULT_CONFIG_PATH",
    "LatentStepRecord",
    "LatentTeeConfig",
    "LatentTeeRingBuffer",
    "LogitLensPrediction",
    "PrunedTraceSummary",
    "RegimeExperimentConfig",
    "RingBufferLatentTee",
    "SLMArchitectureSpec",
    "SidecarProbe",
    "TokenCategory",
    "TrackedLatentTensor",
    "classify_cot_tokens",
    "estimate_regime_compute",
    "extract_detached_array",
    "load_regime_configs",
    "prune_cot_trace",
]
