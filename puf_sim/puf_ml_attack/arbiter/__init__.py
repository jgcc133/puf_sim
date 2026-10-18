"""ML attack modules for Arbiter PUF variants."""
from __future__ import annotations

from typing import Any

import numpy as np

from pypuf.simulation import ArbiterPUF

from ..common import AgentTrainer as _BaseAgentTrainer, TorchPUFModel as _BaseTorchPUFModel
from . import feed_forward, interpose, xor_apuf

STAGE_NAME = "arbiter_mlp_ensemble"


def feature_map(challenges: np.ndarray) -> np.ndarray:
    """Apply the Arbiter Threshold Transform used by pypuf's ArbiterPUF."""
    return ArbiterPUF.transform_atf(np.asarray(challenges), 1)[:, 0, :].copy()


class AgentTrainer(_BaseAgentTrainer):
    """Train on Arbiter Threshold Transformed challenge features."""

    @staticmethod
    def feature_map(challenges: np.ndarray) -> np.ndarray:
        return feature_map(challenges)


class TorchPUFModel(_BaseTorchPUFModel):
    """Apply the Arbiter Threshold Transform before bipolar model inference."""

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return super().eval(feature_map(challenges))


__all__ = [
    "AgentTrainer",
    "TorchPUFModel",
    "STAGE_NAME",
    "feature_map",
    "feed_forward",
    "interpose",
    "xor_apuf",
]
