# Standard
from collections import defaultdict
from enum import StrEnum, auto  # Python3.11+

# Local
from vllm_detector_adapter.generative_detectors import (
    base,
    granite_guardian,
    llama_guard,
)

# To only expose certain objects from this file
__all__ = ["ModelTypes", "MODEL_CLASS_MAP"]


class CaseInsensitiveStrEnum(StrEnum):
    """Custom StrEnum to make member lookup case insensitive"""

    @classmethod
    def _missing_(cls, value):
        return cls.__members__.get(value.upper(), None)


class ModelTypes(CaseInsensitiveStrEnum):

    GRANITE_GUARDIAN = auto()
    LLAMA_GUARD = auto()


# Use ChatCompletionDetectionBase as the base class
def __default_detection_base__():
    return base.ChatCompletionDetectionBase


# Dictionary mapping generative detection classes with model types.
# This gets used for configuring which detection processing to use with which model.
MODEL_CLASS_MAP = defaultdict(__default_detection_base__)

# This is to add values to above MAP. This is private, to discourage these values to
# be used directly.
__MODEL_CLASS_MAP__ = {
    ModelTypes.GRANITE_GUARDIAN: granite_guardian.GraniteGuardian,
    ModelTypes.LLAMA_GUARD: llama_guard.LlamaGuard,
}

# Feede all the values to the MODEL_CLASS_MAP
for key, value in __MODEL_CLASS_MAP__.items():
    MODEL_CLASS_MAP[key] = value
