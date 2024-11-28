# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger

logger = init_logger(__name__)


class LlamaGuard(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"

    # Model specific tokens
    SAFE_TOKEN = "safe"
    UNSAFE_TOKEN = "unsafe"

    # NOTE: More intelligent template parsing can be done here, potentially
    # as a regex template for safe vs. unsafe and the 'unsafe' category
