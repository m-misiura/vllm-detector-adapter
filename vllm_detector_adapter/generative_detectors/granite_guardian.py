# Standard
from typing import Union

# Third Party
from vllm.entrypoints.openai.protocol import ErrorResponse

# Local
from vllm_detector_adapter.generative_detectors.base import ChatCompletionDetectionBase
from vllm_detector_adapter.logging import init_logger
from vllm_detector_adapter.protocol import ChatDetectionRequest

logger = init_logger(__name__)


class GraniteGuardian(ChatCompletionDetectionBase):

    DETECTION_TYPE = "risk"
    # User text pattern in task template
    USER_TEXT_PATTERN = "user_text"

    # Model specific tokens
    SAFE_TOKEN = "No"
    UNSAFE_TOKEN = "Yes"

    def preprocess(
        self, request: ChatDetectionRequest
    ) -> Union[ChatDetectionRequest, ErrorResponse]:
        """Granite guardian specific parameter updates for risk name and risk definition"""
        # Validation that one of the 'defined' risks is requested will be
        # done through the chat template on each request. Errors will
        # be propagated for chat completion separately
        guardian_config = {}
        if not request.detector_params:
            return request

        if risk_name := request.detector_params.pop("risk_name", None):
            guardian_config["risk_name"] = risk_name
        if risk_definition := request.detector_params.pop("risk_definition", None):
            guardian_config["risk_definition"] = risk_definition
        if guardian_config:
            logger.debug("guardian_config {} provided for request", guardian_config)
            # Move the risk name and/or risk definition to chat_template_kwargs
            # to be propagated to tokenizer.apply_chat_template during
            # chat completion
            request.detector_params["chat_template_kwargs"] = {
                "guardian_config": guardian_config
            }

        return request
