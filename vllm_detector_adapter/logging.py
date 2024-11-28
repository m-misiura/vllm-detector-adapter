"""Configure logger for vllm_detector_adapter
vllm logger configures app / logger for `vllm`.
Since vllm_detector_adapter is built on top of vllm,
we need to add this app to the logger configuration for the
logs to show up. Otherwise, it will get filtered out.
vllm logger ref: https://github.com/vllm-project/vllm/blob/04de9057ab8099291e66ad876e78693c7c2f2ce5/vllm/logger.py#L81
"""

# Standard
import logging

# Third Party
from vllm.logger import init_logger  # noqa: F401
from vllm.logger import DEFAULT_LOGGING_CONFIG

config = {**DEFAULT_LOGGING_CONFIG}

config["formatters"]["vllm_detector_adapter"] = DEFAULT_LOGGING_CONFIG["formatters"][
    "vllm"
]

handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
handler_config["formatter"] = "vllm_detector_adapter"
config["handlers"]["vllm_detector_adapter"] = handler_config

logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
logger_config["handlers"] = ["vllm_detector_adapter"]
config["loggers"]["vllm_detector_adapter"] = logger_config

logging.config.dictConfig(config)
