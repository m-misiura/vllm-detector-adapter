# Standard
from dataclasses import dataclass
from http import HTTPStatus
from unittest.mock import patch
import asyncio

# Third Party
from vllm.config import MultiModalConfig
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_engine import BaseModelPath
import pytest
import pytest_asyncio

# Local
from vllm_detector_adapter.generative_detectors.llama_guard import LlamaGuard
from vllm_detector_adapter.protocol import (
    ChatDetectionRequest,
    ChatDetectionResponse,
    DetectionChatMessageParam,
)

MODEL_NAME = "meta-llama/Llama-Guard-3-8B"  # Example llama guard model
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    embedding_mode = False
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()


@dataclass
class MockEngine:
    async def get_model_config(self):
        return MockModelConfig()


async def _llama_guard_init():
    """Initialize a llama guard"""
    engine = MockEngine()
    model_config = await engine.get_model_config()

    llama_guard_detection = LlamaGuard(
        task_template=None,
        output_template=None,
        engine_client=engine,
        model_config=model_config,
        base_model_paths=BASE_MODEL_PATHS,
        response_role="assistant",
        chat_template=CHAT_TEMPLATE,
        lora_modules=None,
        prompt_adapters=None,
        request_logger=None,
    )
    return llama_guard_detection


@pytest_asyncio.fixture
async def llama_guard_detection():
    return _llama_guard_init()


@pytest.fixture(scope="module")
def llama_guard_completion_response():
    log_probs_content_random = ChatCompletionLogProbsContent(
        token="\n\n",
        logprob=0.0,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="\n\n", logprob=0.0),
            ChatCompletionLogProb(token='"\n\n', logprob=-29.68),
            ChatCompletionLogProb(token="\n", logprob=-30.57),
            ChatCompletionLogProb(token=")\n\n", logprob=-31.64),
            ChatCompletionLogProb(token="()\n\n", logprob=-32.18),
        ],
    )
    log_probs_content_safe = ChatCompletionLogProbsContent(
        token="safe",
        logprob=-0.0013,
        # 5 logprobs requested for scoring, skipping bytes for conciseness
        top_logprobs=[
            ChatCompletionLogProb(token="safe", logprob=-0.0013),
            ChatCompletionLogProb(token="unsafe", logprob=-6.61),
            ChatCompletionLogProb(token="1", logprob=-16.90),
            ChatCompletionLogProb(token="2", logprob=-17.39),
            ChatCompletionLogProb(token="3", logprob=-17.61),
        ],
    )
    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role="assistant",
            content="safe",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_safe]
        ),
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(
            role="assistant",
            content="safe",
        ),
        logprobs=ChatCompletionLogProbs(
            content=[log_probs_content_random, log_probs_content_safe]
        ),
    )
    yield ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[choice_0, choice_1],
        usage=UsageInfo(prompt_tokens=200, total_tokens=206, completion_tokens=6),
    )


### Tests #####################################################################

# NOTE: currently these functions are basically just the base implementations,
# where safe/unsafe tokens are defined in the llama guard class


def test_calculate_scores(llama_guard_detection, llama_guard_completion_response):
    llama_guard_detection_instance = asyncio.run(llama_guard_detection)
    scores = llama_guard_detection_instance.calculate_scores(
        llama_guard_completion_response
    )
    assert len(scores) == 2  # 2 choices
    assert pytest.approx(scores[0]) == 0.001346767
    assert pytest.approx(scores[1]) == 0.001346767


def test_chat_detection(llama_guard_detection, llama_guard_completion_response):
    llama_guard_detection_instance = asyncio.run(llama_guard_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(
                role="user", content="How do I search for moose?"
            ),
            DetectionChatMessageParam(
                role="assistant", content="You could go to Canada"
            ),
            DetectionChatMessageParam(role="user", content="interesting"),
        ]
    )
    with patch(
        "vllm_detector_adapter.generative_detectors.llama_guard.LlamaGuard.create_chat_completion",
        return_value=llama_guard_completion_response,
    ):
        detection_response = asyncio.run(
            llama_guard_detection_instance.chat(chat_request)
        )
        assert type(detection_response) == ChatDetectionResponse
        detections = detection_response.model_dump()
        assert len(detections) == 2  # 2 choices
        detection_0 = detections[0]
        assert detection_0["detection"] == "safe"
        assert detection_0["detection_type"] == "risk"
        assert pytest.approx(detection_0["score"]) == 0.001346767


def test_chat_detection_with_extra_unallowed_params(llama_guard_detection):
    llama_guard_detection_instance = asyncio.run(llama_guard_detection)
    chat_request = ChatDetectionRequest(
        messages=[
            DetectionChatMessageParam(role="user", content="How do I search for moose?")
        ],
        detector_params={"moo": "unallowed"},  # unallowed param
    )
    detection_response = asyncio.run(llama_guard_detection_instance.chat(chat_request))
    assert type(detection_response) == ErrorResponse
    assert detection_response.code == HTTPStatus.BAD_REQUEST.value
