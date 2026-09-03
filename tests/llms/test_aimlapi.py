import os
import re
from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.aimlapi import AimlapiConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.aimlapi import AimlapiLLM
from mem0.utils.aimlapi import build_default_headers
from mem0.utils.factory import LlmFactory


@pytest.fixture(autouse=True)
def clear_aimlapi_env():
    saved = {k: os.environ.pop(k, None) for k in ("AIMLAPI_API_KEY", "AIMLAPI_API_BASE")}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture
def mock_aimlapi_client():
    with patch("mem0.llms.aimlapi.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_openai, mock_client


def test_aimlapi_base_url_resolution():
    # case1: default
    llm = AimlapiLLM(AimlapiConfig(api_key="api_key"))
    assert str(llm.client.base_url) == "https://api.aimlapi.com/v1/"

    # case2: AIMLAPI_API_BASE env var
    os.environ["AIMLAPI_API_BASE"] = "https://gateway.example.com/v1"
    llm = AimlapiLLM(AimlapiConfig(api_key="api_key"))
    assert str(llm.client.base_url) == "https://gateway.example.com/v1/"

    # case3: config.aimlapi_base_url wins over env
    llm = AimlapiLLM(AimlapiConfig(api_key="api_key", aimlapi_base_url="https://config.example.com/v1"))
    assert str(llm.client.base_url) == "https://config.example.com/v1/"


def test_aimlapi_default_model():
    llm = AimlapiLLM(AimlapiConfig(api_key="k"))
    assert llm.config.model == "openai/gpt-5-mini"


def test_aimlapi_reads_api_key_from_env():
    os.environ["AIMLAPI_API_KEY"] = "env_key"
    llm = AimlapiLLM(AimlapiConfig())
    assert llm.client.api_key == "env_key"


def test_aimlapi_requires_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        AimlapiLLM(AimlapiConfig())


def test_aimlapi_accepts_base_llm_config():
    # The factory may hand over a plain BaseLlmConfig; aimlapi_base_url must still exist.
    llm = AimlapiLLM(BaseLlmConfig(model="openai/gpt-5-mini", api_key="k"))
    assert isinstance(llm.config, AimlapiConfig)
    assert llm.config.aimlapi_base_url is None


def test_aimlapi_registered_in_factory():
    llm = LlmFactory.create("aimlapi", {"model": "openai/gpt-5-mini", "api_key": "k"})
    assert isinstance(llm, AimlapiLLM)


# --- attribution headers -------------------------------------------------


def test_partner_id_matches_gateway_pattern():
    # A malformed partner id is dropped silently by the gateway and earns nothing,
    # so the shape is asserted here rather than discovered in production.
    headers = build_default_headers("https://api.aimlapi.com/v1")
    assert re.fullmatch(r"^part_[A-Za-z0-9]{1,64}$", headers["X-AIMLAPI-Partner-ID"])
    assert re.fullmatch(r"^(web|agent|mcp)/[a-z0-9-]{1,32}$", headers["X-AIMLAPI-Source"])


def test_attribution_headers_sent_to_aimlapi(mock_aimlapi_client):
    mock_openai, _ = mock_aimlapi_client
    AimlapiLLM(AimlapiConfig(api_key="k"))
    headers = mock_openai.call_args.kwargs["default_headers"]
    assert headers["X-AIMLAPI-Source"] == "agent/mem0"
    # HTTP-Referer / X-Title identify the calling application, which is Mem0.
    assert headers["HTTP-Referer"] == "https://github.com/mem0ai/mem0"
    assert headers["X-Title"] == "Mem0"


def test_attribution_headers_not_sent_to_other_hosts(mock_aimlapi_client):
    # A user pointing the provider at their own gateway must not have our
    # attribution ride along to a third party.
    mock_openai, _ = mock_aimlapi_client
    AimlapiLLM(AimlapiConfig(api_key="k", aimlapi_base_url="https://proxy.example.com/v1"))
    assert mock_openai.call_args.kwargs["default_headers"] is None


def test_build_default_headers_does_not_mutate_shared_constant():
    first = build_default_headers("https://api.aimlapi.com/v1", {"X-Title": "Other"})
    second = build_default_headers("https://api.aimlapi.com/v1")
    assert first["X-Title"] == "Other"  # caller wins on a clash
    assert second["X-Title"] == "Mem0"  # and the constant is untouched
    assert first is not second


# --- request payload -----------------------------------------------------


def test_generate_response_without_tools(mock_aimlapi_client):
    _, mock_client = mock_aimlapi_client
    config = AimlapiConfig(model="openai/gpt-4o-mini", temperature=0.7, max_tokens=100, top_p=1.0, api_key="k")
    llm = AimlapiLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well!"))]
    mock_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages)

    mock_client.chat.completions.create.assert_called_once_with(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
    )
    assert response == "I'm doing well!"


def test_generate_response_with_tools(mock_aimlapi_client):
    _, mock_client = mock_aimlapi_client
    llm = AimlapiLLM(AimlapiConfig(model="openai/gpt-4o-mini", api_key="k"))
    messages = [{"role": "user", "content": "Add a new memory: Today is a sunny day."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response = Mock()
    mock_message = Mock(content="I've added the memory.")
    mock_tool_call = Mock()
    mock_tool_call.function.name = "add_memory"
    mock_tool_call.function.arguments = '{"data": "Today is a sunny day."}'
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert response["content"] == "I've added the memory."
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}


def test_unset_parameters_are_omitted_not_nulled(mock_aimlapi_client):
    """The API answers 400 to an explicit ``null`` on tools, temperature, top_p and
    friends instead of treating it as unset. Every key on the wire must have a value.
    """
    _, mock_client = mock_aimlapi_client
    config = AimlapiConfig(model="openai/gpt-5-mini", api_key="k")
    config.temperature = None
    config.top_p = None
    config.max_tokens = None
    llm = AimlapiLLM(config)

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="ok"))]
    mock_client.chat.completions.create.return_value = mock_response

    llm.generate_response(
        [{"role": "user", "content": "hi"}],
        response_format=None,
        tools=None,
    )

    sent = mock_client.chat.completions.create.call_args.kwargs
    assert None not in sent.values()
    assert "tools" not in sent
    assert "temperature" not in sent
