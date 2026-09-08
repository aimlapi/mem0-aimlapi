import os
from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.aimlapi import AimlapiEmbedding
from mem0.utils.factory import EmbedderFactory


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
    with patch("mem0.embeddings.aimlapi.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_openai, mock_client


def test_default_model_and_base_url(mock_aimlapi_client):
    mock_openai, _ = mock_aimlapi_client
    embedder = AimlapiEmbedding(BaseEmbedderConfig(api_key="k"))
    assert embedder.config.model == "openai/text-embedding-3-small"
    assert embedder.config.embedding_dims == 1536
    assert mock_openai.call_args.kwargs["base_url"] == "https://api.aimlapi.com/v1"


def test_base_url_override(mock_aimlapi_client):
    mock_openai, _ = mock_aimlapi_client
    os.environ["AIMLAPI_API_BASE"] = "https://gateway.example.com/v1"
    AimlapiEmbedding(BaseEmbedderConfig(api_key="k"))
    assert mock_openai.call_args.kwargs["base_url"] == "https://gateway.example.com/v1"

    AimlapiEmbedding(BaseEmbedderConfig(api_key="k", aimlapi_base_url="https://config.example.com/v1"))
    assert mock_openai.call_args.kwargs["base_url"] == "https://config.example.com/v1"


def test_requires_api_key():
    with pytest.raises(ValueError, match="API key is required"):
        AimlapiEmbedding(BaseEmbedderConfig())


def test_attribution_headers(mock_aimlapi_client):
    mock_openai, _ = mock_aimlapi_client
    AimlapiEmbedding(BaseEmbedderConfig(api_key="k"))
    headers = mock_openai.call_args.kwargs["default_headers"]
    assert headers["X-AIMLAPI-Partner-ID"].startswith("part_")
    assert headers["X-AIMLAPI-Source"] == "agent/mem0"

    # Not on a request to somebody else's gateway.
    AimlapiEmbedding(BaseEmbedderConfig(api_key="k", aimlapi_base_url="https://proxy.example.com/v1"))
    assert mock_openai.call_args.kwargs["default_headers"] is None


def test_embed_sends_string_input(mock_aimlapi_client):
    """The API rejects the pre-tokenised integer-array form of ``input`` with a 400,
    so the text must reach it as a string.
    """
    _, mock_client = mock_aimlapi_client
    embedder = AimlapiEmbedding(BaseEmbedderConfig(api_key="k"))
    mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])

    result = embedder.embed("Hello\nworld")

    mock_client.embeddings.create.assert_called_once_with(
        input=["Hello world"],
        model="openai/text-embedding-3-small",
        encoding_format="float",
    )
    assert all(isinstance(item, str) for item in mock_client.embeddings.create.call_args.kwargs["input"])
    assert result == [0.1, 0.2, 0.3]


def test_embed_passes_dimensions_only_when_configured(mock_aimlapi_client):
    _, mock_client = mock_aimlapi_client
    embedder = AimlapiEmbedding(
        BaseEmbedderConfig(api_key="k", model="openai/text-embedding-3-large", embedding_dims=1024)
    )
    mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=[0.4])])

    embedder.embed("hi")

    mock_client.embeddings.create.assert_called_once_with(
        input=["hi"],
        model="openai/text-embedding-3-large",
        encoding_format="float",
        dimensions=1024,
    )


def test_embed_batch(mock_aimlapi_client):
    _, mock_client = mock_aimlapi_client
    embedder = AimlapiEmbedding(BaseEmbedderConfig(api_key="k"))
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(index=1, embedding=[0.3]), Mock(index=0, embedding=[0.1])]
    )

    result = embedder.embed_batch(["a", "b"])

    assert result == [[0.1], [0.3]]
    assert mock_client.embeddings.create.call_args.kwargs["input"] == ["a", "b"]


def test_registered_in_factory(mock_aimlapi_client):
    embedder = EmbedderFactory.create("aimlapi", {"api_key": "k"}, None)
    assert isinstance(embedder, AimlapiEmbedding)
