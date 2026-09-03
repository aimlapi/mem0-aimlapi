import os
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from mem0.utils.aimlapi import build_default_headers, resolve_base_url


class AimlapiEmbedding(EmbeddingBase):
    """aimlapi.com embeddings — the OpenAI-compatible `/v1/embeddings` route.

    Model ids are the gateway's namespaced ids, e.g. `openai/text-embedding-3-small`
    (1536 dims) or `openai/text-embedding-3-large` (3072 dims).
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "openai/text-embedding-3-small"
        # Only pass `dimensions` to the API when the user set embedding_dims; not every
        # model behind the gateway is matryoshka, and those reject the parameter.
        self._pass_dimensions_to_api = self.config.embedding_dims is not None
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("AIMLAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "aimlapi.com API key is required. Set the AIMLAPI_API_KEY environment variable "
                "or pass api_key in the config."
            )

        base_url = resolve_base_url(self.config.aimlapi_base_url, os.getenv("AIMLAPI_API_BASE"))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=build_default_headers(base_url),
        )

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using aimlapi.com.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        # `input` must be a string or a list of strings. The API rejects the
        # pre-tokenised integer-array form with a 400 naming `input`.
        kwargs = {
            "input": [text],
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts in a single aimlapi.com API call.

        Automatically chunks into batches of 100 to stay within API limits.
        """
        MAX_BATCH = 100
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i : i + MAX_BATCH]
            kwargs = {
                "input": chunk,
                "model": self.config.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"aimlapi.com embed_batch() returned {len(all_embeddings)} embeddings for {len(texts)} texts"
                f" using model '{self.config.model}'"
            )
        return all_embeddings
