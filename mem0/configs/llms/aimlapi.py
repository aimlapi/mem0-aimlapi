from typing import Optional

from mem0.configs.llms.base import BaseLlmConfig


class AimlapiConfig(BaseLlmConfig):
    """
    Configuration class for aimlapi.com-specific parameters.
    Inherits from BaseLlmConfig and adds aimlapi.com-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        reasoning_effort: Optional[str] = None,
        http_client_proxies: Optional[dict] = None,
        is_reasoning_model: Optional[bool] = None,
        # aimlapi.com-specific parameters
        aimlapi_base_url: Optional[str] = None,
    ):
        """
        Initialize aimlapi.com configuration.

        Args:
            model: aimlapi.com model id to use (e.g. "openai/gpt-5-mini"), defaults to None
            temperature: Controls randomness, defaults to 0.1
            api_key: aimlapi.com API key, defaults to None
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            reasoning_effort: Effort level for reasoning models, defaults to None
            http_client_proxies: HTTP client proxy settings, defaults to None
            is_reasoning_model: Explicit reasoning-model override, defaults to None
            aimlapi_base_url: aimlapi.com API base URL, defaults to None
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            reasoning_effort=reasoning_effort,
            http_client_proxies=http_client_proxies,
            is_reasoning_model=is_reasoning_model,
        )

        # aimlapi.com-specific parameters
        self.aimlapi_base_url = aimlapi_base_url
