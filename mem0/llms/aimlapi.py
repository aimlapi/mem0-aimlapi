import json
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.aimlapi import AimlapiConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json
from mem0.utils.aimlapi import build_default_headers, drop_none, resolve_base_url


class AimlapiLLM(LLMBase):
    """aimlapi.com — an OpenAI-compatible gateway over 350+ chat models.

    Model ids are the gateway's own namespaced ids, e.g. `openai/gpt-5-mini`,
    `anthropic/claude-sonnet-4.6`, `google/gemini-2.5-flash`. The full catalog is at
    `https://api.aimlapi.com/v1/models`.
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, AimlapiConfig, Dict]] = None):
        # Convert to AimlapiConfig if needed
        if config is None:
            config = AimlapiConfig()
        elif isinstance(config, dict):
            config = AimlapiConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AimlapiConfig):
            # Convert BaseLlmConfig to AimlapiConfig so aimlapi_base_url is available
            config = AimlapiConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                reasoning_effort=getattr(config, "reasoning_effort", None),
                http_client_proxies=config.http_client_proxies,
                is_reasoning_model=getattr(config, "is_reasoning_model", None),
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "openai/gpt-5-mini"

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

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using aimlapi.com.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional provider-specific parameters.

        Returns:
            str or dict: The generated response. A string when tools are not requested;
            a dict ``{"content": ..., "tool_calls": [...]}`` when tools are requested.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # The API answers 400 to an explicit `null` on several fields rather than
        # treating it as unset — `tools: null` is the common one, and it turns the
        # second turn of every tool-clearing agent loop into an error. Omit unset keys.
        response = self.client.chat.completions.create(**drop_none(params))
        return self._parse_response(response, tools)
