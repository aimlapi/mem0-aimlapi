import { OpenAILLM } from "./openai";
import { LLMConfig, Message } from "../types";
import { LLMResponse } from "./base";
import { buildAimlapiHeaders, resolveAimlapiBaseURL } from "../utils/aimlapi";

/**
 * aimlapi.com LLM provider — an OpenAI-compatible gateway over 350+ chat models.
 *
 * The API is OpenAI-compatible, so this reuses {@link OpenAILLM} and overrides the
 * connection defaults — mirroring `mem0/llms/aimlapi.py` in the Python SDK. The API
 * key resolves from `config.apiKey` or `AIMLAPI_API_KEY`, and the base URL from
 * `config.baseURL`, `AIMLAPI_API_BASE`, else `https://api.aimlapi.com/v1`.
 *
 * Model ids are the gateway's namespaced ids, e.g. `openai/gpt-5-mini`,
 * `anthropic/claude-sonnet-4.6`. The catalog is at
 * `https://api.aimlapi.com/v1/models`.
 */
export class AimlapiLLM extends OpenAILLM {
  constructor(config: LLMConfig) {
    const apiKey = config.apiKey || process.env.AIMLAPI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "aimlapi.com API key is required. Set AIMLAPI_API_KEY or pass apiKey in the config.",
      );
    }
    const baseURL = resolveAimlapiBaseURL(
      config.baseURL || config.url,
      process.env.AIMLAPI_API_BASE,
    );
    super({
      ...config,
      apiKey,
      baseURL,
      model: config.model || "openai/gpt-5-mini",
      defaultHeaders: buildAimlapiHeaders(baseURL, config.defaultHeaders),
    });
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    try {
      return await super.generateResponse(messages, responseFormat, tools);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`aimlapi.com LLM failed: ${message}`);
    }
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    try {
      return await super.generateChat(messages);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`aimlapi.com LLM failed: ${message}`);
    }
  }
}
