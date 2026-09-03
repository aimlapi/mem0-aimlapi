import { OpenAIEmbedder } from "./openai";
import { EmbeddingConfig } from "../types";
import { buildAimlapiHeaders, resolveAimlapiBaseURL } from "../utils/aimlapi";

const DEFAULT_MODEL = "openai/text-embedding-3-small";

/**
 * aimlapi.com embedder — the OpenAI-compatible `/v1/embeddings` route.
 *
 * Mirrors `mem0/embeddings/aimlapi.py` in the Python SDK. Model ids are the
 * gateway's namespaced ids, e.g. `openai/text-embedding-3-small` (1536 dims) or
 * `openai/text-embedding-3-large` (3072 dims).
 */
export class AimlapiEmbedder extends OpenAIEmbedder {
  constructor(config: EmbeddingConfig) {
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
      model: config.model || DEFAULT_MODEL,
      defaultHeaders: buildAimlapiHeaders(baseURL, config.defaultHeaders),
    });
  }
}
