/**
 * Shared connection details for the aimlapi.com LLM and embedding providers.
 *
 * Mirrors `mem0/utils/aimlapi.py` in the Python SDK: both providers speak the
 * OpenAI-compatible surface of https://api.aimlapi.com/v1, so base-URL resolution
 * and the attribution headers live here rather than being duplicated.
 */

export const AIMLAPI_DEFAULT_BASE_URL = "https://api.aimlapi.com/v1";

/**
 * Host the attribution headers may be sent to. Users can point the providers at a
 * gateway of their own, and in that case the headers must not ride along to a
 * third party.
 */
export const AIMLAPI_ATTRIBUTION_HOST = "api.aimlapi.com";

/**
 * Identifies Mem0 as the calling application. HTTP-Referer / X-Title are the
 * OpenRouter convention and name the *host* project.
 */
const ATTRIBUTION_HEADERS: Record<string, string> = {
  "HTTP-Referer": "https://github.com/mem0ai/mem0",
  "X-Title": "Mem0",
  "X-AIMLAPI-Partner-ID": "part_mem0",
  "X-AIMLAPI-Source": "agent/mem0",
};

export function resolveAimlapiBaseURL(
  configured?: string,
  envValue?: string,
): string {
  return configured || envValue || AIMLAPI_DEFAULT_BASE_URL;
}

/**
 * Build the per-client `defaultHeaders` for a request to `baseURL`.
 *
 * Returns a new object every call, so the module constant is never mutated and two
 * clients cannot share one header map. Caller-supplied headers win on a key clash.
 * Returns undefined when `baseURL` does not point at aimlapi.com, so attribution
 * never travels to another provider or to a proxy fronting this API.
 */
export function buildAimlapiHeaders(
  baseURL: string,
  extra?: Record<string, string>,
): Record<string, string> | undefined {
  let host: string;
  try {
    host = new URL(baseURL).hostname;
  } catch {
    return extra ? { ...extra } : undefined;
  }
  if (host !== AIMLAPI_ATTRIBUTION_HOST) {
    return extra ? { ...extra } : undefined;
  }
  return { ...ATTRIBUTION_HEADERS, ...(extra || {}) };
}
