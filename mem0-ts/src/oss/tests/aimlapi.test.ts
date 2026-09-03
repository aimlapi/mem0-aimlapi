/// <reference types="jest" />
/**
 * aimlapi.com LLM + embedder — unit tests (mocked OpenAI).
 */

import { AimlapiLLM } from "../src/llms/aimlapi";
import { AimlapiEmbedder } from "../src/embeddings/aimlapi";
import { buildAimlapiHeaders } from "../src/utils/aimlapi";

const mockCreate = jest.fn();
const mockEmbeddingsCreate = jest.fn();
const mockOpenAICtor = jest.fn();

jest.mock("openai", () => {
  return jest.fn().mockImplementation((config) => {
    mockOpenAICtor(config);
    return {
      chat: { completions: { create: mockCreate } },
      embeddings: { create: mockEmbeddingsCreate },
    };
  });
});

describe("aimlapi.com providers (unit)", () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...ORIGINAL_ENV };
    delete process.env.AIMLAPI_API_KEY;
    delete process.env.AIMLAPI_API_BASE;
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "hi", role: "assistant" } }],
    });
    mockEmbeddingsCreate.mockResolvedValue({
      data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
    });
  });

  afterAll(() => {
    process.env = ORIGINAL_ENV;
  });

  describe("attribution headers", () => {
    it("uses a partner id the gateway will accept", () => {
      // A malformed partner id is dropped silently and earns nothing, so the
      // shape is asserted here rather than discovered in production.
      const headers = buildAimlapiHeaders("https://api.aimlapi.com/v1")!;
      expect(headers["X-AIMLAPI-Partner-ID"]).toMatch(
        /^part_[A-Za-z0-9]{1,64}$/,
      );
      expect(headers["X-AIMLAPI-Source"]).toMatch(
        /^(web|agent|mcp)\/[a-z0-9-]{1,32}$/,
      );
    });

    it("sends attribution to aimlapi.com and nowhere else", () => {
      new AimlapiLLM({ apiKey: "test-key" });
      expect(mockOpenAICtor).toHaveBeenCalledWith(
        expect.objectContaining({
          defaultHeaders: expect.objectContaining({
            "X-AIMLAPI-Source": "agent/mem0",
            "HTTP-Referer": "https://github.com/mem0ai/mem0",
            "X-Title": "Mem0",
          }),
        }),
      );

      // A user pointing the provider at their own gateway must not have our
      // attribution ride along to a third party.
      new AimlapiLLM({
        apiKey: "test-key",
        baseURL: "https://proxy.example.com/v1",
      });
      expect(mockOpenAICtor).toHaveBeenLastCalledWith(
        expect.not.objectContaining({ defaultHeaders: expect.anything() }),
      );
    });

    it("does not mutate the shared header constant", () => {
      const first = buildAimlapiHeaders("https://api.aimlapi.com/v1", {
        "X-Title": "Other",
      })!;
      const second = buildAimlapiHeaders("https://api.aimlapi.com/v1")!;
      expect(first["X-Title"]).toBe("Other");
      expect(second["X-Title"]).toBe("Mem0");
      expect(first).not.toBe(second);
    });
  });

  describe("AimlapiLLM", () => {
    it("defaults to openai/gpt-5-mini and the aimlapi.com base URL (matching the Python provider)", async () => {
      const llm = new AimlapiLLM({ apiKey: "test-key" });
      const result = await llm.generateResponse([
        { role: "user", content: "hello" },
      ]);

      expect(mockOpenAICtor).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: "test-key",
          baseURL: "https://api.aimlapi.com/v1",
        }),
      );
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({ model: "openai/gpt-5-mini" }),
      );
      expect(result).toBe("hi");
    });

    it("resolves AIMLAPI_API_KEY / AIMLAPI_API_BASE from the environment", () => {
      process.env.AIMLAPI_API_KEY = "env-key";
      process.env.AIMLAPI_API_BASE = "https://gateway.example.com/v1";

      new AimlapiLLM({});

      expect(mockOpenAICtor).toHaveBeenCalledWith(
        expect.objectContaining({
          apiKey: "env-key",
          baseURL: "https://gateway.example.com/v1",
        }),
      );
    });

    it("prefers explicit config over defaults and the environment", async () => {
      process.env.AIMLAPI_API_KEY = "env-key";

      const llm = new AimlapiLLM({
        apiKey: "explicit-key",
        model: "anthropic/claude-sonnet-4.6",
      });
      await llm.generateResponse([{ role: "user", content: "hello" }]);

      expect(mockOpenAICtor).toHaveBeenCalledWith(
        expect.objectContaining({ apiKey: "explicit-key" }),
      );
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({ model: "anthropic/claude-sonnet-4.6" }),
      );
    });

    it("throws when no API key is provided", () => {
      expect(() => new AimlapiLLM({})).toThrow("API key is required");
    });

    it("generateResponse() handles tool calls", async () => {
      mockCreate.mockResolvedValueOnce({
        choices: [
          {
            message: {
              content: "",
              role: "assistant",
              tool_calls: [
                {
                  function: {
                    name: "add_memory",
                    arguments: '{"data": "likes pizza"}',
                  },
                },
              ],
            },
          },
        ],
      });

      const llm = new AimlapiLLM({ apiKey: "test-key" });
      const result = await llm.generateResponse(
        [{ role: "user", content: "remember this" }],
        undefined,
        [{ type: "function", function: { name: "add_memory" } }],
      );

      expect(result).toEqual({
        content: "",
        role: "assistant",
        toolCalls: [
          { name: "add_memory", arguments: '{"data": "likes pizza"}' },
        ],
      });
    });

    it("wraps downstream errors with a provider-specific message", async () => {
      mockCreate.mockRejectedValueOnce(new Error("Connection refused"));
      const llm = new AimlapiLLM({ apiKey: "test-key" });

      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow("aimlapi.com LLM failed: Connection refused");
    });

    it("generateChat() returns the LLMResponse shape", async () => {
      const llm = new AimlapiLLM({ apiKey: "test-key" });
      const result = await llm.generateChat([
        { role: "user", content: "help me" },
      ]);
      expect(result).toEqual({ content: "hi", role: "assistant" });
    });
  });

  describe("AimlapiEmbedder", () => {
    it("defaults to openai/text-embedding-3-small and sends string input", async () => {
      const embedder = new AimlapiEmbedder({ apiKey: "test-key" });
      const result = await embedder.embed("hello world");

      expect(mockOpenAICtor).toHaveBeenCalledWith(
        expect.objectContaining({ baseURL: "https://api.aimlapi.com/v1" }),
      );
      // The API rejects the pre-tokenised integer-array form of `input` with a
      // 400, so the text must reach it as a string.
      expect(mockEmbeddingsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "openai/text-embedding-3-small",
          input: "hello world",
          encoding_format: "float",
        }),
      );
      expect(result).toEqual([0.1, 0.2, 0.3]);
    });

    it("throws when no API key is provided", () => {
      expect(() => new AimlapiEmbedder({})).toThrow("API key is required");
    });

    it("passes dimensions only when embeddingDims is configured", async () => {
      const embedder = new AimlapiEmbedder({
        apiKey: "test-key",
        model: "openai/text-embedding-3-large",
        embeddingDims: 1024,
      });
      await embedder.embed("hi");

      expect(mockEmbeddingsCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "openai/text-embedding-3-large",
          dimensions: 1024,
        }),
      );
    });
  });
});
