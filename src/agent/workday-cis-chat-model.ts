import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { AIMessageChunk, isAIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import {
  ChatGenerationChunk,
  type ChatResult,
} from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import {
  buildCisRequest,
  cisChunkToGeneration,
  converseChunkToGeneration,
  parseSseEvent,
  resolveCisDialect,
  splitSseEvents,
  toGeminiFunctionDeclarations,
  type CisDialect,
  type GeminiFunctionDeclaration,
} from "./workday-cis-protocol.js";

export type ChatWorkdayCisCallOptions = BaseChatModelCallOptions & {
  cisFunctionDeclarations?: GeminiFunctionDeclaration[];
};

export type ChatWorkdayCisParams = BaseChatModelParams & {
  model: string;
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
  query?: Record<string, string>;
  targetProvider: string;
  taskType: string;
  predictionType: string | undefined;
  generationConfig: Record<string, unknown>;
  fetchImpl?: typeof fetch;
};

export class ChatWorkdayCis extends BaseChatModel<ChatWorkdayCisCallOptions> {
  model: string;
  baseURL: string;
  apiKey?: string;
  headers: Record<string, string>;
  query: Record<string, string>;
  targetProvider: string;
  taskType: string;
  predictionType: string | undefined;
  generationConfig: Record<string, unknown>;
  private readonly dialect: CisDialect;
  private readonly fetchImpl: typeof fetch;

  constructor(params: ChatWorkdayCisParams) {
    super(params);
    this.model = params.model;
    this.baseURL = params.baseURL;
    this.apiKey = params.apiKey;
    this.headers = params.headers ?? {};
    this.query = params.query ?? {};
    this.targetProvider = params.targetProvider;
    this.taskType = params.taskType;
    this.predictionType = params.predictionType;
    this.generationConfig = params.generationConfig;
    this.dialect = resolveCisDialect(params.taskType);
    this.fetchImpl = params.fetchImpl ?? ((...args) => fetch(...args));
  }

  _llmType(): string {
    return "workday-cis";
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatWorkdayCisCallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatWorkdayCisCallOptions> {
    return this.withConfig({
      cisFunctionDeclarations: toGeminiFunctionDeclarations(
        tools as Parameters<typeof toGeminiFunctionDeclarations>[0],
      ),
      ...kwargs,
    });
  }

  private buildUrl(): string {
    const base = this.baseURL.replace(/\/+$/u, "");
    const queryString = new URLSearchParams(this.query).toString();

    return queryString
      ? `${base}/predictions/stream?${queryString}`
      : `${base}/predictions/stream`;
  }

  private buildHeaders(): Record<string, string> {
    return {
      "content-type": "application/json",
      ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
      ...this.headers,
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const request = buildCisRequest(
      {
        targetProvider: this.targetProvider,
        model: this.model,
        taskType: this.taskType,
        predictionType: this.predictionType,
        generationConfig: this.generationConfig,
      },
      messages,
      options.cisFunctionDeclarations ?? [],
    );

    const response = await this.fetchImpl(this.buildUrl(), {
      method: "POST",
      headers: this.buildHeaders(),
      body: JSON.stringify(request),
      signal: options.signal,
    });

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");

      throw new Error(
        `Workday CIS request failed: ${response.status} ${response.statusText} ${errorBody}`.trim(),
      );
    }

    if (!response.body) {
      throw new Error("Workday CIS response has no body.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let doneReceived = false;
    let yieldedAny = false;

    const handleSseEvent = (rawEvent: string): ChatGenerationChunk | "done" | null => {
      const { event, data } = parseSseEvent(rawEvent);

      if (event === "error") {
        throw new Error(`Workday CIS streaming error: ${data}`);
      }

      if (data === "[DONE]") {
        return "done";
      }

      if (data.length === 0) {
        return null;
      }

      let payload: unknown;

      try {
        payload = JSON.parse(data);
      } catch {
        return null;
      }

      return this.dialect === "converse"
        ? converseChunkToGeneration(payload)
        : cisChunkToGeneration(payload);
    };

    const processRawEvents = async function* (
      rawEvents: string[],
    ): AsyncGenerator<ChatGenerationChunk> {
      for (const rawEvent of rawEvents) {
        const result = handleSseEvent(rawEvent);

        if (result === "done") {
          doneReceived = true;

          continue;
        }

        if (result) {
          yieldedAny = true;

          if (result.text.length > 0) {
            await runManager?.handleLLMNewToken(result.text);
          }

          yield result;
        }
      }
    };

    try {
      while (true) {
        const readResult = await reader.read();

        if (readResult.done) {
          break;
        }

        buffer += decoder
          .decode(readResult.value as Uint8Array, { stream: true })
          .replace(/\r/gu, "");
        const split = splitSseEvents(buffer);
        buffer = split.rest;

        yield* processRawEvents(split.events);

        if (doneReceived) {
          break;
        }
      }

      buffer += decoder.decode().replace(/\r/gu, "");

      if (buffer.length > 0) {
        yield* processRawEvents([buffer]);
      }

      if (!doneReceived && !yieldedAny) {
        throw new Error(
          "Workday CIS stream ended without any data ([DONE] not received).",
        );
      }
    } finally {
      try {
        await reader.cancel();
      } catch {
        // stream may already be closed/errored
      }

      reader.releaseLock();
    }
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    let aggregate: ChatGenerationChunk | undefined;

    for await (const chunk of this._streamResponseChunks(
      messages,
      options,
      runManager,
    )) {
      aggregate = aggregate ? aggregate.concat(chunk) : chunk;
    }

    const message = aggregate?.message ?? new AIMessageChunk({ content: "" });
    const text = aggregate?.text ?? "";
    const usageMetadata = isAIMessageChunk(message)
      ? message.usage_metadata
      : undefined;
    const llmOutput = usageMetadata
      ? {
          tokenUsage: {
            promptTokens: usageMetadata.input_tokens,
            completionTokens: usageMetadata.output_tokens,
            totalTokens: usageMetadata.total_tokens,
          },
        }
      : undefined;

    return { generations: [{ text, message }], llmOutput };
  }
}
