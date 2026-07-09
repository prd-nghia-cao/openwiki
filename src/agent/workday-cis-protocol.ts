import { randomUUID } from "node:crypto";
import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
  type BaseMessage,
  type MessageContent,
} from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import { toJsonSchema } from "@langchain/core/utils/json_schema";

export type GeminiPart =
  | { text: string }
  | {
      functionCall: { name: string; args: Record<string, unknown> };
      thoughtSignature?: string;
    }
  | { functionResponse: { name: string; response: Record<string, unknown> } };

/**
 * Key under an `AIMessage.additional_kwargs` where we stash Gemini
 * `thoughtSignature` values (keyed by tool-call id). Gemini 3 "thinking" models
 * require the signature returned with each `functionCall` to be echoed back on
 * the next turn, or Vertex rejects the request ("Upstream streaming error").
 */
export const CIS_THOUGHT_SIGNATURES_KEY = "__cis_thought_signatures";

export type GeminiContent = {
  role: "user" | "model";
  parts: GeminiPart[];
};

export type GeminiSystemInstruction = { parts: { text: string }[] };

export type GeminiFunctionDeclaration = {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
};

export type CisRequestConfig = {
  targetProvider: string;
  model: string;
  taskType: string;
  predictionType: string | undefined;
  generationConfig: Record<string, unknown>;
};

export type GeminiInput = {
  contents: GeminiContent[];
  systemInstruction?: GeminiSystemInstruction;
  tools?: { functionDeclarations: GeminiFunctionDeclaration[] }[];
  generationConfig: Record<string, unknown>;
};

export type ConverseContentBlock =
  | { text: string }
  | {
      toolUse: { toolUseId: string; name: string; input: Record<string, unknown> };
    }
  | { toolResult: { toolUseId: string; content: { text: string }[] } };

export type ConverseMessage = {
  role: "user" | "assistant";
  content: ConverseContentBlock[];
};

export type ConverseToolSpec = {
  toolSpec: {
    name: string;
    description?: string;
    inputSchema: { json: Record<string, unknown> };
  };
};

export type ConverseInput = {
  messages: ConverseMessage[];
  system?: { text: string }[];
  toolConfig?: { tools: ConverseToolSpec[] };
  inferenceConfig: Record<string, unknown>;
};

export type CisRequest = {
  target: { provider: string; model: string };
  task: {
    type: string;
    prediction_type?: string;
    input: GeminiInput | ConverseInput;
  };
};

export type CisDialect = "converse" | "gemini";

/**
 * Chooses the request/response shape for a CIS task type. AWS Bedrock–backed
 * tasks (e.g. `aws-converse-v1`) use the Bedrock Converse shape
 * (`input.messages`); everything else defaults to the Gemini/GCP multimodal
 * shape (`input.contents`).
 */
export function resolveCisDialect(taskType: string): CisDialect {
  const normalized = taskType.toLowerCase();

  if (normalized.startsWith("aws") || normalized.includes("converse")) {
    return "converse";
  }

  return "gemini";
}

type ToolLike = {
  name?: string;
  description?: string;
  schema?: unknown;
  function?: { name?: string; description?: string; parameters?: unknown };
};

const GEMINI_UNSUPPORTED_SCHEMA_KEYS = new Set([
  "$schema",
  "$ref",
  "$defs",
  "definitions",
  "additionalProperties",
]);

export function extractText(content: MessageContent): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .map((block) =>
      typeof block === "object" &&
      block !== null &&
      "type" in block &&
      block.type === "text" &&
      "text" in block &&
      typeof block.text === "string"
        ? block.text
        : "",
    )
    .join("");
}

function toResponseObject(content: MessageContent): Record<string, unknown> {
  return { result: extractText(content) };
}

function getThoughtSignatures(message: BaseMessage): Record<string, string> {
  const stored = message.additional_kwargs?.[CIS_THOUGHT_SIGNATURES_KEY];

  if (typeof stored !== "object" || stored === null) {
    return {};
  }

  const signatures: Record<string, string> = {};

  for (const [key, value] of Object.entries(stored)) {
    if (typeof value === "string") {
      signatures[key] = value;
    }
  }

  return signatures;
}

export function messagesToGeminiContents(messages: BaseMessage[]): {
  contents: GeminiContent[];
  systemInstruction?: GeminiSystemInstruction;
} {
  const toolNameById = new Map<string, string>();

  for (const message of messages) {
    if (AIMessage.isInstance(message)) {
      for (const toolCall of message.tool_calls ?? []) {
        if (toolCall.id) {
          toolNameById.set(toolCall.id, toolCall.name);
        }
      }
    }
  }

  const contents: GeminiContent[] = [];
  const systemTexts: string[] = [];

  for (const message of messages) {
    const type = message.getType();

    if (type === "system") {
      const text = extractText(message.content);

      if (text) {
        systemTexts.push(text);
      }

      continue;
    }

    if (type === "human") {
      contents.push({
        role: "user",
        parts: [{ text: extractText(message.content) }],
      });

      continue;
    }

    if (AIMessage.isInstance(message)) {
      const parts: GeminiPart[] = [];
      const text = extractText(message.content);

      if (text) {
        parts.push({ text });
      }

      const signatures = getThoughtSignatures(message);

      for (const toolCall of message.tool_calls ?? []) {
        const part: Extract<GeminiPart, { functionCall: unknown }> = {
          functionCall: { name: toolCall.name, args: toolCall.args ?? {} },
        };
        const signature = toolCall.id ? signatures[toolCall.id] : undefined;

        if (signature) {
          part.thoughtSignature = signature;
        }

        parts.push(part);
      }

      if (parts.length === 0) {
        parts.push({ text: "" });
      }

      contents.push({ role: "model", parts });

      continue;
    }

    if (ToolMessage.isInstance(message) || type === "function") {
      const toolCallId = ToolMessage.isInstance(message)
        ? message.tool_call_id
        : undefined;
      const name =
        message.name ??
        (toolCallId ? toolNameById.get(toolCallId) : undefined) ??
        "tool";
      const responsePart: GeminiPart = {
        functionResponse: { name, response: toResponseObject(message.content) },
      };

      // Parallel tool calls in one model turn produce several ToolMessages in a
      // row. Gemini requires strict model/user alternation, so their responses
      // must be merged into a single user content with multiple
      // functionResponse parts rather than separate consecutive user turns.
      const last = contents[contents.length - 1];

      if (
        last?.role === "user" &&
        last.parts.every((part) => "functionResponse" in part)
      ) {
        last.parts.push(responsePart);
      } else {
        contents.push({ role: "user", parts: [responsePart] });
      }
    }
  }

  const systemInstruction =
    systemTexts.length > 0
      ? { parts: [{ text: systemTexts.join("\n\n") }] }
      : undefined;

  return systemInstruction ? { contents, systemInstruction } : { contents };
}

function sanitizeGeminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(sanitizeGeminiSchema);
  }

  if (typeof schema !== "object" || schema === null) {
    return schema;
  }

  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(schema)) {
    if (GEMINI_UNSUPPORTED_SCHEMA_KEYS.has(key)) {
      continue;
    }

    result[key] = sanitizeGeminiSchema(value);
  }

  return result;
}

function toGeminiFunctionDeclaration(
  tool: ToolLike,
): GeminiFunctionDeclaration | null {
  const name = tool.name ?? tool.function?.name;

  if (!name) {
    return null;
  }

  const description = tool.description ?? tool.function?.description;
  const rawSchema = tool.schema ?? tool.function?.parameters;
  const declaration: GeminiFunctionDeclaration = { name };

  if (description !== undefined) {
    declaration.description = description;
  }

  if (rawSchema !== undefined) {
    declaration.parameters = sanitizeGeminiSchema(
      toJsonSchema(rawSchema as Parameters<typeof toJsonSchema>[0]),
    ) as Record<string, unknown>;
  }

  return declaration;
}

export function toGeminiFunctionDeclarations(
  tools: ToolLike[],
): GeminiFunctionDeclaration[] {
  return tools
    .map(toGeminiFunctionDeclaration)
    .filter((declaration): declaration is GeminiFunctionDeclaration =>
      declaration !== null,
    );
}

export function messagesToConverse(messages: BaseMessage[]): {
  messages: ConverseMessage[];
  system?: { text: string }[];
} {
  const toolNameById = new Map<string, string>();

  for (const message of messages) {
    if (AIMessage.isInstance(message)) {
      for (const toolCall of message.tool_calls ?? []) {
        if (toolCall.id) {
          toolNameById.set(toolCall.id, toolCall.name);
        }
      }
    }
  }

  const converseMessages: ConverseMessage[] = [];
  const systemTexts: string[] = [];

  for (const message of messages) {
    const type = message.getType();

    if (type === "system") {
      const text = extractText(message.content);

      if (text) {
        systemTexts.push(text);
      }

      continue;
    }

    if (type === "human") {
      converseMessages.push({
        role: "user",
        content: [{ text: extractText(message.content) }],
      });

      continue;
    }

    if (AIMessage.isInstance(message)) {
      const content: ConverseContentBlock[] = [];
      const text = extractText(message.content);

      if (text) {
        content.push({ text });
      }

      for (const toolCall of message.tool_calls ?? []) {
        content.push({
          toolUse: {
            toolUseId: toolCall.id ?? randomUUID(),
            name: toolCall.name,
            input: toolCall.args ?? {},
          },
        });
      }

      if (content.length === 0) {
        content.push({ text: "" });
      }

      converseMessages.push({ role: "assistant", content });

      continue;
    }

    if (ToolMessage.isInstance(message) || type === "function") {
      const toolCallId = ToolMessage.isInstance(message)
        ? message.tool_call_id
        : undefined;
      const resultBlock: ConverseContentBlock = {
        toolResult: {
          toolUseId: toolCallId ?? randomUUID(),
          content: [{ text: extractText(message.content) }],
        },
      };

      // Parallel tool calls produce consecutive ToolMessages. Bedrock requires
      // all toolResults for a turn to share a single user message, so merge
      // them rather than emitting consecutive user messages.
      const last = converseMessages[converseMessages.length - 1];

      if (
        last?.role === "user" &&
        last.content.every((block) => "toolResult" in block)
      ) {
        last.content.push(resultBlock);
      } else {
        converseMessages.push({ role: "user", content: [resultBlock] });
      }
    }
  }

  const system =
    systemTexts.length > 0 ? [{ text: systemTexts.join("\n\n") }] : undefined;

  return system ? { messages: converseMessages, system } : { messages: converseMessages };
}

export function toConverseToolConfig(
  functionDeclarations: GeminiFunctionDeclaration[],
): { tools: ConverseToolSpec[] } | undefined {
  if (functionDeclarations.length === 0) {
    return undefined;
  }

  return {
    tools: functionDeclarations.map((declaration) => {
      const toolSpec: ConverseToolSpec["toolSpec"] = {
        name: declaration.name,
        inputSchema: {
          json:
            declaration.parameters ??
            ({ type: "object", properties: {} }),
        },
      };

      if (declaration.description) {
        toolSpec.description = declaration.description;
      }

      return { toolSpec };
    }),
  };
}

/**
 * Translates a Gemini-style generationConfig into a Bedrock Converse
 * inferenceConfig. `maxOutputTokens` maps to `maxTokens`; `topK` has no
 * Converse inferenceConfig equivalent and is dropped.
 */
export function toConverseInferenceConfig(
  generationConfig: Record<string, unknown>,
): Record<string, unknown> {
  const inferenceConfig: Record<string, unknown> = {};

  if (generationConfig.maxOutputTokens !== undefined) {
    inferenceConfig.maxTokens = generationConfig.maxOutputTokens;
  }

  if (generationConfig.maxTokens !== undefined) {
    inferenceConfig.maxTokens = generationConfig.maxTokens;
  }

  if (generationConfig.temperature !== undefined) {
    inferenceConfig.temperature = generationConfig.temperature;
  }

  if (generationConfig.topP !== undefined) {
    inferenceConfig.topP = generationConfig.topP;
  }

  if (generationConfig.stopSequences !== undefined) {
    inferenceConfig.stopSequences = generationConfig.stopSequences;
  }

  return inferenceConfig;
}

function buildGeminiInput(
  config: CisRequestConfig,
  messages: BaseMessage[],
  functionDeclarations: GeminiFunctionDeclaration[],
): GeminiInput {
  const { contents, systemInstruction } = messagesToGeminiContents(messages);

  const input: GeminiInput = {
    contents,
    generationConfig: config.generationConfig,
  };

  if (systemInstruction) {
    input.systemInstruction = systemInstruction;
  }

  if (functionDeclarations.length > 0) {
    input.tools = [{ functionDeclarations }];
  }

  return input;
}

function buildConverseInput(
  config: CisRequestConfig,
  messages: BaseMessage[],
  functionDeclarations: GeminiFunctionDeclaration[],
): ConverseInput {
  const { messages: converseMessages, system } = messagesToConverse(messages);

  const input: ConverseInput = {
    messages: converseMessages,
    inferenceConfig: toConverseInferenceConfig(config.generationConfig),
  };

  if (system) {
    input.system = system;
  }

  const toolConfig = toConverseToolConfig(functionDeclarations);

  if (toolConfig) {
    input.toolConfig = toolConfig;
  }

  return input;
}

export function buildCisRequest(
  config: CisRequestConfig,
  messages: BaseMessage[],
  functionDeclarations: GeminiFunctionDeclaration[],
): CisRequest {
  const input =
    resolveCisDialect(config.taskType) === "converse"
      ? buildConverseInput(config, messages, functionDeclarations)
      : buildGeminiInput(config, messages, functionDeclarations);

  const task: CisRequest["task"] = { type: config.taskType, input };

  if (config.predictionType) {
    task.prediction_type = config.predictionType;
  }

  return {
    target: { provider: config.targetProvider, model: config.model },
    task,
  };
}

export function splitSseEvents(buffer: string): {
  events: string[];
  rest: string;
} {
  const events: string[] = [];
  let rest = buffer;
  let index = rest.indexOf("\n\n");

  while (index !== -1) {
    events.push(rest.slice(0, index));
    rest = rest.slice(index + 2);
    index = rest.indexOf("\n\n");
  }

  return { events, rest };
}

export function parseSseEvent(raw: string): { event: string; data: string } {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).replace(/^ /u, ""));
    }
  }

  return { event, data: dataLines.join("\n") };
}

type GeminiUsageMetadata = {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
};

function mapUsage(usage: unknown):
  | { input_tokens: number; output_tokens: number; total_tokens: number }
  | undefined {
  if (typeof usage !== "object" || usage === null) {
    return undefined;
  }

  const meta = usage as GeminiUsageMetadata;

  return {
    input_tokens: meta.promptTokenCount ?? 0,
    output_tokens: meta.candidatesTokenCount ?? 0,
    total_tokens: meta.totalTokenCount ?? 0,
  };
}

export function cisChunkToGeneration(payload: unknown): ChatGenerationChunk | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const output = (payload as { output?: unknown }).output;

  if (typeof output !== "object" || output === null) {
    return null;
  }

  const candidates =
    (output as { candidates?: unknown[] }).candidates ?? [];

  let text = "";
  const toolCallChunks: {
    type: "tool_call_chunk";
    name: string;
    args: string;
    id: string;
    index: number;
  }[] = [];
  let index = 0;
  let finishReason: string | undefined;
  const thoughtSignatures: Record<string, string> = {};

  for (const candidate of candidates) {
    if (typeof candidate !== "object" || candidate === null) {
      continue;
    }

    const candidateFinish = (candidate as { finishReason?: string }).finishReason;

    if (candidateFinish) {
      finishReason = candidateFinish;
    }

    const parts =
      (candidate as { content?: { parts?: unknown[] } }).content?.parts ?? [];

    for (const part of parts) {
      if (typeof part !== "object" || part === null) {
        continue;
      }

      const partText = (part as { text?: unknown }).text;

      if (typeof partText === "string" && partText.length > 0) {
        text += partText;

        continue;
      }

      const functionCall = (part as {
        functionCall?: { name?: string; args?: Record<string, unknown> };
      }).functionCall;

      if (functionCall?.name) {
        const id = randomUUID();
        const signature = (part as { thoughtSignature?: unknown })
          .thoughtSignature;

        toolCallChunks.push({
          type: "tool_call_chunk",
          name: functionCall.name,
          args: JSON.stringify(functionCall.args ?? {}),
          id,
          index: index++,
        });

        if (typeof signature === "string") {
          thoughtSignatures[id] = signature;
        }
      }
    }
  }

  const usageMetadata = mapUsage(
    (output as { usageMetadata?: unknown }).usageMetadata,
  );

  if (
    text.length === 0 &&
    toolCallChunks.length === 0 &&
    !usageMetadata &&
    !finishReason
  ) {
    return null;
  }

  const message = new AIMessageChunk({
    content: text,
    tool_call_chunks: toolCallChunks,
    ...(usageMetadata ? { usage_metadata: usageMetadata } : {}),
    ...(Object.keys(thoughtSignatures).length > 0
      ? { additional_kwargs: { [CIS_THOUGHT_SIGNATURES_KEY]: thoughtSignatures } }
      : {}),
    response_metadata: finishReason ? { finishReason } : {},
  });

  return new ChatGenerationChunk({ message, text });
}

type ConverseUsageMetadata = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
};

function mapConverseUsage(usage: unknown):
  | { input_tokens: number; output_tokens: number; total_tokens: number }
  | undefined {
  if (typeof usage !== "object" || usage === null) {
    return undefined;
  }

  const meta = usage as ConverseUsageMetadata;

  if (
    meta.inputTokens === undefined &&
    meta.outputTokens === undefined &&
    meta.totalTokens === undefined
  ) {
    return undefined;
  }

  const inputTokens = meta.inputTokens ?? 0;
  const outputTokens = meta.outputTokens ?? 0;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: meta.totalTokens ?? inputTokens + outputTokens,
  };
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

type ConverseToolCallChunk = {
  type: "tool_call_chunk";
  name?: string;
  args?: string;
  id?: string;
  index: number;
};

/**
 * Maps a CIS `aws-converse-v1` streaming chunk (Bedrock ConverseStream event)
 * to a LangChain generation chunk. Handles both streaming events
 * (`contentBlockStart`/`contentBlockDelta`/`messageStop`/`metadata`) and a
 * full non-streaming Converse response (`output.message.content[]`). Because
 * the exact CIS envelope may nest the event under `output`, both the payload
 * root and `payload.output` are probed.
 */
export function converseChunkToGeneration(
  payload: unknown,
): ChatGenerationChunk | null {
  const root = asObject(payload);

  if (!root) {
    return null;
  }

  const containers = [root, asObject(root.output)].filter(
    (value): value is Record<string, unknown> => value !== undefined,
  );

  let text = "";
  const toolCallChunks: ConverseToolCallChunk[] = [];
  let fallbackIndex = 0;
  let finishReason: string | undefined;
  let usageMetadata: ReturnType<typeof mapConverseUsage>;

  const readIndex = (container: Record<string, unknown>): number => {
    const raw = container.contentBlockIndex;

    return typeof raw === "number" ? raw : fallbackIndex++;
  };

  for (const container of containers) {
    // Full (non-streaming) Converse message: output.message.content[]
    const message = asObject(container.message);
    const content = message?.content;

    if (Array.isArray(content)) {
      for (const block of content) {
        const blockObj = asObject(block);

        if (!blockObj) {
          continue;
        }

        if (typeof blockObj.text === "string") {
          text += blockObj.text;

          continue;
        }

        const toolUse = asObject(blockObj.toolUse);

        if (toolUse && typeof toolUse.name === "string") {
          toolCallChunks.push({
            type: "tool_call_chunk",
            name: toolUse.name,
            args: JSON.stringify(toolUse.input ?? {}),
            id:
              typeof toolUse.toolUseId === "string"
                ? toolUse.toolUseId
                : randomUUID(),
            index: fallbackIndex++,
          });
        }
      }
    }

    // Streaming: contentBlockStart carries a new toolUse (id + name)
    const contentBlockStart = asObject(container.contentBlockStart);

    if (contentBlockStart) {
      const start = asObject(contentBlockStart.start);
      const toolUse = asObject(start?.toolUse);

      if (toolUse && typeof toolUse.name === "string") {
        toolCallChunks.push({
          type: "tool_call_chunk",
          name: toolUse.name,
          args: "",
          id:
            typeof toolUse.toolUseId === "string"
              ? toolUse.toolUseId
              : randomUUID(),
          index: readIndex(contentBlockStart),
        });
      }
    }

    // Streaming: contentBlockDelta carries text or partial toolUse input
    const contentBlockDelta = asObject(container.contentBlockDelta);

    if (contentBlockDelta) {
      const delta = asObject(contentBlockDelta.delta);

      if (delta && typeof delta.text === "string") {
        text += delta.text;
      }

      const toolUse = asObject(delta?.toolUse);

      if (toolUse && typeof toolUse.input === "string") {
        toolCallChunks.push({
          type: "tool_call_chunk",
          args: toolUse.input,
          index: readIndex(contentBlockDelta),
        });
      }
    }

    // Streaming stop reason / non-streaming top-level stopReason
    const messageStop = asObject(container.messageStop);
    const stopReason =
      (typeof messageStop?.stopReason === "string"
        ? messageStop.stopReason
        : undefined) ??
      (typeof container.stopReason === "string"
        ? container.stopReason
        : undefined);

    if (stopReason) {
      finishReason = stopReason;
    }

    // Usage: metadata.usage (streaming) or top-level usage (non-streaming)
    const metadata = asObject(container.metadata);
    usageMetadata =
      mapConverseUsage(metadata?.usage) ??
      mapConverseUsage(container.usage) ??
      usageMetadata;
  }

  if (
    text.length === 0 &&
    toolCallChunks.length === 0 &&
    !usageMetadata &&
    !finishReason
  ) {
    return null;
  }

  const chunkMessage = new AIMessageChunk({
    content: text,
    tool_call_chunks: toolCallChunks,
    ...(usageMetadata ? { usage_metadata: usageMetadata } : {}),
    response_metadata: finishReason ? { finishReason } : {},
  });

  return new ChatGenerationChunk({ message: chunkMessage, text });
}
