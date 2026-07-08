import {
  AIMessage,
  ToolMessage,
  type BaseMessage,
  type MessageContent,
} from "@langchain/core/messages";
import { toJsonSchema } from "@langchain/core/utils/json_schema";

export type GeminiPart =
  | { text: string }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | { functionResponse: { name: string; response: Record<string, unknown> } };

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

export type CisRequest = {
  target: { provider: string; model: string };
  task: {
    type: string;
    prediction_type?: string;
    input: {
      contents: GeminiContent[];
      systemInstruction?: GeminiSystemInstruction;
      tools?: { functionDeclarations: GeminiFunctionDeclaration[] }[];
      generationConfig: Record<string, unknown>;
    };
  };
};

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

      for (const toolCall of message.tool_calls ?? []) {
        parts.push({
          functionCall: { name: toolCall.name, args: toolCall.args ?? {} },
        });
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

      contents.push({
        role: "user",
        parts: [
          { functionResponse: { name, response: toResponseObject(message.content) } },
        ],
      });
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

export function buildCisRequest(
  config: CisRequestConfig,
  messages: BaseMessage[],
  functionDeclarations: GeminiFunctionDeclaration[],
): CisRequest {
  const { contents, systemInstruction } = messagesToGeminiContents(messages);

  const input: CisRequest["task"]["input"] = {
    contents,
    generationConfig: config.generationConfig,
  };

  if (systemInstruction) {
    input.systemInstruction = systemInstruction;
  }

  if (functionDeclarations.length > 0) {
    input.tools = [{ functionDeclarations }];
  }

  const task: CisRequest["task"] = { type: config.taskType, input };

  if (config.predictionType) {
    task.prediction_type = config.predictionType;
  }

  return {
    target: { provider: config.targetProvider, model: config.model },
    task,
  };
}
