# Workday CIS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `workday-cis` provider that drives OpenWiki's deep agent through the Workday CIS `POST /v1alpha1/predictions/stream` (Gemini-shaped) SSE endpoint, with full function/tool calling.

**Architecture:** A custom LangChain `BaseChatModel` (`ChatWorkdayCis`) converts LangChain messages + bound tools into the CIS `{target, task}` envelope (Gemini `contents`/`systemInstruction`/`tools`/`generationConfig`), POSTs to `/predictions/stream`, parses the SSE stream (`event:`/`data:`, `[DONE]`, `event: error`), and yields `AIMessageChunk`s (text + `tool_call_chunks`). Pure protocol logic lives in a separate `workday-cis-protocol.ts` module for testability. All existing providers and streaming (deepagents/LangGraph) stay unchanged.

**Tech Stack:** TypeScript (ESM), `@langchain/core` (`BaseChatModel`, messages, outputs, `toJsonSchema`), Vitest, existing OpenWiki provider/env infrastructure.

**Spec:** `docs/superpowers/specs/2026-07-07-workday-cis-provider-design.md`

---

## Conventions used in every task

- Run tests with: `pnpm test` (single file: `pnpm vitest run test/<file>.test.ts`).
- Typecheck: `pnpm typecheck`. Lint: `pnpm lint`.
- Tests import source with explicit extension, e.g. `from "../src/constants.ts"` (see existing `test/constants.test.ts`).
- Commit after each task once its tests + typecheck pass.

---

## Task 1: Provider registration, env keys, and envelope resolvers (constants)

**Files:**
- Modify: `src/constants.ts`
- Test: `test/constants.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `test/constants.test.ts` (add the new imports to the existing top import block from `../src/constants.ts`):

```ts
// add to the existing import list:
//   DEFAULT_CIS_GENERATION_CONFIG,
//   DEFAULT_CIS_TASK_TYPE,
//   isValidProvider,
//   providerRequiresApiKey,
//   providerRequiresBaseUrl,
//   resolveCisGenerationConfig,
//   resolveCisPredictionType,
//   resolveCisTargetProvider,
//   resolveCisTaskType,
//   WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY,
//   WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY,
//   WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
//   WORKDAY_CIS_TASK_TYPE_ENV_KEY,

describe("workday-cis provider registration", () => {
  test("is a valid, base-url-requiring provider with optional api key", () => {
    expect(isValidProvider("workday-cis")).toBe(true);
    expect(providerRequiresBaseUrl("workday-cis")).toBe(true);
    expect(providerRequiresApiKey("workday-cis")).toBe(false);
    expect(providerRequiresApiKey("openai")).toBe(true);
  });
});

describe("resolveCisTargetProvider", () => {
  test("returns trimmed value or undefined", () => {
    expect(
      resolveCisTargetProvider({ [WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY]: " google " }),
    ).toBe("google");
    expect(resolveCisTargetProvider({})).toBeUndefined();
  });
});

describe("resolveCisTaskType", () => {
  test("defaults to gcp-multimodal-v2 and honors override", () => {
    expect(resolveCisTaskType({})).toBe(DEFAULT_CIS_TASK_TYPE);
    expect(resolveCisTaskType({ [WORKDAY_CIS_TASK_TYPE_ENV_KEY]: " custom " })).toBe(
      "custom",
    );
  });
});

describe("resolveCisPredictionType", () => {
  test("returns trimmed value or undefined", () => {
    expect(
      resolveCisPredictionType({ [WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY]: " text " }),
    ).toBe("text");
    expect(resolveCisPredictionType({})).toBeUndefined();
  });
});

describe("resolveCisGenerationConfig", () => {
  test("returns defaults when unset", () => {
    expect(resolveCisGenerationConfig({})).toEqual(DEFAULT_CIS_GENERATION_CONFIG);
  });

  test("shallow-merges JSON override over defaults", () => {
    expect(
      resolveCisGenerationConfig({
        [WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY]: '{"temperature":0.9,"maxOutputTokens":100}',
      }),
    ).toEqual({ ...DEFAULT_CIS_GENERATION_CONFIG, temperature: 0.9, maxOutputTokens: 100 });
  });

  test("throws on invalid JSON or non-object", () => {
    expect(() =>
      resolveCisGenerationConfig({ [WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY]: "nope" }),
    ).toThrow(/must be a JSON object/);
    expect(() =>
      resolveCisGenerationConfig({ [WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY]: "[1,2]" }),
    ).toThrow(/must be a JSON object/);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run test/constants.test.ts`
Expected: FAIL (imports undefined / `isValidProvider("workday-cis")` false).

- [ ] **Step 3: Implement in `src/constants.ts`**

Add env-key constants after line 14 (`OPENWIKI_MODEL_ID_ENV_KEY`):

```ts
export const WORKDAY_CIS_API_KEY_ENV_KEY = "WORKDAY_CIS_API_KEY";
export const WORKDAY_CIS_BASE_URL_ENV_KEY = "WORKDAY_CIS_BASE_URL";
export const WORKDAY_CIS_HEADERS_ENV_KEY = "WORKDAY_CIS_HEADERS";
export const WORKDAY_CIS_QUERY_ENV_KEY = "WORKDAY_CIS_QUERY";
export const WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY = "WORKDAY_CIS_TARGET_PROVIDER";
export const WORKDAY_CIS_TASK_TYPE_ENV_KEY = "WORKDAY_CIS_TASK_TYPE";
export const WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY = "WORKDAY_CIS_PREDICTION_TYPE";
export const WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY =
  "WORKDAY_CIS_GENERATION_CONFIG";

export const DEFAULT_CIS_TASK_TYPE = "gcp-multimodal-v2";
export const DEFAULT_CIS_GENERATION_CONFIG: Record<string, unknown> = {
  temperature: 0.2,
  maxOutputTokens: 8192,
  topK: 40,
  topP: 0.95,
};
```

Add `"workday-cis"` to the `OpenWikiProvider` union:

```ts
export type OpenWikiProvider =
  | "anthropic"
  | "baseten"
  | "fireworks"
  | "openai"
  | "openai-compatible"
  | "openrouter"
  | "workday-cis";
```

Add `apiKeyOptional?: boolean;` to the `ProviderConfig` type (next to `requiresBaseUrl`):

```ts
  /**
   * When true, the provider can run without an API key (auth supplied via
   * headers/query instead), so the API key is not required at startup.
   */
  apiKeyOptional?: boolean;
```

Add `"workday-cis"` to `SELECTABLE_OPENWIKI_PROVIDERS` (after `"openai-compatible"`):

```ts
export const SELECTABLE_OPENWIKI_PROVIDERS = [
  "openrouter",
  "baseten",
  "fireworks",
  "openai",
  "openai-compatible",
  "workday-cis",
  "anthropic",
] as const satisfies readonly SelectableOpenWikiProvider[];
```

Add the provider config entry to `PROVIDER_CONFIGS` (after the `"openai-compatible"` entry):

```ts
  "workday-cis": {
    apiKeyEnvKey: WORKDAY_CIS_API_KEY_ENV_KEY,
    apiKeyOptional: true,
    baseUrlEnvKey: WORKDAY_CIS_BASE_URL_ENV_KEY,
    headersEnvKey: WORKDAY_CIS_HEADERS_ENV_KEY,
    queryEnvKey: WORKDAY_CIS_QUERY_ENV_KEY,
    requiresBaseUrl: true,
    label: "Workday CIS",
    modelOptions: [],
  },
```

Add `providerRequiresApiKey` next to `providerRequiresBaseUrl` (after line 281):

```ts
export function providerRequiresApiKey(provider: OpenWikiProvider): boolean {
  return getProviderConfig(provider).apiKeyOptional !== true;
}
```

Add the envelope resolvers at the end of the file (before `OPENWIKI_VERSION`):

```ts
export function resolveCisTargetProvider(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  const value = env[WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY]?.trim();

  return value ? value : undefined;
}

export function resolveCisTaskType(
  env: NodeJS.ProcessEnv = process.env,
): string {
  const value = env[WORKDAY_CIS_TASK_TYPE_ENV_KEY]?.trim();

  return value ? value : DEFAULT_CIS_TASK_TYPE;
}

export function resolveCisPredictionType(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  const value = env[WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY]?.trim();

  return value ? value : undefined;
}

export function resolveCisGenerationConfig(
  env: NodeJS.ProcessEnv = process.env,
): Record<string, unknown> {
  const raw = env[WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY]?.trim();

  if (!raw) {
    return { ...DEFAULT_CIS_GENERATION_CONFIG };
  }

  let parsed: unknown;

  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(
      `${WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY} must be a JSON object.`,
    );
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(
      `${WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY} must be a JSON object.`,
    );
  }

  return { ...DEFAULT_CIS_GENERATION_CONFIG, ...(parsed as Record<string, unknown>) };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run test/constants.test.ts && pnpm typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/constants.ts test/constants.test.ts
git commit -m "feat: register workday-cis provider and CIS envelope resolvers"
```

---

## Task 2: Env management & diagnostics for WORKDAY_CIS_* keys

**Files:**
- Modify: `src/env.ts`
- Test: `test/env.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `test/env.test.ts` (reuse its existing imports; add the ones referenced below from `../src/env.ts` and `../src/constants.ts`). Check the top of the existing file first for the import style.

```ts
import {
  CREDENTIAL_DIAGNOSTIC_ENV_KEYS,
  DEBUG_ENV_KEYS,
  MANAGED_ENV_KEYS,
} from "../src/env.ts";
import {
  WORKDAY_CIS_API_KEY_ENV_KEY,
  WORKDAY_CIS_BASE_URL_ENV_KEY,
  WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY,
  WORKDAY_CIS_HEADERS_ENV_KEY,
  WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY,
  WORKDAY_CIS_QUERY_ENV_KEY,
  WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
  WORKDAY_CIS_TASK_TYPE_ENV_KEY,
} from "../src/constants.ts";

describe("workday-cis managed env keys", () => {
  const cisKeys = [
    WORKDAY_CIS_API_KEY_ENV_KEY,
    WORKDAY_CIS_BASE_URL_ENV_KEY,
    WORKDAY_CIS_HEADERS_ENV_KEY,
    WORKDAY_CIS_QUERY_ENV_KEY,
    WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
    WORKDAY_CIS_TASK_TYPE_ENV_KEY,
    WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY,
    WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY,
  ];

  test("all CIS keys are managed and appear in diagnostics + debug dump", () => {
    for (const key of cisKeys) {
      expect(MANAGED_ENV_KEYS).toContain(key);
      expect(CREDENTIAL_DIAGNOSTIC_ENV_KEYS).toContain(key);
      expect(DEBUG_ENV_KEYS).toContain(key);
    }
  });
});
```

If `test/env.test.ts` already tests `getCredentialDiagnostics`, add a case verifying the API key is masked and the base URL is shown in full. Example (adapt to the file's existing setup/teardown of `process.env`):

```ts
describe("workday-cis credential diagnostics", () => {
  test("masks api key, shows base url and target provider in full", async () => {
    process.env[WORKDAY_CIS_API_KEY_ENV_KEY] = "supersecretkey123";
    process.env[WORKDAY_CIS_BASE_URL_ENV_KEY] =
      "https://host/ml/inference/cis/v1alpha1";
    process.env[WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY] = "google";

    const diagnostics = await getCredentialDiagnostics();
    const byKey = new Map(diagnostics.map((d) => [d.key, d]));

    expect(byKey.get(WORKDAY_CIS_API_KEY_ENV_KEY)?.preview).not.toContain(
      "supersecretkey123",
    );
    expect(byKey.get(WORKDAY_CIS_BASE_URL_ENV_KEY)?.preview).toBe(
      JSON.stringify("https://host/ml/inference/cis/v1alpha1"),
    );
    expect(byKey.get(WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY)?.preview).toBe(
      JSON.stringify("google"),
    );

    delete process.env[WORKDAY_CIS_API_KEY_ENV_KEY];
    delete process.env[WORKDAY_CIS_BASE_URL_ENV_KEY];
    delete process.env[WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY];
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run test/env.test.ts`
Expected: FAIL (keys not in `MANAGED_ENV_KEYS`).

- [ ] **Step 3: Implement in `src/env.ts`**

Add imports to the `from "./constants.js"` block:

```ts
  WORKDAY_CIS_API_KEY_ENV_KEY,
  WORKDAY_CIS_BASE_URL_ENV_KEY,
  WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY,
  WORKDAY_CIS_HEADERS_ENV_KEY,
  WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY,
  WORKDAY_CIS_QUERY_ENV_KEY,
  WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
  WORKDAY_CIS_TASK_TYPE_ENV_KEY,
```

Add the keys to `MANAGED_ENV_KEYS` (after the `OPENAI_COMPATIBLE_QUERY_ENV_KEY` line, before `ANTHROPIC_API_KEY_ENV_KEY`):

```ts
  WORKDAY_CIS_API_KEY_ENV_KEY,
  WORKDAY_CIS_BASE_URL_ENV_KEY,
  WORKDAY_CIS_HEADERS_ENV_KEY,
  WORKDAY_CIS_QUERY_ENV_KEY,
  WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
  WORKDAY_CIS_TASK_TYPE_ENV_KEY,
  WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY,
  WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY,
```

Extend the headers-warning branch in `createCredentialDiagnostic` so the CIS headers key also gets JSON warnings:

```ts
          : key === OPENAI_COMPATIBLE_HEADERS_ENV_KEY ||
              key === WORKDAY_CIS_HEADERS_ENV_KEY
            ? getHeadersWarnings(value)
            : getCredentialWarnings(value),
```

Extend `isNonSecretDiagnosticKey` to include the non-secret CIS keys:

```ts
function isNonSecretDiagnosticKey(key: string): boolean {
  return (
    key === OPENWIKI_MODEL_ID_ENV_KEY ||
    key === OPENWIKI_PROVIDER_ENV_KEY ||
    key === ANTHROPIC_BASE_URL_ENV_KEY ||
    key === OPENAI_COMPATIBLE_BASE_URL_ENV_KEY ||
    key === OPENAI_COMPATIBLE_QUERY_ENV_KEY ||
    key === WORKDAY_CIS_BASE_URL_ENV_KEY ||
    key === WORKDAY_CIS_QUERY_ENV_KEY ||
    key === WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY ||
    key === WORKDAY_CIS_TASK_TYPE_ENV_KEY ||
    key === WORKDAY_CIS_PREDICTION_TYPE_ENV_KEY ||
    key === WORKDAY_CIS_GENERATION_CONFIG_ENV_KEY
  );
}
```

Note: `WORKDAY_CIS_API_KEY_ENV_KEY` and `WORKDAY_CIS_HEADERS_ENV_KEY` are intentionally NOT in the non-secret set, so they stay masked.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run test/env.test.ts && pnpm typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/env.ts test/env.test.ts
git commit -m "feat: manage and diagnose WORKDAY_CIS_* env keys"
```

---

## Task 3: CIS protocol module — message & tool conversion (pure functions)

**Files:**
- Create: `src/agent/workday-cis-protocol.ts`
- Test: `test/workday-cis-protocol.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `test/workday-cis-protocol.test.ts`:

```ts
import { describe, expect, test } from "vitest";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  buildCisRequest,
  messagesToGeminiContents,
  toGeminiFunctionDeclarations,
} from "../src/agent/workday-cis-protocol.ts";

describe("messagesToGeminiContents", () => {
  test("maps system to systemInstruction and human to user content", () => {
    const { contents, systemInstruction } = messagesToGeminiContents([
      new SystemMessage("be brief"),
      new HumanMessage("hello"),
    ]);

    expect(systemInstruction).toEqual({ parts: [{ text: "be brief" }] });
    expect(contents).toEqual([{ role: "user", parts: [{ text: "hello" }] }]);
  });

  test("maps ai tool calls to functionCall and tool results to functionResponse", () => {
    const messages = [
      new HumanMessage("list files"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "call_1", name: "ls", args: { path: "." } }],
      }),
      new ToolMessage({ content: "a.txt", tool_call_id: "call_1" }),
    ];

    const { contents } = messagesToGeminiContents(messages);

    expect(contents[1]).toEqual({
      role: "model",
      parts: [{ functionCall: { name: "ls", args: { path: "." } } }],
    });
    expect(contents[2]).toEqual({
      role: "user",
      parts: [{ functionResponse: { name: "ls", response: { result: "a.txt" } } }],
    });
  });
});

describe("toGeminiFunctionDeclarations", () => {
  test("converts a tool schema and strips unsupported keywords", () => {
    const tool = {
      name: "search",
      description: "search docs",
      schema: {
        $schema: "http://json-schema.org/draft-07/schema#",
        type: "object",
        additionalProperties: false,
        properties: { q: { type: "string" } },
        required: ["q"],
      },
    };

    expect(toGeminiFunctionDeclarations([tool])).toEqual([
      {
        name: "search",
        description: "search docs",
        parameters: {
          type: "object",
          properties: { q: { type: "string" } },
          required: ["q"],
        },
      },
    ]);
  });
});

describe("buildCisRequest", () => {
  test("wraps input in target/task envelope with generationConfig", () => {
    const request = buildCisRequest(
      {
        targetProvider: "google",
        model: "gemini-x",
        taskType: "gcp-multimodal-v2",
        predictionType: undefined,
        generationConfig: { temperature: 0.2 },
      },
      [new HumanMessage("hi")],
      [{ name: "ls", description: "", parameters: { type: "object" } }],
    );

    expect(request).toEqual({
      target: { provider: "google", model: "gemini-x" },
      task: {
        type: "gcp-multimodal-v2",
        input: {
          contents: [{ role: "user", parts: [{ text: "hi" }] }],
          generationConfig: { temperature: 0.2 },
          tools: [
            { functionDeclarations: [{ name: "ls", description: "", parameters: { type: "object" } }] },
          ],
        },
      },
    });
  });

  test("includes prediction_type only when provided", () => {
    const request = buildCisRequest(
      {
        targetProvider: "google",
        model: "m",
        taskType: "t",
        predictionType: "text",
        generationConfig: {},
      },
      [new HumanMessage("hi")],
      [],
    );

    expect(request.task.prediction_type).toBe("text");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run test/workday-cis-protocol.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/agent/workday-cis-protocol.ts` (conversion half)**

```ts
import type { BaseMessage, MessageContent } from "@langchain/core/messages";
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
    if (message.getType() === "ai") {
      const toolCalls =
        (message as { tool_calls?: { id?: string; name: string }[] })
          .tool_calls ?? [];

      for (const toolCall of toolCalls) {
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

    if (type === "ai") {
      const parts: GeminiPart[] = [];
      const text = extractText(message.content);

      if (text) {
        parts.push({ text });
      }

      const toolCalls =
        (message as { tool_calls?: { name: string; args?: Record<string, unknown> }[] })
          .tool_calls ?? [];

      for (const toolCall of toolCalls) {
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

    if (type === "tool" || type === "function") {
      const toolCallId = (message as { tool_call_id?: string }).tool_call_id;
      const name =
        (message as { name?: string }).name ??
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
```

Note: `toJsonSchema` accepts an already-plain JSON schema object and returns it (idempotent), and also converts zod schemas — so both the test's plain-object `schema` and real deepagents zod tools work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run test/workday-cis-protocol.test.ts && pnpm typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/workday-cis-protocol.ts test/workday-cis-protocol.test.ts
git commit -m "feat: add CIS message and tool conversion helpers"
```

---

## Task 4: CIS protocol module — SSE parsing & chunk mapping

**Files:**
- Modify: `src/agent/workday-cis-protocol.ts`
- Test: `test/workday-cis-protocol.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `test/workday-cis-protocol.test.ts` (add these names to the existing import from the protocol module: `splitSseEvents`, `parseSseEvent`, `cisChunkToGeneration`):

```ts
describe("splitSseEvents", () => {
  test("splits complete events and keeps the incomplete remainder", () => {
    const { events, rest } = splitSseEvents("event: a\ndata: 1\n\ndata: 2\n\ndata: par");

    expect(events).toEqual(["event: a\ndata: 1", "data: 2"]);
    expect(rest).toBe("data: par");
  });
});

describe("parseSseEvent", () => {
  test("parses event type and joins data lines, stripping one leading space", () => {
    expect(parseSseEvent("event: error\ndata: boom")).toEqual({
      event: "error",
      data: "boom",
    });
    expect(parseSseEvent("data: line1\ndata: line2")).toEqual({
      event: "message",
      data: "line1\nline2",
    });
  });
});

describe("cisChunkToGeneration", () => {
  test("maps text parts to a chunk", () => {
    const chunk = cisChunkToGeneration({
      output: { candidates: [{ content: { parts: [{ text: "Hello" }] } }] },
    });

    expect(chunk?.text).toBe("Hello");
  });

  test("maps functionCall parts to tool_call_chunks with stringified args", () => {
    const chunk = cisChunkToGeneration({
      output: {
        candidates: [
          { content: { parts: [{ functionCall: { name: "ls", args: { path: "." } } }] } },
        ],
      },
    });

    const toolCallChunks = chunk?.message.tool_call_chunks ?? [];
    expect(toolCallChunks[0]?.name).toBe("ls");
    expect(toolCallChunks[0]?.args).toBe('{"path":"."}');
    expect(toolCallChunks[0]?.index).toBe(0);
  });

  test("returns null for empty payloads", () => {
    expect(cisChunkToGeneration({ output: { candidates: [] } })).toBeNull();
    expect(cisChunkToGeneration({})).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run test/workday-cis-protocol.test.ts`
Expected: FAIL (new exports undefined).

- [ ] **Step 3: Implement (append to `src/agent/workday-cis-protocol.ts`)**

Add imports at the top of the file:

```ts
import { randomUUID } from "node:crypto";
import { AIMessageChunk } from "@langchain/core/messages";
import { ChatGenerationChunk } from "@langchain/core/outputs";
```

Append the parsing + mapping functions:

```ts
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

      if (typeof partText === "string") {
        text += partText;

        continue;
      }

      const functionCall = (part as {
        functionCall?: { name?: string; args?: Record<string, unknown> };
      }).functionCall;

      if (functionCall?.name) {
        toolCallChunks.push({
          type: "tool_call_chunk",
          name: functionCall.name,
          args: JSON.stringify(functionCall.args ?? {}),
          id: randomUUID(),
          index: index++,
        });
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
    response_metadata: finishReason ? { finishReason } : {},
  });

  return new ChatGenerationChunk({ message, text });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run test/workday-cis-protocol.test.ts && pnpm typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/workday-cis-protocol.ts test/workday-cis-protocol.test.ts
git commit -m "feat: add CIS SSE parsing and chunk mapping"
```

---

## Task 5: `ChatWorkdayCis` chat model

**Files:**
- Create: `src/agent/workday-cis-chat-model.ts`
- Test: `test/workday-cis-chat-model.test.ts`

- [ ] **Step 1: Write the failing test (end-to-end with mocked fetch)**

Create `test/workday-cis-chat-model.test.ts`:

```ts
import { afterEach, describe, expect, test, vi } from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { ChatWorkdayCis } from "../src/agent/workday-cis-chat-model.ts";

function sseStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }

      controller.close();
    },
  });
}

function mockFetchOk(chunks: string[]): typeof fetch {
  return vi.fn(async () =>
    new Response(sseStream(chunks), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }),
  ) as unknown as typeof fetch;
}

function createModel(fetchImpl: typeof fetch): ChatWorkdayCis {
  return new ChatWorkdayCis({
    model: "gemini-x",
    baseURL: "https://host/ml/inference/cis/v1alpha1",
    apiKey: "k",
    headers: { "wd-pca-feature-key": "user" },
    query: { bypass_auth: "true" },
    targetProvider: "google",
    taskType: "gcp-multimodal-v2",
    predictionType: undefined,
    generationConfig: { temperature: 0.2 },
    fetchImpl,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ChatWorkdayCis", () => {
  test("streams text tokens from SSE data events", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}}\n\n',
      'data: {"output":{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const model = createModel(fetchImpl);

    const tokens: string[] = [];
    for await (const chunk of await model.stream([new HumanMessage("hi")])) {
      tokens.push(chunk.content as string);
    }

    expect(tokens.join("")).toBe("Hello");
  });

  test("POSTs the CIS envelope to /predictions/stream with headers and query", async () => {
    const fetchImpl = mockFetchOk(["data: [DONE]\n\n"]);
    const model = createModel(fetchImpl);

    await model.invoke([new HumanMessage("hi")]);

    const [url, init] = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock
      .calls[0];
    expect(String(url)).toBe(
      "https://host/ml/inference/cis/v1alpha1/predictions/stream?bypass_auth=true",
    );
    const headers = init.headers as Record<string, string>;
    expect(headers.authorization).toBe("Bearer k");
    expect(headers["wd-pca-feature-key"]).toBe("user");
    const body = JSON.parse(init.body as string);
    expect(body.target).toEqual({ provider: "google", model: "gemini-x" });
    expect(body.task.type).toBe("gcp-multimodal-v2");
  });

  test("throws on an SSE error event", async () => {
    const fetchImpl = mockFetchOk(["event: error\ndata: upstream boom\n\n"]);
    const model = createModel(fetchImpl);

    await expect(model.invoke([new HumanMessage("hi")])).rejects.toThrow(
      /upstream boom/,
    );
  });

  test("throws a clear error on non-OK HTTP", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response("bad request", { status: 422, statusText: "Unprocessable" }),
    ) as unknown as typeof fetch;
    const model = createModel(fetchImpl);

    await expect(model.invoke([new HumanMessage("hi")])).rejects.toThrow(/422/);
  });

  test("emits functionCall parts as tool calls", async () => {
    const fetchImpl = mockFetchOk([
      'data: {"output":{"candidates":[{"content":{"parts":[{"functionCall":{"name":"ls","args":{"path":"."}}}]}}]}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const model = createModel(fetchImpl);

    const result = await model.invoke([new HumanMessage("hi")]);

    expect(result.tool_calls?.[0]?.name).toBe("ls");
    expect(result.tool_calls?.[0]?.args).toEqual({ path: "." });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm vitest run test/workday-cis-chat-model.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/agent/workday-cis-chat-model.ts`**

```ts
import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import {
  ChatGenerationChunk,
  type ChatResult,
} from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import {
  buildCisRequest,
  cisChunkToGeneration,
  parseSseEvent,
  splitSseEvents,
  toGeminiFunctionDeclarations,
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
    } as Partial<ChatWorkdayCisCallOptions>);
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

    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true }).replace(/\r/gu, "");
      const split = splitSseEvents(buffer);
      buffer = split.rest;

      for (const rawEvent of split.events) {
        const { event, data } = parseSseEvent(rawEvent);

        if (event === "error") {
          throw new Error(`Workday CIS streaming error: ${data}`);
        }

        if (data === "[DONE]") {
          doneReceived = true;

          continue;
        }

        if (data.length === 0) {
          continue;
        }

        let payload: unknown;

        try {
          payload = JSON.parse(data);
        } catch {
          continue;
        }

        const chunk = cisChunkToGeneration(payload);

        if (chunk) {
          yieldedAny = true;

          if (chunk.text.length > 0) {
            await runManager?.handleLLMNewToken(chunk.text);
          }

          yield chunk;
        }
      }

      if (doneReceived) {
        break;
      }
    }

    if (!doneReceived && !yieldedAny) {
      throw new Error(
        "Workday CIS stream ended without any data ([DONE] not received).",
      );
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

    return { generations: [{ text, message }] };
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm vitest run test/workday-cis-chat-model.test.ts && pnpm typecheck`
Expected: PASS.

Note: `this.withConfig(...)` is the correct mechanism here (this is what `ChatOpenAI.bindTools` itself uses internally in this LangChain version to merge custom call options like `tools` onto the bound runnable) — `Runnable.bind()` does NOT exist in `@langchain/core@1.2.1`, only `withConfig()`. Do not use `.bind()`.

If typecheck flags the `bindTools` return type or the `BaseLanguageModelInput` type import, adjust the generics/imports to satisfy the compiler — the runtime behavior (merging `cisFunctionDeclarations` into call options via `this.withConfig`) is what matters.

- [ ] **Step 5: Commit**

```bash
git add src/agent/workday-cis-chat-model.ts test/workday-cis-chat-model.test.ts
git commit -m "feat: add ChatWorkdayCis streaming chat model"
```

---

## Task 6: Wire the provider into the agent

**Files:**
- Modify: `src/agent/index.ts`

- [ ] **Step 1: Add imports**

In the `from "../constants.js"` import block, add:

```ts
  providerRequiresApiKey,
  resolveCisGenerationConfig,
  resolveCisPredictionType,
  resolveCisTargetProvider,
  resolveCisTaskType,
  WORKDAY_CIS_API_KEY_ENV_KEY,
  WORKDAY_CIS_HEADERS_ENV_KEY,
  WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY,
```

Add a new import for the model class (after the `../constants.js` block or near the other agent imports):

```ts
import { ChatWorkdayCis } from "./workday-cis-chat-model.js";
```

- [ ] **Step 2: Make the API key optional for providers that allow it**

Replace the body of `ensureProviderKey` (around line 392):

```ts
function ensureProviderKey(provider: OpenWikiProvider): void {
  if (!providerRequiresApiKey(provider)) {
    return;
  }

  const apiKeyEnvKey = getProviderApiKeyEnvKey(provider);

  if (!process.env[apiKeyEnvKey]) {
    throw new Error(
      `${apiKeyEnvKey} is required to run OpenWiki with ${getProviderLabel(provider)}.`,
    );
  }
}
```

- [ ] **Step 3: Add the `workday-cis` branch in `createModel`**

Insert this block in `createModel` (before the final generic `ChatOpenAI` block, i.e. after the `openrouter` branch that ends around line 456):

```ts
  if (provider === "workday-cis") {
    const targetProvider = resolveCisTargetProvider();

    if (!targetProvider) {
      throw new Error(
        `${WORKDAY_CIS_TARGET_PROVIDER_ENV_KEY} is required to run OpenWiki with ${getProviderLabel(provider)}.`,
      );
    }

    const baseURL = resolveProviderBaseUrl(provider);

    if (!baseURL) {
      throw new Error(
        `${getProviderBaseUrlEnvKey(provider) ?? "base URL"} is required to run OpenWiki with ${getProviderLabel(provider)}.`,
      );
    }

    return new ChatWorkdayCis({
      model: modelId,
      baseURL,
      apiKey: process.env[getProviderApiKeyEnvKey(provider)],
      headers: resolveProviderHeaders(provider),
      query: resolveProviderQuery(provider),
      targetProvider,
      taskType: resolveCisTaskType(),
      predictionType: resolveCisPredictionType(),
      generationConfig: resolveCisGenerationConfig(),
    });
  }
```

- [ ] **Step 4: Capture the CIS endpoint in debug fetch**

Update `isProviderChatFetchInput` (around line 1146) so it also matches the predictions stream path:

```ts
function isProviderChatFetchInput(
  input: Parameters<typeof fetch>[0],
  providerBaseUrl?: string,
): boolean {
  const url = getFetchInputUrl(input);

  if (
    url === null ||
    !(url.includes("/chat/completions") || url.includes("/predictions/stream"))
  ) {
    return false;
  }

  if (url.startsWith(OPENROUTER_BASE_URL)) {
    return true;
  }

  return providerBaseUrl !== undefined && url.startsWith(providerBaseUrl);
}
```

- [ ] **Step 5: Mask CIS secrets in the env debug dump**

Update `formatDebugValue` (around line 1361). The API key is already handled by the `key.endsWith("_API_KEY")` branch. Add the CIS headers key to the masked-headers branch:

```ts
  if (
    key === OPENAI_COMPATIBLE_HEADERS_ENV_KEY ||
    key === WORKDAY_CIS_HEADERS_ENV_KEY
  ) {
    return `set(length=${value.length})`;
  }
```

The other `WORKDAY_CIS_*` keys fall through to the default `set(value=...)`/length rendering, which is acceptable (they are non-secret).

- [ ] **Step 6: Typecheck, lint, and run the full suite**

Run: `pnpm typecheck && pnpm lint && pnpm test`
Expected: PASS (all existing + new tests).

- [ ] **Step 7: Commit**

```bash
git add src/agent/index.ts
git commit -m "feat: wire workday-cis provider into agent model creation"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `openwiki/operations/credentials-and-updates.md`
- Modify: `openwiki/cli/usage.md`
- Modify: `openwiki/agent/workflow.md`

- [ ] **Step 1: Locate the existing openai-compatible docs**

Search the four files for `OPENAI_COMPATIBLE_HEADERS` and `openai-compatible` to find where providers/env vars are documented, so the new `workday-cis` content sits next to them.

- [ ] **Step 2: Add `workday-cis` documentation**

In each file, alongside the existing `openai-compatible` documentation, add a `workday-cis` section describing:

- Purpose: talks to Workday CIS `POST /v1alpha1/predictions/stream` (Gemini-shaped, SSE), with tool calling.
- Env vars (copy this table):

```markdown
| Env var | Purpose | Default |
|---|---|---|
| `WORKDAY_CIS_BASE_URL` | Base URL up to `/v1alpha1` (required) | — |
| `WORKDAY_CIS_API_KEY` | Optional bearer token | unset |
| `WORKDAY_CIS_HEADERS` | Extra headers as a JSON object string | unset |
| `WORKDAY_CIS_QUERY` | Raw query string appended to every call (e.g. `bypass_auth=true`) | unset |
| `WORKDAY_CIS_TARGET_PROVIDER` | `target.provider` in the CIS envelope (required) | — |
| `WORKDAY_CIS_TASK_TYPE` | `task.type` | `gcp-multimodal-v2` |
| `WORKDAY_CIS_PREDICTION_TYPE` | `task.prediction_type` (omitted when unset) | unset |
| `WORKDAY_CIS_GENERATION_CONFIG` | JSON overriding defaults `{temperature:0.2,maxOutputTokens:8192,topK:40,topP:0.95}` | unset |
```

- Note: `target.model` comes from `OPENWIKI_MODEL_ID`.
- Diagnostics: `WORKDAY_CIS_API_KEY` and `WORKDAY_CIS_HEADERS` are masked; the rest show their full value.
- Example `~/.openwiki/.env`:

```
OPENWIKI_PROVIDER=workday-cis
OPENWIKI_MODEL_ID=gemini-2.5-pro
WORKDAY_CIS_BASE_URL=https://host/ml/inference/cis/v1alpha1
WORKDAY_CIS_TARGET_PROVIDER=google
WORKDAY_CIS_HEADERS={"wd-pca-feature-key":"your-user"}
WORKDAY_CIS_QUERY=bypass_auth=true
```

- [ ] **Step 3: Commit**

```bash
git add README.md openwiki/operations/credentials-and-updates.md openwiki/cli/usage.md openwiki/agent/workflow.md
git commit -m "docs: document the workday-cis provider"
```

---

## Final verification

- [ ] **Step 1: Full gate**

Run: `pnpm typecheck && pnpm lint && pnpm test && pnpm build`
Expected: all PASS; `postbuild` chmods `dist/cli.js`.

- [ ] **Step 2: Manual smoke (optional, requires real CIS access)**

```bash
OPENWIKI_DEBUG=1 openwiki chat "hello"
```
Expected: streamed response; on failure, the `openRouterDebug.request.body` diagnostic shows the CIS envelope sent to `/predictions/stream`.

---

## Notes for the implementer

- **DRY:** Header/query/base-url resolution reuses existing `resolveProviderHeaders`, `resolveProviderQuery`, `resolveProviderBaseUrl`; do not duplicate parsing.
- **YAGNI:** No multimodal/image parts, no retries, no non-`gcp-multimodal-v2` response handling.
- **Type names to keep consistent across tasks:** `GeminiFunctionDeclaration`, `CisRequest`, `CisRequestConfig`, `ChatWorkdayCisParams`, `ChatWorkdayCisCallOptions`, and functions `messagesToGeminiContents`, `toGeminiFunctionDeclarations`, `buildCisRequest`, `splitSseEvents`, `parseSseEvent`, `cisChunkToGeneration`.
- The deep agent (`createDeepAgent`) and `parseStreamEvent` are unchanged — `ChatWorkdayCis` presents as a standard streaming, tool-calling chat model.
