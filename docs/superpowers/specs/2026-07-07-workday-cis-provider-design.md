# Workday CIS Provider (predictions/stream) — Design

Date: 2026-07-07
Status: Approved (pending user review of this spec)

## Summary

Add a new OpenWiki provider, `workday-cis`, that talks to the Workday CIS
(Cortex Inference Service) native streaming endpoint
`POST /v1alpha1/predictions/stream` instead of an OpenAI `/chat/completions`
surface.

CIS uses a Gemini-shaped protocol: the request body is a
`{ target, task }` envelope whose `task.input` is a Gemini
`generateContent`-style payload (`contents`, `systemInstruction`, `tools`,
`generationConfig`), and the response is a Server-Sent Events (SSE) stream of
`{ output: <GeminiResponse> }` chunks terminated by a `data: [DONE]` sentinel.

The provider is implemented as a custom LangChain `BaseChatModel`
(`ChatWorkdayCis`) so the rest of the stack (deepagents / LangGraph) keeps
seeing a standard streaming, tool-calling chat model while all CIS-specific
translation stays in one module.

## Goals

- New provider `workday-cis`, separate from the generic `openai-compatible`
  provider (which remains an OpenAI client).
- Full function/tool calling so the agentic `init`/`update` doc generation
  works (deepagents injects filesystem/todo/shell tools that require tool
  calling), plus the plain `chat` command.
- Streaming via SSE; non-streaming call paths supported by aggregating the
  stream.
- Configuration via dedicated `WORKDAY_CIS_*` env vars, mirroring the existing
  base-url/api-key/headers/query pattern.
- Fail-fast, well-diagnosed errors (no silent "empty response").

## Non-goals (YAGNI)

- Multimodal/image input parts (text + function calling only).
- Provider-level retries / fallback model routing.
- Task types other than the default (`gcp-multimodal-v2`) response shape.
- A local OpenAI↔CIS proxy shim.

## Chosen approach

Approach A: a custom `ChatWorkdayCis extends BaseChatModel` with hand-written
LangChain↔Gemini message/tool conversion and a hand-written SSE parser. No new
runtime dependencies; all logic is unit-testable and isolated in one module.

Rejected:
- Approach B (reuse `@langchain/google-*` converters): adds a dependency and
  couples to unstable internal shapes while the CIS envelope + SSE still need
  custom handling.
- Approach C (local OpenAI↔CIS proxy): adds a runtime process and more failure
  modes; unsuitable for a CLI.

## Configuration

New provider `workday-cis` registered in `src/constants.ts` via the existing
`ProviderConfig` pattern. `target.model` is sourced from the existing
`OPENWIKI_MODEL_ID`.

| Env var | Purpose | Secret? | Default |
|---|---|---|---|
| `WORKDAY_CIS_BASE_URL` | Base up to `/v1alpha1`; code appends `/predictions/stream` | no | required |
| `WORKDAY_CIS_API_KEY` | Optional → `Authorization: Bearer <key>` when set | yes | unset |
| `WORKDAY_CIS_HEADERS` | Custom headers, JSON object string | masked | unset |
| `WORKDAY_CIS_QUERY` | Raw query string (e.g. `bypass_auth=true`) | no | unset |
| `WORKDAY_CIS_TARGET_PROVIDER` | `target.provider` | no | required |
| `WORKDAY_CIS_TASK_TYPE` | `task.type` | no | `gcp-multimodal-v2` |
| `WORKDAY_CIS_PREDICTION_TYPE` | `task.prediction_type`; omitted when unset (endpoint default) | no | unset |
| `WORKDAY_CIS_GENERATION_CONFIG` | JSON override, shallow-merged over defaults | no | unset |

- `generationConfig` defaults:
  `{ temperature: 0.2, maxOutputTokens: 8192, topK: 40, topP: 0.95 }`.
- Resolver helpers in `constants.ts`: `resolveCisTargetProvider`,
  `resolveCisTaskType`, `resolveCisPredictionType`,
  `resolveCisGenerationConfig`; reuse existing `resolveProviderHeaders` /
  `resolveProviderQuery` / `resolveProviderBaseUrl`.
- Malformed JSON in `WORKDAY_CIS_HEADERS` or `WORKDAY_CIS_GENERATION_CONFIG`
  fails fast with a clear error (matching current header behavior).

## `ChatWorkdayCis` chat model

New module `src/agent/workday-cis-chat-model.ts` exporting
`class ChatWorkdayCis extends BaseChatModel`. `_llmType()` returns
`"workday-cis"`.

### Tool binding & schema conversion

- Override `bindTools(tools)` to store converted Gemini `functionDeclarations`
  (via `this.withConfig`/bound call options), read back at request time.
- `toGeminiFunctionDeclarations()` helper: LangChain tool (zod) → JSON Schema
  (`zod-to-json-schema`, transitive via LangChain) → Gemini-supported subset
  (strip `$schema`, `additionalProperties`, and unsupported keywords). Own
  tests.

### Request building — `buildCisRequest(messages, tools)`

Message role mapping to Gemini `contents`:

- `SystemMessage` → `systemInstruction: { parts: [{ text }] }` (concatenated
  across all system messages).
- `HumanMessage` → `{ role: "user", parts: [{ text }] }`.
- `AIMessage` → `{ role: "model", parts: [ ...textParts,
  ...toolCalls.map(tc => ({ functionCall: { name, args } })) ] }`.
- `ToolMessage` → `{ role: "user", parts: [{ functionResponse: { name,
  response } }] }` where `name` is resolved from the matching prior tool call
  (by `tool_call_id`).

Assembled `input = { contents, systemInstruction?, tools?, generationConfig }`.

Envelope:
`{ target: { provider, model }, task: { type, prediction_type?, input } }`.

### Streaming — `_streamResponseChunks` (primary path)

1. `POST ${baseURL}/predictions/stream${query}` with headers (+ `Authorization:
   Bearer` when API key set), body = envelope JSON.
2. On non-OK HTTP: read body and throw a descriptive error (also captured by
   the debug-fetch mechanism).
3. Read `response.body` as a byte stream; buffer and split SSE events on
   `\n\n`.
4. Per event, parse line prefixes `event:` and `data:`:
   - `event: error` → throw with the `data` payload text.
   - `data: [DONE]` → clean end of stream.
   - otherwise `JSON.parse(payload)` → `output.candidates[].content.parts[]`:
     - `part.text` → `AIMessageChunk` content + `runManager.handleLLMNewToken`.
     - `part.functionCall` → `tool_call_chunks: [{ name, args:
       JSON.stringify(args), index, id }]`.
   - map `output.usageMetadata` → `usage_metadata`; `candidates[].finishReason`
     → `response_metadata`.
5. Yield `ChatGenerationChunk`s as they arrive.

### Non-streaming — `_generate`

Consumes `_streamResponseChunks` internally and concatenates into a single
`ChatResult`, so non-streaming LangChain call paths work.

The deep agent keeps seeing standard LangChain chunks, so `agent.stream(...)`
and `parseStreamEvent` in `src/agent/index.ts` are unchanged.

## Wiring

`src/agent/index.ts`:

- Add a `provider === "workday-cis"` branch in `createModel` returning
  `new ChatWorkdayCis({ model: modelId, baseURL, apiKey, headers, query,
  targetProvider, taskType, predictionType, generationConfig })`.
- Extend `isProviderChatFetchInput` to also match `/predictions/stream` (today
  it only matches `/chat/completions`) so request-body debug capture works for
  this provider.
- Update `formatDebugValue` to mask the API key and headers and show the other
  `WORKDAY_CIS_*` values.

## Env & diagnostics — `src/env.ts`

- Add all `WORKDAY_CIS_*` keys to `MANAGED_ENV_KEYS`.
- Diagnostics classification:
  - `WORKDAY_CIS_API_KEY` → masked.
  - `WORKDAY_CIS_HEADERS` → masked, JSON-validity warnings via existing
    `getHeadersWarnings`.
  - `WORKDAY_CIS_BASE_URL`, `_QUERY`, `_TARGET_PROVIDER`, `_TASK_TYPE`,
    `_PREDICTION_TYPE`, `_GENERATION_CONFIG` → non-secret (full value) via
    `isNonSecretDiagnosticKey`.

## Error handling

- Non-OK HTTP → throw with status + body preview.
- SSE `event: error` → throw with the payload text.
- Stream ends without `[DONE]` and produced no content → throw a clear
  "incomplete/empty CIS stream" error (avoids the silent empty-response class of
  bug).

## Testing (Vitest)

- `toGeminiFunctionDeclarations`: zod tool → sanitized Gemini schema.
- `buildCisRequest`: role mapping, tool calls, tool responses,
  `systemInstruction`, envelope fields, `generationConfig` merge.
- SSE parser: multi-event buffers, events split across chunks, `event: error`,
  `[DONE]`, text + `functionCall` parts → chunks.
- `_streamResponseChunks` end-to-end with a mocked `fetch` returning a
  `ReadableStream` of SSE bytes.
- Env resolution + diagnostics classification (secret vs non-secret, JSON
  fail-fast).

## Docs

Update `README.md`, `openwiki/operations/credentials-and-updates.md`,
`openwiki/cli/usage.md`, and `openwiki/agent/workflow.md` to document the
`workday-cis` provider, its env vars, formats, and diagnostics behavior.
