# OpenAI-compatible Custom POST Headers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `OPENAI_COMPATIBLE_HEADERS` env var (a JSON object of string headers) that injects custom HTTP headers into the `openai-compatible` provider's requests.

**Architecture:** Extend the existing generic provider-config pattern (mirroring `baseUrlEnvKey`) with a `headersEnvKey` and a `resolveProviderHeaders` resolver in `src/constants.ts`. The resolved headers flow into the shared `ChatOpenAI` client as `configuration.defaultHeaders` in `src/agent/index.ts`. The key is registered in `MANAGED_ENV_KEYS` (`src/env.ts`) and treated as a masked secret in diagnostics/debug.

**Tech Stack:** TypeScript (ESM, `.js` import specifiers), Vitest, `@langchain/openai` `ChatOpenAI`.

---

## File Structure

- `src/constants.ts` — new env-key constant, `headersEnvKey` config field, `getProviderHeadersEnvKey`, `resolveProviderHeaders`, `isValidHeadersRecord`.
- `src/agent/index.ts` — inject `defaultHeaders` in `createModel`; fail-fast resolve + debug in `runOpenWikiAgent`; mask headers key in `formatDebugValue`.
- `src/env.ts` — register key in `MANAGED_ENV_KEYS`; headers-specific diagnostic warnings.
- `test/constants.test.ts` — resolver/validator tests.
- `test/env.test.ts` — managed-key + diagnostics masking tests.
- `README.md`, `openwiki/operations/credentials-and-updates.md`, `openwiki/cli/usage.md`, `openwiki/agent/workflow.md` — docs.

Conventions to follow:

- All local imports use `.js` specifiers (e.g. `from "../src/constants.ts"` in tests, `from "./constants.js"` in src).
- Run tests with `pnpm test` (Vitest). Typecheck with `pnpm typecheck`.

---

## Task 1: Constant + config field + validator/resolver

**Files:**

- Modify: `src/constants.ts`
- Test: `test/constants.test.ts`

- [ ] **Step 1: Write the failing tests**

Add to `test/constants.test.ts`. First extend the import block at the top:

```ts
import {
  DEFAULT_MODEL_ID,
  DEFAULT_PROVIDER,
  getDefaultModelId,
  getProviderHeadersEnvKey,
  isValidBaseUrl,
  isValidModelId,
  isValidProvider,
  normalizeModelId,
  normalizeProvider,
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  resolveConfiguredProvider,
  resolveProviderBaseUrl,
  resolveProviderHeaders,
} from "../src/constants.ts";
```

Then append these `describe` blocks at the end of the file:

```ts
describe("getProviderHeadersEnvKey", () => {
  test("returns the headers key for openai-compatible", () => {
    expect(getProviderHeadersEnvKey("openai-compatible")).toBe(
      OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
    );
  });

  test("returns undefined for providers without a headers key", () => {
    expect(getProviderHeadersEnvKey("openrouter")).toBeUndefined();
    expect(getProviderHeadersEnvKey("anthropic")).toBeUndefined();
  });
});

describe("resolveProviderHeaders", () => {
  test("returns undefined when the header env var is unset", () => {
    expect(resolveProviderHeaders("openai-compatible", {})).toBeUndefined();
  });

  test("returns undefined when the value is whitespace only", () => {
    expect(
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: "   ",
      }),
    ).toBeUndefined();
  });

  test("returns undefined for providers without a headers key", () => {
    expect(
      resolveProviderHeaders("openrouter", {
        OPENAI_COMPATIBLE_HEADERS: '{"X-Api-Key":"abc"}',
      }),
    ).toBeUndefined();
  });

  test("parses a valid JSON object of string values", () => {
    expect(
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: '{"X-Api-Key":"abc","X-Org":"acme"}',
      }),
    ).toEqual({ "X-Api-Key": "abc", "X-Org": "acme" });
  });

  test("throws on invalid JSON", () => {
    expect(() =>
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: "{not json}",
      }),
    ).toThrow(/OPENAI_COMPATIBLE_HEADERS/u);
  });

  test("throws when the value is a JSON array", () => {
    expect(() =>
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: '["X-Api-Key","abc"]',
      }),
    ).toThrow(/OPENAI_COMPATIBLE_HEADERS/u);
  });

  test("throws when a value is not a string", () => {
    expect(() =>
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: '{"X-Count":1}',
      }),
    ).toThrow(/OPENAI_COMPATIBLE_HEADERS/u);
  });

  test("throws when a value is a nested object", () => {
    expect(() =>
      resolveProviderHeaders("openai-compatible", {
        OPENAI_COMPATIBLE_HEADERS: '{"X-Meta":{"a":"b"}}',
      }),
    ).toThrow(/OPENAI_COMPATIBLE_HEADERS/u);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm test test/constants.test.ts`
Expected: FAIL — `getProviderHeadersEnvKey`, `resolveProviderHeaders`, and `OPENAI_COMPATIBLE_HEADERS_ENV_KEY` are not exported.

- [ ] **Step 3: Add the constant and config field**

In `src/constants.ts`, add the env-key constant next to the other openai-compatible keys (after line 7, `OPENAI_COMPATIBLE_BASE_URL_ENV_KEY`):

```ts
export const OPENAI_COMPATIBLE_HEADERS_ENV_KEY = "OPENAI_COMPATIBLE_HEADERS";
```

Add a `headersEnvKey` field to the `ProviderConfig` type (after the `baseUrlEnvKey` block, before `requiresBaseUrl`):

```ts
  /**
   * Environment variable that, when set, supplies extra request headers (as a
   * JSON object of string values) sent on every request for this provider.
   */
  headersEnvKey?: string;
```

Set it on the `openai-compatible` config:

```ts
  "openai-compatible": {
    apiKeyEnvKey: OPENAI_COMPATIBLE_API_KEY_ENV_KEY,
    baseUrlEnvKey: OPENAI_COMPATIBLE_BASE_URL_ENV_KEY,
    headersEnvKey: OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
    requiresBaseUrl: true,
    label: "OpenAI-compatible",
    modelOptions: [],
  },
```

- [ ] **Step 4: Add the resolver, getter, and validator**

In `src/constants.ts`, add after `getProviderBaseUrlEnvKey` (around line 169):

```ts
export function getProviderHeadersEnvKey(
  provider: OpenWikiProvider,
): string | undefined {
  return getProviderConfig(provider).headersEnvKey;
}

/**
 * Resolves custom request headers for a provider from its configured env var.
 * Returns `undefined` when the provider has no headers key or the value is
 * unset/blank. Throws when the value is set but is not a JSON object whose
 * every value is a string.
 */
export function resolveProviderHeaders(
  provider: OpenWikiProvider,
  env: NodeJS.ProcessEnv = process.env,
): Record<string, string> | undefined {
  const headersEnvKey = getProviderConfig(provider).headersEnvKey;

  if (!headersEnvKey) {
    return undefined;
  }

  const raw = env[headersEnvKey]?.trim();

  if (!raw) {
    return undefined;
  }

  let parsed: unknown;

  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(
      `${headersEnvKey} must be a JSON object of string header values.`,
    );
  }

  if (!isValidHeadersRecord(parsed)) {
    throw new Error(
      `${headersEnvKey} must be a JSON object of string header values.`,
    );
  }

  return parsed;
}

export function isValidHeadersRecord(
  value: unknown,
): value is Record<string, string> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }

  return Object.values(value).every((entry) => typeof entry === "string");
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm test test/constants.test.ts`
Expected: PASS (all new tests plus existing ones).

- [ ] **Step 6: Typecheck**

Run: `pnpm typecheck`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/constants.ts test/constants.test.ts
git commit -m "feat: add OPENAI_COMPATIBLE_HEADERS resolver and validation"
```

---

## Task 2: Register key in env management + diagnostics

**Files:**

- Modify: `src/env.ts`
- Test: `test/env.test.ts`

- [ ] **Step 1: Write the failing tests**

In `test/env.test.ts`, extend the import on line 2 to include the managed-key lists and diagnostics helper. Replace:

```ts
import { formatEnv, parseEnv } from "../src/env.ts";
```

with:

```ts
import {
  CREDENTIAL_DIAGNOSTIC_ENV_KEYS,
  DEBUG_ENV_KEYS,
  formatEnv,
  MANAGED_ENV_KEYS,
  parseEnv,
} from "../src/env.ts";
import { OPENAI_COMPATIBLE_HEADERS_ENV_KEY } from "../src/constants.ts";
```

Append this `describe` block at the end of the file:

```ts
describe("OPENAI_COMPATIBLE_HEADERS registration", () => {
  test("is a managed env key", () => {
    expect(MANAGED_ENV_KEYS).toContain(OPENAI_COMPATIBLE_HEADERS_ENV_KEY);
  });

  test("appears in credential diagnostics and debug key lists", () => {
    expect(CREDENTIAL_DIAGNOSTIC_ENV_KEYS).toContain(
      OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
    );
    expect(DEBUG_ENV_KEYS).toContain(OPENAI_COMPATIBLE_HEADERS_ENV_KEY);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm test test/env.test.ts`
Expected: FAIL — `MANAGED_ENV_KEYS` does not contain the new key (and the imports for the list exports resolve, since they already exist).

- [ ] **Step 3: Register the key in MANAGED_ENV_KEYS**

In `src/env.ts`, add the import to the existing `./constants.js` import block:

```ts
  OPENAI_COMPATIBLE_BASE_URL_ENV_KEY,
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
```

Add the key to `MANAGED_ENV_KEYS`, right after `OPENAI_COMPATIBLE_BASE_URL_ENV_KEY`:

```ts
  OPENAI_COMPATIBLE_API_KEY_ENV_KEY,
  OPENAI_COMPATIBLE_BASE_URL_ENV_KEY,
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  ANTHROPIC_API_KEY_ENV_KEY,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm test test/env.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/env.ts test/env.test.ts
git commit -m "feat: register OPENAI_COMPATIBLE_HEADERS as a managed env key"
```

---

## Task 3: Headers-specific diagnostic warning (avoid false quote warning)

**Files:**

- Modify: `src/env.ts`
- Test: `test/env.test.ts`

Context: `getCredentialWarnings` flags any value containing a quote character. A valid headers JSON always contains quotes, so it must use a JSON-validity warning instead. The diagnostics preview must stay **masked** (do NOT add the key to `isNonSecretDiagnosticKey`).

- [ ] **Step 1: Write the failing test**

Append to `test/env.test.ts`. This tests the pure warning helper. First add `getHeadersWarnings` to the env import block created in Task 2:

```ts
import {
  CREDENTIAL_DIAGNOSTIC_ENV_KEYS,
  DEBUG_ENV_KEYS,
  formatEnv,
  getHeadersWarnings,
  MANAGED_ENV_KEYS,
  parseEnv,
} from "../src/env.ts";
```

Then append:

```ts
describe("getHeadersWarnings", () => {
  test("returns no warnings for a valid headers JSON object", () => {
    expect(getHeadersWarnings('{"X-Api-Key":"abc"}')).toEqual([]);
  });

  test("warns for invalid JSON", () => {
    expect(getHeadersWarnings("{not json}")).toEqual(["invalid headers JSON"]);
  });

  test("warns when the value is not an object of strings", () => {
    expect(getHeadersWarnings('["a","b"]')).toEqual(["invalid headers JSON"]);
    expect(getHeadersWarnings('{"X-Count":1}')).toEqual([
      "invalid headers JSON",
    ]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test test/env.test.ts`
Expected: FAIL — `getHeadersWarnings` is not exported.

- [ ] **Step 3: Implement and wire the warning helper**

In `src/env.ts`, add `OPENAI_COMPATIBLE_HEADERS_ENV_KEY` and `isValidHeadersRecord` to the `./constants.js` import block (the key was added in Task 2; add the validator too):

```ts
  isValidHeadersRecord,
  ...
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
```

Add the exported helper near `getProviderWarnings` (after line 253):

```ts
export function getHeadersWarnings(value: string): string[] {
  try {
    const parsed: unknown = JSON.parse(value);

    return isValidHeadersRecord(parsed) ? [] : ["invalid headers JSON"];
  } catch {
    return ["invalid headers JSON"];
  }
}
```

Wire it into `createCredentialDiagnostic`. Replace the `warnings` ternary (currently lines 180-185):

```ts
    warnings:
      key === OPENWIKI_MODEL_ID_ENV_KEY
        ? getModelWarnings(value)
        : key === OPENWIKI_PROVIDER_ENV_KEY
          ? getProviderWarnings(value)
          : getCredentialWarnings(value),
```

with:

```ts
    warnings:
      key === OPENWIKI_MODEL_ID_ENV_KEY
        ? getModelWarnings(value)
        : key === OPENWIKI_PROVIDER_ENV_KEY
          ? getProviderWarnings(value)
          : key === OPENAI_COMPATIBLE_HEADERS_ENV_KEY
            ? getHeadersWarnings(value)
            : getCredentialWarnings(value),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm test test/env.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck**

Run: `pnpm typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/env.ts test/env.test.ts
git commit -m "feat: use JSON-validity warning for OPENAI_COMPATIBLE_HEADERS diagnostics"
```

---

## Task 4: Inject headers into the ChatOpenAI client + fail-fast

**Files:**

- Modify: `src/agent/index.ts`

Note: `src/agent/index.ts` has no dedicated unit test (its logic is integration-level). This task is verified via typecheck, the full test suite, and manual reasoning. Keep changes minimal and mirror the existing `baseURL` handling.

- [ ] **Step 1: Import the resolver**

In `src/agent/index.ts`, add to the `../constants.js` import block:

```ts
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  resolveProviderHeaders,
```

(Place `OPENAI_COMPATIBLE_HEADERS_ENV_KEY` and `resolveProviderHeaders` in alphabetical position within that block, alongside `OPENAI_COMPATIBLE_BASE_URL_ENV_KEY` and `resolveProviderBaseUrl`.)

- [ ] **Step 2: Inject defaultHeaders in createModel**

Replace the final `ChatOpenAI` branch of `createModel` (currently lines 432-443):

```ts
const baseURL = resolveProviderBaseUrl(provider);

return new ChatOpenAI({
  apiKey: process.env[getProviderApiKeyEnvKey(provider)],
  configuration: baseURL
    ? {
        baseURL,
      }
    : undefined,
  model: modelId,
});
```

with:

```ts
const baseURL = resolveProviderBaseUrl(provider);
const defaultHeaders = resolveProviderHeaders(provider);
const configuration =
  baseURL || defaultHeaders
    ? {
        ...(baseURL ? { baseURL } : {}),
        ...(defaultHeaders ? { defaultHeaders } : {}),
      }
    : undefined;

return new ChatOpenAI({
  apiKey: process.env[getProviderApiKeyEnvKey(provider)],
  configuration,
  model: modelId,
});
```

- [ ] **Step 3: Fail fast + debug in runOpenWikiAgent**

In `runOpenWikiAgent`, after the `ensureProviderBaseUrl(provider);` call (currently line 92), add a resolve-and-debug block that fails fast on malformed input and logs header names only:

```ts
ensureProviderBaseUrl(provider);
const providerHeaders = resolveProviderHeaders(provider);
if (providerHeaders) {
  emitDebug(
    options,
    `provider.headers=${JSON.stringify(Object.keys(providerHeaders))}`,
  );
}
```

`resolveProviderHeaders` throws on malformed input, so this runs before any model creation and surfaces the error through the normal run-failure path.

- [ ] **Step 4: Mask the headers key in formatDebugValue**

In `formatDebugValue` (around lines 1297-1325), add a branch so the headers key never prints a value preview. After the `_API_KEY` branch (line 1310-1312), add:

```ts
if (key === OPENAI_COMPATIBLE_HEADERS_ENV_KEY) {
  return `set(length=${value.length})`;
}
```

- [ ] **Step 5: Typecheck**

Run: `pnpm typecheck`
Expected: no errors. (In particular, confirm `ChatOpenAI`'s `configuration.defaultHeaders` accepts `Record<string, string>`. `@langchain/openai` passes `configuration` through to the OpenAI client's `ClientOptions`, which accepts `defaultHeaders`.)

- [ ] **Step 6: Run the full test suite**

Run: `pnpm test`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add src/agent/index.ts
git commit -m "feat: inject OPENAI_COMPATIBLE_HEADERS into openai-compatible requests"
```

---

## Task 5: Documentation

**Files:**

- Modify: `README.md`
- Modify: `openwiki/operations/credentials-and-updates.md`
- Modify: `openwiki/cli/usage.md`
- Modify: `openwiki/agent/workflow.md`

- [ ] **Step 1: README**

In `README.md`, in the openai-compatible env block (currently lines 98-99):

```
OPENAI_COMPATIBLE_API_KEY=your-gateway-key
OPENAI_COMPATIBLE_BASE_URL=https://your-gateway.example.com/v1
```

add a third line and a short note beneath the block:

```
OPENAI_COMPATIBLE_HEADERS={"X-Api-Key":"abc123","X-Org":"acme"}
```

Note text to add after the code block:

```
`OPENAI_COMPATIBLE_HEADERS` is optional. When set, it must be a JSON object of
string values; those headers are sent on every request to the gateway (useful
for gateways that require extra auth or routing headers). Invalid JSON causes
the run to fail with a clear error.
```

- [ ] **Step 2: credentials-and-updates.md**

In `openwiki/operations/credentials-and-updates.md`, extend the "Base URLs" bullet (line 22) by adding a new bullet after it:

```
- Optional headers: `OPENAI_COMPATIBLE_HEADERS` (optional — a JSON object of string values injected as request headers for the openai-compatible provider; malformed values fail the run)
```

And update the diagnostics coverage sentence (line 60) to mention the headers key is covered and masked:

```
Diagnostics cover all six provider keys plus `OPENWIKI_PROVIDER`, `OPENWIKI_MODEL_ID`, the base URLs (`ANTHROPIC_BASE_URL`, `OPENAI_COMPATIBLE_BASE_URL`), the optional `OPENAI_COMPATIBLE_HEADERS` (shown masked, since it may carry secrets), and `LANGSMITH_API_KEY`. This makes startup problems easier to diagnose without exposing secret values (non-secret values such as the provider, model ID, and base URLs are shown in full).
```

- [ ] **Step 3: cli/usage.md**

In `openwiki/cli/usage.md`, update the openai-compatible table row (line 66) to mention the optional headers key, and add a sentence to the openai-compatible prose section (near line 82-94). After the `OPENAI_COMPATIBLE_BASE_URL` example block, add:

```
Optionally set `OPENAI_COMPATIBLE_HEADERS` to a JSON object of string values to
send extra headers on every request (for gateways that need custom auth or
routing headers), e.g. `OPENAI_COMPATIBLE_HEADERS={"X-Api-Key":"abc"}`.
```

- [ ] **Step 4: agent/workflow.md**

In `openwiki/agent/workflow.md`, extend the openai-compatible model-creation bullet (line 28) to note that `configuration.defaultHeaders` is populated from `OPENAI_COMPATIBLE_HEADERS` when set. Append to that bullet:

```
When `OPENAI_COMPATIBLE_HEADERS` is set (a JSON object of string values), those headers are passed through as `configuration.defaultHeaders` on the ChatOpenAI client.
```

- [ ] **Step 5: Verify docs formatting**

Run: `pnpm format:check`
Expected: PASS (or run `pnpm format` then re-check).

- [ ] **Step 6: Commit**

```bash
git add README.md openwiki/operations/credentials-and-updates.md openwiki/cli/usage.md openwiki/agent/workflow.md
git commit -m "docs: document OPENAI_COMPATIBLE_HEADERS env var"
```

---

## Final Verification

- [ ] **Step 1: Full test suite**

Run: `pnpm test`
Expected: PASS.

- [ ] **Step 2: Typecheck + lint**

Run: `pnpm typecheck && pnpm lint:check`
Expected: no errors.

- [ ] **Step 3: Format check**

Run: `pnpm format:check`
Expected: PASS.
