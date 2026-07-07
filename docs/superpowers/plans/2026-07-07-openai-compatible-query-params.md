# OpenAI-compatible Custom Query Params Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `OPENAI_COMPATIBLE_QUERY` env var (a raw query string) that appends query params to every request for the `openai-compatible` provider.

**Architecture:** Extend the existing generic provider-config pattern (mirroring `headersEnvKey`) with a `queryEnvKey` and a `resolveProviderQuery` resolver in `src/constants.ts`. The resolved params flow into the shared `ChatOpenAI` client as `configuration.defaultQuery` in `src/agent/index.ts`. The key is registered in `MANAGED_ENV_KEYS` (`src/env.ts`) and shown in full (non-secret) in diagnostics/debug.

**Tech Stack:** TypeScript (ESM, `.js` import specifiers), Vitest, `@langchain/openai` `ChatOpenAI`, `URLSearchParams`.

---

## File Structure

- `src/constants.ts` — new env-key constant, `queryEnvKey` config field, `getProviderQueryEnvKey`, `resolveProviderQuery`.
- `src/agent/index.ts` — inject `defaultQuery` in `createModel`; resolve + debug in `runOpenWikiAgent`; full-value display in `formatDebugValue`.
- `src/env.ts` — register key in `MANAGED_ENV_KEYS`; mark non-secret in `isNonSecretDiagnosticKey`.
- `test/constants.test.ts` — resolver tests.
- `test/env.test.ts` — managed-key membership test.
- `README.md`, `openwiki/operations/credentials-and-updates.md`, `openwiki/cli/usage.md`, `openwiki/agent/workflow.md` — docs.

Conventions: local imports use `.js` specifiers; tests import from `../src/*.ts`. Run tests with `pnpm test`, typecheck with `pnpm typecheck`, build with `pnpm build`.

---

## Task 1: Constant + config field + resolver

**Files:**
- Modify: `src/constants.ts`
- Test: `test/constants.test.ts`

- [ ] **Step 1: Write the failing tests**

Extend the import block at the top of `test/constants.test.ts` to add the three new symbols (keep existing ones):

```ts
import {
  DEFAULT_MODEL_ID,
  DEFAULT_PROVIDER,
  getDefaultModelId,
  getProviderHeadersEnvKey,
  getProviderQueryEnvKey,
  isValidBaseUrl,
  isValidModelId,
  isValidProvider,
  normalizeModelId,
  normalizeProvider,
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  OPENAI_COMPATIBLE_QUERY_ENV_KEY,
  resolveConfiguredProvider,
  resolveProviderBaseUrl,
  resolveProviderHeaders,
  resolveProviderQuery,
} from "../src/constants.ts";
```

Append these `describe` blocks at the end of the file:

```ts
describe("getProviderQueryEnvKey", () => {
  test("returns the query key for openai-compatible", () => {
    expect(getProviderQueryEnvKey("openai-compatible")).toBe(
      OPENAI_COMPATIBLE_QUERY_ENV_KEY,
    );
  });

  test("returns undefined for providers without a query key", () => {
    expect(getProviderQueryEnvKey("openrouter")).toBeUndefined();
    expect(getProviderQueryEnvKey("anthropic")).toBeUndefined();
  });
});

describe("resolveProviderQuery", () => {
  test("returns undefined when the query env var is unset", () => {
    expect(resolveProviderQuery("openai-compatible", {})).toBeUndefined();
  });

  test("returns undefined when the value is whitespace only", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "   ",
      }),
    ).toBeUndefined();
  });

  test("returns undefined for providers without a query key", () => {
    expect(
      resolveProviderQuery("openrouter", {
        OPENAI_COMPATIBLE_QUERY: "a=1",
      }),
    ).toBeUndefined();
  });

  test("parses a basic query string", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "api-version=2024-06-01&region=us",
      }),
    ).toEqual({ "api-version": "2024-06-01", region: "us" });
  });

  test("strips a single leading question mark", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "?a=1",
      }),
    ).toEqual({ a: "1" });
  });

  test("percent-decodes values", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "q=a%20b",
      }),
    ).toEqual({ q: "a b" });
  });

  test("duplicate keys resolve last-wins", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "a=1&a=2",
      }),
    ).toEqual({ a: "2" });
  });

  test("returns undefined when the value parses to zero params", () => {
    expect(
      resolveProviderQuery("openai-compatible", {
        OPENAI_COMPATIBLE_QUERY: "?",
      }),
    ).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm test test/constants.test.ts`
Expected: FAIL — `getProviderQueryEnvKey`, `resolveProviderQuery`, `OPENAI_COMPATIBLE_QUERY_ENV_KEY` not exported.

- [ ] **Step 3: Add the constant and config field**

In `src/constants.ts`, add the env-key constant right after `OPENAI_COMPATIBLE_HEADERS_ENV_KEY`:

```ts
export const OPENAI_COMPATIBLE_QUERY_ENV_KEY = "OPENAI_COMPATIBLE_QUERY";
```

Add a `queryEnvKey` field to the `ProviderConfig` type, right after the `headersEnvKey` block:

```ts
  /**
   * Environment variable that, when set, supplies extra query-string parameters
   * (a raw query string) appended to every request for this provider.
   */
  queryEnvKey?: string;
```

Set it on the `openai-compatible` config (after `headersEnvKey`):

```ts
  "openai-compatible": {
    apiKeyEnvKey: OPENAI_COMPATIBLE_API_KEY_ENV_KEY,
    baseUrlEnvKey: OPENAI_COMPATIBLE_BASE_URL_ENV_KEY,
    headersEnvKey: OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
    queryEnvKey: OPENAI_COMPATIBLE_QUERY_ENV_KEY,
    requiresBaseUrl: true,
    label: "OpenAI-compatible",
    modelOptions: [],
  },
```

- [ ] **Step 4: Add the getter and resolver**

In `src/constants.ts`, add after `resolveProviderHeaders` / `isValidHeadersRecord`:

```ts
export function getProviderQueryEnvKey(
  provider: OpenWikiProvider,
): string | undefined {
  return getProviderConfig(provider).queryEnvKey;
}

/**
 * Resolves custom query-string parameters for a provider from its configured
 * env var. Returns `undefined` when the provider has no query key, the value is
 * unset/blank, or it parses to zero params. Duplicate keys resolve last-wins.
 */
export function resolveProviderQuery(
  provider: OpenWikiProvider,
  env: NodeJS.ProcessEnv = process.env,
): Record<string, string> | undefined {
  const queryEnvKey = getProviderConfig(provider).queryEnvKey;

  if (!queryEnvKey) {
    return undefined;
  }

  const raw = env[queryEnvKey]?.trim();

  if (!raw) {
    return undefined;
  }

  const normalized = raw.startsWith("?") ? raw.slice(1) : raw;
  const params: Record<string, string> = {};

  for (const [key, value] of new URLSearchParams(normalized)) {
    params[key] = value;
  }

  return Object.keys(params).length > 0 ? params : undefined;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm test test/constants.test.ts`
Expected: PASS.

- [ ] **Step 6: Typecheck**

Run: `pnpm typecheck`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/constants.ts test/constants.test.ts
git commit -m "feat: add OPENAI_COMPATIBLE_QUERY resolver"
```

---

## Task 2: Register key in env management + mark non-secret

**Files:**
- Modify: `src/env.ts`
- Test: `test/env.test.ts`

- [ ] **Step 1: Write the failing test**

In `test/env.test.ts`, extend the constants import (which currently imports `OPENAI_COMPATIBLE_HEADERS_ENV_KEY`) to also import the query key:

```ts
import {
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  OPENAI_COMPATIBLE_QUERY_ENV_KEY,
} from "../src/constants.ts";
```

Append this `describe` block at the end of the file:

```ts
describe("OPENAI_COMPATIBLE_QUERY registration", () => {
  test("is a managed env key", () => {
    expect(MANAGED_ENV_KEYS).toContain(OPENAI_COMPATIBLE_QUERY_ENV_KEY);
  });

  test("appears in credential diagnostics and debug key lists", () => {
    expect(CREDENTIAL_DIAGNOSTIC_ENV_KEYS).toContain(
      OPENAI_COMPATIBLE_QUERY_ENV_KEY,
    );
    expect(DEBUG_ENV_KEYS).toContain(OPENAI_COMPATIBLE_QUERY_ENV_KEY);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm test test/env.test.ts`
Expected: FAIL — `MANAGED_ENV_KEYS` does not contain the new key.

- [ ] **Step 3: Register the key and mark non-secret**

In `src/env.ts`, add the import to the existing `./constants.js` import block (next to `OPENAI_COMPATIBLE_HEADERS_ENV_KEY`):

```ts
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  OPENAI_COMPATIBLE_QUERY_ENV_KEY,
```

Add the key to `MANAGED_ENV_KEYS`, right after `OPENAI_COMPATIBLE_HEADERS_ENV_KEY`:

```ts
  OPENAI_COMPATIBLE_API_KEY_ENV_KEY,
  OPENAI_COMPATIBLE_BASE_URL_ENV_KEY,
  OPENAI_COMPATIBLE_HEADERS_ENV_KEY,
  OPENAI_COMPATIBLE_QUERY_ENV_KEY,
  ANTHROPIC_API_KEY_ENV_KEY,
```

Mark it non-secret so diagnostics show the full value. In `isNonSecretDiagnosticKey`, add the new key:

```ts
function isNonSecretDiagnosticKey(key: string): boolean {
  return (
    key === OPENWIKI_MODEL_ID_ENV_KEY ||
    key === OPENWIKI_PROVIDER_ENV_KEY ||
    key === ANTHROPIC_BASE_URL_ENV_KEY ||
    key === OPENAI_COMPATIBLE_BASE_URL_ENV_KEY ||
    key === OPENAI_COMPATIBLE_QUERY_ENV_KEY
  );
}
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
git commit -m "feat: register OPENAI_COMPATIBLE_QUERY as a non-secret managed env key"
```

---

## Task 3: Inject query into the ChatOpenAI client + debug

**Files:**
- Modify: `src/agent/index.ts`

Note: `src/agent/index.ts` has no dedicated unit test; verify via typecheck and the full suite.

- [ ] **Step 1: Import the resolver and key**

In `src/agent/index.ts`, add to the `../constants.js` import block (alphabetical position, next to `OPENAI_COMPATIBLE_HEADERS_ENV_KEY` / `resolveProviderHeaders`):

```ts
  OPENAI_COMPATIBLE_QUERY_ENV_KEY,
  ...
  resolveProviderQuery,
```

- [ ] **Step 2: Inject defaultQuery in createModel**

Replace the current `ChatOpenAI` branch of `createModel`:

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

with:

```ts
  const baseURL = resolveProviderBaseUrl(provider);
  const defaultHeaders = resolveProviderHeaders(provider);
  const defaultQuery = resolveProviderQuery(provider);
  const configuration =
    baseURL || defaultHeaders || defaultQuery
      ? {
          ...(baseURL ? { baseURL } : {}),
          ...(defaultHeaders ? { defaultHeaders } : {}),
          ...(defaultQuery ? { defaultQuery } : {}),
        }
      : undefined;

  return new ChatOpenAI({
    apiKey: process.env[getProviderApiKeyEnvKey(provider)],
    configuration,
    model: modelId,
  });
```

- [ ] **Step 3: Debug line in runOpenWikiAgent**

In `runOpenWikiAgent`, right after the existing `providerHeaders` debug block:

```ts
  const providerHeaders = resolveProviderHeaders(provider);
  if (providerHeaders) {
    emitDebug(
      options,
      `provider.headers=${JSON.stringify(Object.keys(providerHeaders))}`,
    );
  }
```

add:

```ts
  const providerQuery = resolveProviderQuery(provider);
  if (providerQuery) {
    emitDebug(options, `provider.query=${JSON.stringify(providerQuery)}`);
  }
```

- [ ] **Step 4: Full-value display in formatDebugValue**

In `formatDebugValue`, extend the model/provider branch so the query key also prints its full value:

```ts
  if (
    key === OPENWIKI_MODEL_ID_ENV_KEY ||
    key === OPENWIKI_PROVIDER_ENV_KEY ||
    key === OPENAI_COMPATIBLE_QUERY_ENV_KEY
  ) {
    return `set(value=${JSON.stringify(value)})`;
  }
```

(This replaces the existing two-key condition; keep everything else in the function unchanged. `OPENAI_COMPATIBLE_QUERY_ENV_KEY` is imported in Step 1.)

- [ ] **Step 5: Typecheck**

Run: `pnpm typecheck`
Expected: no errors. (`configuration.defaultQuery` accepts `Record<string,string>` — verified in `openai@6.44.0` `client.d.ts`: `defaultQuery?: Record<string, string | undefined>`.)

- [ ] **Step 6: Run the full test suite**

Run: `pnpm test`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add src/agent/index.ts
git commit -m "feat: inject OPENAI_COMPATIBLE_QUERY into openai-compatible requests"
```

---

## Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `openwiki/operations/credentials-and-updates.md`
- Modify: `openwiki/cli/usage.md`
- Modify: `openwiki/agent/workflow.md`

- [ ] **Step 1: README**

In `README.md`, in the openai-compatible env block, add a line after the
`OPENAI_COMPATIBLE_HEADERS` line:

```
OPENAI_COMPATIBLE_QUERY=api-version=2024-06-01&region=us
```

And add this note after the existing `OPENAI_COMPATIBLE_HEADERS` note paragraph:

```
`OPENAI_COMPATIBLE_QUERY` is optional. When set, it is a raw query string (e.g.
`api-version=2024-06-01&region=us`) whose parameters are appended to every
request to the gateway. Duplicate keys use the last value.
```

- [ ] **Step 2: credentials-and-updates.md**

In `openwiki/operations/credentials-and-updates.md`, add a bullet after the
`OPENAI_COMPATIBLE_HEADERS` "Optional headers" bullet:

```
- Optional query params: `OPENAI_COMPATIBLE_QUERY` (optional — a raw query string appended to every request for the openai-compatible provider; duplicate keys use the last value)
```

Update the diagnostics coverage sentence to also mention the query key is shown
in full. Replace the phrase listing the optional headers:

```
the optional `OPENAI_COMPATIBLE_HEADERS` (shown masked, since it may carry secrets), the optional `OPENAI_COMPATIBLE_QUERY` (shown in full), and `LANGSMITH_API_KEY`.
```

(i.e. insert the `OPENAI_COMPATIBLE_QUERY` clause before `and \`LANGSMITH_API_KEY\``.)

- [ ] **Step 3: cli/usage.md**

In `openwiki/cli/usage.md`, update the openai-compatible table row's Models cell
to also mention the optional query key:

```
| openai-compatible | `OPENAI_COMPATIBLE_API_KEY` | `OPENAI_COMPATIBLE_BASE_URL` (required) | custom model ID only (optional `OPENAI_COMPATIBLE_HEADERS`, `OPENAI_COMPATIBLE_QUERY`) |
```

After the existing `OPENAI_COMPATIBLE_HEADERS` prose paragraph, add:

```
Optionally set `OPENAI_COMPATIBLE_QUERY` to a raw query string (e.g.
`OPENAI_COMPATIBLE_QUERY=api-version=2024-06-01&region=us`) to append query
parameters to every request. It is passed through as
`configuration.defaultQuery` on the ChatOpenAI client; duplicate keys use the
last value.
```

- [ ] **Step 4: agent/workflow.md**

In `openwiki/agent/workflow.md`, update the openai-compatible model-creation
bullet to include `defaultQuery?` in the shown signature and append a sentence:

Change `configuration: { baseURL?, defaultHeaders? }` to
`configuration: { baseURL?, defaultHeaders?, defaultQuery? }`, then append:

```
When `OPENAI_COMPATIBLE_QUERY` is set (a raw query string), its parsed parameters are passed through as `configuration.defaultQuery` and appended to every request; `resolveProviderQuery()` parses it with `URLSearchParams` (duplicate keys last-wins).
```

- [ ] **Step 5: Verify formatting**

Run: `pnpm format:check`
Expected: PASS (or run `pnpm exec prettier --write` on the changed files, then re-check).

- [ ] **Step 6: Commit**

```bash
git add README.md openwiki/operations/credentials-and-updates.md openwiki/cli/usage.md openwiki/agent/workflow.md
git commit -m "docs: document OPENAI_COMPATIBLE_QUERY env var"
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

- [ ] **Step 4: Rebuild the global binary**

Run: `pnpm build`
Expected: build succeeds; `postbuild` keeps `dist/cli.js` executable.
