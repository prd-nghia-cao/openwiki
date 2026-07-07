# OpenAI-compatible custom query params — design

## Summary

Add a new, optional environment variable that appends custom query-string
parameters to every request sent by the `openai-compatible` provider. Some
OpenAI-compatible gateways require request-level query params — for example an
`api-version`, a `region`, or a routing selector. Today OpenWiki has no way to
supply these, so those gateways cannot be used.

This mirrors the existing `OPENAI_COMPATIBLE_HEADERS` feature, but for query
params, using the OpenAI SDK's `configuration.defaultQuery`
(`Record<string, string | undefined>`, applied to every request — verified in
`openai@6.44.0` `client.d.ts`).

## Goals

- Allow arbitrary custom query params for the `openai-compatible` provider.
- Keep the change minimal and consistent with the `headersEnvKey` /
  `baseUrlEnvKey` provider-config pattern.
- Accept a raw query-string value (the format the user asked for).

## Non-goals

- No support for the other providers (`openai`, `baseten`, `fireworks`,
  `anthropic`, `openrouter`). Scope is `openai-compatible` only.
- No interactive setup-wizard step. The variable is manual/advanced only.

## Env variable

- **Name:** `OPENAI_COMPATIBLE_QUERY`
- **Format:** a raw query string, e.g.:

  ```
  OPENAI_COMPATIBLE_QUERY=api-version=2024-06-01&region=us
  ```

  A single leading `?` is tolerated and stripped
  (`?api-version=2024-06-01` works too).

- **Empty/unset:** no query params are added (current behavior).
- **Parsing:** parsed with `URLSearchParams`, which is lenient and performs
  standard percent-decoding. Duplicate keys resolve to **last value wins**.

## Behavior

- When set and non-empty, the parsed params are injected into the `ChatOpenAI`
  client as `configuration.defaultQuery`, so they are appended to every request
  to the configured base URL.
- When unset, blank (after trim), or it parses to zero params, no query params
  are added.
- `URLSearchParams` never throws, so there is no fail-fast error path (this is
  the one difference from `OPENAI_COMPATIBLE_HEADERS`, whose JSON can be
  malformed).
- The value is treated as **non-secret**: shown in full in diagnostics and
  debug output, and logged in full in the debug run line.

## Implementation

### `src/constants.ts`

- Add `OPENAI_COMPATIBLE_QUERY_ENV_KEY = "OPENAI_COMPATIBLE_QUERY"`.
- Add optional `queryEnvKey?: string` to the `ProviderConfig` type and set it on
  the `openai-compatible` config.
- Add `getProviderQueryEnvKey(provider)` returning the configured key (or
  `undefined`).
- Add `resolveProviderQuery(provider, env = process.env)`:
  - Returns `undefined` when the provider has no `queryEnvKey` or the value is
    unset/blank (after trim).
  - Strips a single leading `?`.
  - Parses with `new URLSearchParams(raw)` and builds a `Record<string, string>`
    via iteration (last value wins for duplicate keys).
  - Returns `undefined` if the resulting record has no entries.
  - Returns `Record<string, string>` on success.

### `src/agent/index.ts`

- Import `resolveProviderQuery` and `OPENAI_COMPATIBLE_QUERY_ENV_KEY`.
- In `runOpenWikiAgent`, after resolving headers, resolve the query params and
  emit a debug line showing them in full (non-secret) when present.
- In `createModel`, in the shared `ChatOpenAI` branch, add `defaultQuery` to the
  `configuration` object when it resolves (built so each of `baseURL`,
  `defaultHeaders`, `defaultQuery` is included only when present; `configuration`
  stays `undefined` when none are present).
- In `formatDebugValue`, add a branch so the query key prints its full value
  (like `OPENWIKI_MODEL_ID` / `OPENWIKI_PROVIDER`), not a masked preview.

### `src/env.ts`

- Add `OPENAI_COMPATIBLE_QUERY_ENV_KEY` to `MANAGED_ENV_KEYS`, placed next to the
  other `OPENAI_COMPATIBLE_*` keys. This automatically:
  - loads it from `~/.openwiki/.env`,
  - includes it in `CREDENTIAL_DIAGNOSTIC_ENV_KEYS` (it is not added to
    `NON_CREDENTIAL_ENV_KEYS`),
  - includes it in `DEBUG_ENV_KEYS`,
  - preserves its ordering in `formatEnv`.
- Add `OPENAI_COMPATIBLE_QUERY_ENV_KEY` to `isNonSecretDiagnosticKey` so the
  diagnostics preview shows the full value (JSON-stringified). Warnings use the
  default `getCredentialWarnings` (same as base URLs).

### Tests

- `test/constants.test.ts`:
  - `resolveProviderQuery` returns `undefined` when unset/blank.
  - parses `"api-version=2024-06-01&region=us"` to
    `{ "api-version": "2024-06-01", region: "us" }`.
  - strips a leading `?` (`"?a=1"` → `{ a: "1" }`).
  - percent-decodes values (`"q=a%20b"` → `{ q: "a b" }`).
  - duplicate keys resolve last-wins (`"a=1&a=2"` → `{ a: "2" }`).
  - returns `undefined` for a provider without a query key.
  - `getProviderQueryEnvKey` returns the key for `openai-compatible` and
    `undefined` otherwise.
- `test/env.test.ts`:
  - `MANAGED_ENV_KEYS` / `CREDENTIAL_DIAGNOSTIC_ENV_KEYS` / `DEBUG_ENV_KEYS`
    include `OPENAI_COMPATIBLE_QUERY`.

### Docs

- `README.md`: document `OPENAI_COMPATIBLE_QUERY` alongside the other
  `openai-compatible` variables.
- `openwiki/operations/credentials-and-updates.md`: add the key to the env
  reference and the diagnostics coverage note.
- `openwiki/cli/usage.md`: mention it in the `openai-compatible` provider notes
  and table row.
- `openwiki/agent/workflow.md`: note that `configuration.defaultQuery` is
  populated from `OPENAI_COMPATIBLE_QUERY`.

## Risks / edge cases

- **Duplicate keys:** only the last value is kept (documented). Multi-value query
  params are not supported (YAGNI).
- **Overriding built-in query params:** custom params could collide with params
  the SDK already sets; this is intentional for advanced gateways.
- **Whitespace-only value:** treated as unset.
- **Value with no `=`** (e.g. `flag`): `URLSearchParams` yields `{ flag: "" }`,
  which is passed through as an empty-value param — acceptable.
