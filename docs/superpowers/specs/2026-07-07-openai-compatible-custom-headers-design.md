# OpenAI-compatible custom POST headers — design

## Summary

Add a new, optional environment variable that lets users attach custom HTTP
headers to requests sent by the `openai-compatible` provider. Some
OpenAI-compatible gateways (proxies, corporate gateways, LiteLLM, etc.) require
extra headers beyond `Authorization` — for example an API-key header, an org or
project identifier, or routing metadata. Today OpenWiki has no way to supply
these, so those gateways cannot be used.

## Goals

- Allow arbitrary custom request headers for the `openai-compatible` provider.
- Keep the change minimal and consistent with existing provider-config patterns
  (mirrors how `baseUrlEnvKey` is handled).
- Fail fast with a clear error when the value is malformed.
- Treat the value as a secret in diagnostics and debug output (headers can carry
  tokens).

## Non-goals

- No support for the other ChatOpenAI-based providers (`openai`, `baseten`,
  `fireworks`) or for `anthropic` / `openrouter`. Scope is `openai-compatible`
  only.
- No interactive setup-wizard step. The variable is manual/advanced only.

## Env variable

- **Name:** `OPENAI_COMPATIBLE_HEADERS`
- **Format:** a JSON object whose keys are header names and whose values are
  strings, e.g.:

  ```
  OPENAI_COMPATIBLE_HEADERS={"X-Api-Key":"abc123","X-Org":"acme"}
  ```

- **Empty/unset:** no custom headers are added (current behavior).
- **Validation:** must parse as JSON and be a flat object whose every value is a
  string. Anything else (invalid JSON, array, nested object, non-string value)
  is an error.

## Behavior

- When set and valid, the parsed headers are injected into the `ChatOpenAI`
  client as `configuration.defaultHeaders`, so they are sent on every request to
  the configured base URL.
- When set but invalid, the run fails fast **before** any model work, with an
  error message that names `OPENAI_COMPATIBLE_HEADERS` and states the expected
  format.
- Debug output lists header **names only** — never values.

## Implementation

### `src/constants.ts`

- Add `OPENAI_COMPATIBLE_HEADERS_ENV_KEY = "OPENAI_COMPATIBLE_HEADERS"`.
- Add optional `headersEnvKey?: string` to the `ProviderConfig` type and set it
  on the `openai-compatible` config.
- Add `getProviderHeadersEnvKey(provider)` returning the configured key (or
  `undefined`).
- Add `resolveProviderHeaders(provider, env = process.env)`:
  - Returns `undefined` when the provider has no `headersEnvKey` or the value is
    unset/empty (after trim).
  - Otherwise parses the JSON and validates it via `isValidHeadersRecord`.
  - Throws an `Error` naming the env key when parsing fails or validation fails.
  - Returns `Record<string, string>` on success.
- Add `isValidHeadersRecord(value): value is Record<string, string>` — true when
  `value` is a non-null, non-array object and every own value is a string.

### `src/agent/index.ts`

- Import `resolveProviderHeaders` (and, if needed for messages,
  `getProviderHeadersEnvKey`).
- In `runOpenWikiAgent`, after `ensureProviderBaseUrl(provider)`, resolve headers
  once so malformed input fails fast. Emit a debug line with header **names**
  only when present.
- In `createModel`, in the shared `ChatOpenAI` branch, add
  `defaultHeaders` to `configuration` when headers resolve. The `configuration`
  object should be built so that `baseURL` and `defaultHeaders` are included only
  when present (avoid passing `configuration: undefined` semantics changes when
  only headers are set and no base URL — but `openai-compatible` always requires
  a base URL, so both will typically be present).

### `src/env.ts`

- Add `OPENAI_COMPATIBLE_HEADERS_ENV_KEY` to `MANAGED_ENV_KEYS` (placed next to
  the other `OPENAI_COMPATIBLE_*` keys). This automatically:
  - loads it from `~/.openwiki/.env`,
  - includes it in `CREDENTIAL_DIAGNOSTIC_ENV_KEYS` (it is a credential, so it
    is NOT added to `NON_CREDENTIAL_ENV_KEYS`),
  - includes it in `DEBUG_ENV_KEYS`,
  - preserves its ordering in `formatEnv`.
- In `createCredentialDiagnostic`, branch for the headers key to use a new
  `getHeadersWarnings(value)` (invalid-JSON warning) instead of
  `getCredentialWarnings`, because the generic warnings would always flag the
  JSON quote characters.
- Keep the diagnostics preview **masked** (do NOT add the key to
  `isNonSecretDiagnosticKey`).
- In `src/agent/index.ts` `formatDebugValue`, add a branch so the headers key
  reports `set(length=...)` only (no value preview), since it can contain tokens.

### Tests

- `test/constants.test.ts`:
  - `resolveProviderHeaders` returns `undefined` when unset/empty.
  - returns parsed record for valid JSON object of strings.
  - throws for invalid JSON, for a JSON array, for a nested object, and for a
    non-string value.
  - `getProviderHeadersEnvKey` returns the key for `openai-compatible` and
    `undefined` for a provider without it.
- `test/env.test.ts`:
  - `MANAGED_ENV_KEYS` / `CREDENTIAL_DIAGNOSTIC_ENV_KEYS` / `DEBUG_ENV_KEYS`
    include `OPENAI_COMPATIBLE_HEADERS`.
  - diagnostics preview for the headers key is masked (not the raw value).
  - invalid-JSON header value produces the invalid-JSON warning.

### Docs

- `README.md`: document `OPENAI_COMPATIBLE_HEADERS` alongside the other
  `openai-compatible` variables.
- `openwiki/operations/credentials-and-updates.md`: add the key to the env
  reference.
- `openwiki/cli/usage.md`: mention it in the `openai-compatible` provider notes.

## Error handling

- Malformed value → thrown `Error` with a message like:
  `OPENAI_COMPATIBLE_HEADERS must be a JSON object of string header values.`
- The error surfaces through the normal run failure path, before the model is
  created.

## Risks / edge cases

- **Secret leakage:** mitigated by masking in diagnostics and logging header
  names only in debug.
- **Overriding managed headers:** custom headers could override
  `Authorization`/`Content-Type`. This is intentional (advanced users may need
  it); documented but not blocked.
- **Whitespace-only value:** treated as unset.
