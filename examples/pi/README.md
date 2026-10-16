# RPC session naming for Pi

`session-title-rpc.ts` is an optional, standalone Pi extension based on the
personal JSON-mode `session-title.ts` pattern. Takopi does not install or enable
it automatically. Copy **only** this `.ts` file into `~/.pi/agent/extensions/`,
or load it explicitly with `pi --mode rpc --extension /path/session-title-rpc.ts`.
A new Pi process is needed to load a newly installed extension; do not restart
an active Takopi task merely to load it.

The original JSON-only extension may remain installed: the new one runs only in
RPC mode. Do not load a second RPC namer. It generates one name for a fresh,
unnamed session, never renames a session with user history (including recovered
sessions), and never overwrites a name set while generation was running.
Existing unnamed topics are not backfilled.

## Model and data

The default is `openrouter/qwen/qwen3.6-flash` (provider `openrouter`, model ID
`qwen/qwen3.6-flash`). It uses Pi's model registry and credentials; configure
OpenRouter in Pi or provide `OPENROUTER_API_KEY`. No credentials belong in the
extension. At verification time the public OpenRouter models API advertised
$0.1875 / million input tokens and $1.125 / million output tokens for short
contexts, versus Google's $0.25 / $1.50 for Gemini 3.1 Flash-Lite. Both Qwen rates
are 25% lower. Prices and routed-provider availability may change. Gemini 3.5
Flash-Lite is newer but more expensive ($0.30 / $2.50).

Sources: <https://openrouter.ai/api/v1/models>,
<https://ai.google.dev/gemini-api/docs/pricing>.

The naming call sends **only the first user prompt**, capped at 2,000 UTF-16 code
units after removing the known leading Takopi live-guidance paragraph, plus
short naming instructions. It does not send tools, history or the main system
prompt. Switching from Google to OpenRouter changes the recipient of that text.
The naming model does not change the main conversation's model.

Naming has a 10-second deadline and is canceled on session shutdown or switch.
Errors and malformed output are nonfatal: the existing topic title remains.
Only a fixed, non-sensitive warning is emitted, not prompts, credentials or raw
provider errors. Takopi currently discards extension UI notifications; do not
mistake this warning for new Telegram error reporting.

Pi publishes the name with `session_info_changed`. Takopi's shared Pi event
translator converts that to `TitleChangedEvent`; the existing topic callback
updates the Telegram title only where topic management is allowed and the title
is not bound to a project/branch. Other topic restrictions are unchanged.

## Noninteractive extension dialogs

Takopi's RPC client responds to `select`, `confirm`, `input` and `editor`
requests with `extension_ui_response` / `cancelled: true`. Nothing is forwarded
to Telegram and no action is automatically approved. Confirmations resolve to
`false`, other dialogs to `undefined`, as specified by Pi. This also works for
dialogs raised before the prompt has been accepted, so the task cannot deadlock
waiting for the UI response. Fire-and-forget notifications/status/widget updates
get no response. This policy applies to all Takopi-owned RPC clients, not Pi's
interactive terminal mode. An extension is still responsible for respecting a
negative confirmation result.

## Verification

With Node.js 22.22+:

```sh
node --experimental-strip-types --test examples/pi/session-title-rpc.test.mjs
```

Takopi transport, lifecycle and topic callback tests:

```sh
uv run pytest --no-cov tests/test_pi_rpc_dialogs.py tests/test_pi_rpc.py tests/test_telegram_loop_title_change.py
just check
```

The Node tests replace only the external registry/provider and extension host
boundary. They exercise the actual extension handlers and inspect emitted names.
For a runtime smoke test use a **new isolated session file** with no tools, no
other extensions and an explicit inexpensive main model. Observe
`session_info_changed`, a successful main response, and no second rename on the
next prompt. A mocked Telegram bot is not proof of production Telegram delivery.
