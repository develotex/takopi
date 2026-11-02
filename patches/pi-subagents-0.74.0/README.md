# Opt-in Takopi RPC lifetime compatibility

For **Pi 0.99.2 + pi-subagents 0.74.0 only**. This is a maintained package-copy
integration, not an installed fix, Pi upgrade, or general-purpose patch manager.
Takopi's normal installations do not require or enable it.

## Why this seam

RPC supplies dialogs (`ctx.hasUI === true`) but Takopi closes the process after
a completed run. Upstream subagents drains only `!ctx.hasUI`, so an empty **or
nonempty** parent stop can settle with a live workflow. Teardown then cancels it.

The opt-in `--takopi-rpc-lifetime` flag enables draining in the existing awaited
`agent_end`. In Pi 0.99.2 that handler still has the actual operation abort
signal. `agent_before_settle` does not; moving the wait there strands abort.
The existing headless branch and ordinary TUI registration remain intact.
Runtime-default changes below apply to **all users of this separate package
copy**, including TUI; the RPC lifetime hook alone remains opt-in. The original
npm installation is unchanged.

The integration uses existing package-owned session/run predicates. Auto-drain
waits in finite interruptible windows (at most 30 minutes each), re-arming until
owned work finishes; it has **no implicit total lifetime deadline**. A caller's
explicit auto-drain `timeoutMs` still bounds the complete drain and fails on expiry.
It then drains native watcher/coalescer/completion batches
within 30 seconds. Native queued messages cause parent synthesis through Pi's
ordinary post-run continuation; no replay prompt, fake owner, or second notifier
is introduced. New work admitted by synthesis is drained on its own boundary.

A pending native supervisor ask returns only after its exact request ID has a
successful same-runtime `pi.sendMessage` enqueue receipt. Pi applies those
runtime actions **after** the handler returns: waiting on
`ctx.hasPendingMessages()` here would deadlock. After a successful supervisor
reply, only that request's resolved run/agent can wait for actual stale
`contact_supervisor` attention to clear (maximum five seconds). Other/unknown
attention still fails; new pending asks retain the normal supervisor barrier.

Draining and delivery are abort-aware. The child-runtime and non-UI main
`agent_end` callers forward their actual operation signal too, including signals
already aborted before the drain starts. A captured real RPC operation signal also
prevents late completion notifications from starting a parent turn after abort;
results can still be appended. A fresh explicit operation resets that gate.
The same optional predicate gates native supervisor, control/steering,
wait-subscription, incremental workflow/failure, compaction and watchdog wakes.
Suppressed supervisor asks are **not** marked seen or delivered: their exact
request files remain for normal polling after a fresh explicit operation. Armed
wait-subscription files are likewise retained. Other notices may append context
without starting a turn. Supervisor reply receipts are expired on insertion,
capped at 256 entries and cleared on session start/shutdown.

The prefixed `TAKOPI_RPC_LIFETIME:` error is deliberately terminal in Takopi's RPC
consumer, because Pi otherwise reports a handler error and can still settle.
Only machine code `TAKOPI_RPC_LIFETIME:CANCELLED:` **and** a known local
`interrupt()`/cancellation request follow normal user-cancellation rendering;
the host still consumes settlement before accepting a fresh prompt. Unexpected
cancellation codes, child/drain failures and ordinary errors remain terminal.
Unrelated extension/naming errors retain their existing policy.

## Runtime deadlines versus waiting windows

Omitted run `timeoutMs`/`maxRuntimeMs` means no runtime deadline in foreground,
async single, chains, tasks, scripted workflows and their nested children. No
Infinity, oversized timer, or larger replacement deadline is injected. Explicit
call aliases, agent `timeoutMs`, supported config defaults and explicit inherited
parent deadlines retain upstream precedence/enforcement; async composites do
not acquire a parent deadline from config (existing upstream behavior).
The tool schema describes this contract. `bg_wait`'s documented 30-minute default
is only a waiting window: work continues when it expires.

Idle/long-tool `needs_attention` is not terminal failure: auto-drain waits through
it with the real waiter. Native pending supervisor requests still yield to their
queued parent wake; unknown supervisor attention and real failed runs retain
strict upstream error behavior. Cancellation and ownership checks remain active
on every window and poll; no errors are silently swallowed.

## Reproducible preparation (operator action, not automatically installed)

Run from the Takopi checkout, with a **new**, separate output directory:

```sh
python3 scripts/prepare_pi_rpc_lifetime.py \
  --source /home/victor/.pi/agent/npm/node_modules/pi-subagents \
  --destination /mnt/victor_vector_volume/pi-subagents-rpc-0.74.0 \
  --pi-package /home/victor/.local/share/pi-node/node-v22.22.3-linux-x64/lib/node_modules/@earendil-works/pi-coding-agent

python3 scripts/prepare_pi_rpc_lifetime.py --verify \
  --destination /mnt/victor_vector_volume/pi-subagents-rpc-0.74.0 \
  --pi-package /home/victor/.local/share/pi-node/node-v22.22.3-linux-x64/lib/node_modules/@earendil-works/pi-coding-agent
```

All touched source files and the package manifest have exact SHA-256 gates.
Preparation validates everything first, patches a staging copy with context
fuzz disabled, verifies output hashes, and only then renames it into place.
The unified patch includes explicit missing-final-newline records, so source-map
EOFs and subsequent file headers cannot be concatenated. A repeated invocation verifies an
existing destination rather than changing it. Mismatch never overwrites either
package. `patch` and Python 3 are required. The copy links its `node_modules` to
the original package's dependency root; **retain that tree and the pinned Pi
installation**. Nothing runs npm, mutates original files, or rewrites settings.

For an approved deployment, retain the original release/package for rollback.
Exclude only the original npm package's **extension**, not its skills/prompts:
change its existing Pi package declaration to the object form with
`"extensions": []` (preserve its existing source and other resource filters).
Load exactly one patched index through the **global Pi `settings.extensions`**
list, preserving all existing extension entries:

```json
{"extensions": ["/mnt/victor_vector_volume/pi-subagents-rpc-0.74.0/index.js"]}
```

This is an illustrative entry to merge, not a replacement settings file. Loading
it globally preserves subagent availability in ordinary interactive Pi after the
original npm extension is filtered out. The lifetime behavior is opt-in and
remains disabled for TUI and default RPC hosts. Add **only the flag** to Takopi's
existing runner arguments; do not load a second index there:

```toml
extra_args = ["--takopi-rpc-lifetime"]
```

Merge that flag with existing options; do not discard naming/extensions or user
intents. The package's unchanged `index.js` child guard skips main extension
registration when `PI_SUBAGENT_CHILD=1`; runtime-owned child hooks remain separate. The custom flag is the startup capability proof: when the
patched extension is missing or fails to load, Pi rejects the unknown option,
rather than silently operating without protection. Verify the copy and runtime
versions before each deployment; do not upgrade packages behind the copied
runtime. Changes to settings/installed packages/service lifecycle remain an
operator/coordinator action after testing and independent reviews, never during
another accepted task or child run. PR8's startup dialog fail-closed gate remains
in place; no dialog is implicitly approved.

## Validation

```sh
SUBAGENTS_ROOT=/mnt/victor_vector_volume/pi-subagents-rpc-0.74.0 \
PI_PACKAGE_ROOT=/home/victor/.local/share/pi-node/node-v22.22.3-linux-x64/lib/node_modules/@earendil-works/pi-coding-agent \
node --test examples/pi/*.test.mjs
uv run pytest
```

`LIFETIME_SCRATCH_ROOT` selects an existing scratch directory (local default
`/mnt/victor_vector_volume`; CI uses its runner temp). The real lifecycle tests
launch the actual pinned Pi CLI, actual patched subagents executor, actual async
workflow and actual child with a deterministic **no-network** provider. They
verify empty and text stops, result synthesis before settlement, supervisor
ask/reply/resume/synthesis, abort ACK under two seconds, late completion beyond
batch delay, and a working subsequent explicit prompt. The late-supervisor regression
explicitly acknowledges `clear_queue` before `abort`, observes no autonomous
restart, then explicitly continues, answers the retained ask and synthesizes.
Focused tests check control/steering gates and retained wait-subscription files.
Tests bound their own harness/ask deadlines and reap only their unique
scratch-path runners. The default child launches intentionally omit deadlines;
a separate real async single test enforces an explicit 100ms deadline.
Caller-level tests also register the real child and main extension hooks with
isolated session state and active background providers, covering already-aborted
and mid-wait cancellation. Launch regressions validate agent child deadlines
through the Node timer maximum and exercise the actual workflow-root branch's
alias agreement and positive-integer checks before discovery/runner admission.

To reproduce the old regression, select an untouched original package and set
`LIFETIME_EXPECT_RED=1`, running only the `empty parent` test. It fails missing
synthesis after settlement. This switch deliberately does not pass the custom
flag. Helper tests also cover cancellation forwarding, no-work no-op, deadline,
failed wait, changed owner, strict attention, bounded delivery and resolved-reply
correlation. Python tests cover prefix-scoped host errors and all-or-nothing,
idempotent copy preparation. CI runs the pinned copy preparation and full Node
suite, in addition to normal Python checks.

The unbounded regression runs four **virtual hours**, across eight re-armed
windows, through both a controlled waiter and the real async status/wait path.
It checks idle attention without spin, foreign-session preservation, exactly one
drain completion, foreground/async/composite default resolution, chain JSON
serialization, nested explicit parent deadlines and explicit drain expiry.
Existing real RPC cases remain short integration proofs, **not** evidence that a
real production child has survived hours: the prior two short passing cases did
not prove long lifetime.

## Limits

- This is not a durable detached-work service or Telegram end-to-end proof.
  Topic/session isolation continues to use package ownership and Takopi's
  existing exclusive session/path claims; no topic routing was changed.
- The inventoried package-owned wake paths share cancellation gating. Normal
  Takopi abort/close cleanup is still required for process/resource lifetime;
  unrelated third-party extensions can own independent wake behavior outside
  this pinned package integration.
- Genuine empty stops with no owned work retain Takopi's existing empty-final
  presentation; provider errors and unrelated naming errors are not rewritten.
- Fourteen existing compiled package JS files plus one new helper are patched. Their
  source maps/declarations remain upstream artifacts; this pinned compatibility
  copy is executed as JavaScript, not published as an upstream TypeScript build.
- This removes implicit **Pi child run** deadlines, not every finite I/O guard.
  HTTP header/body idle limits (default 5m), startup/control handshakes, acceptance
  verification (default 2m), worktree setup hooks (30s), and completion delivery
  (30s) remain bounded. Fast builtin read/grep/find/ls/edit/write/structured_output
  operations retain a 5m per-tool guard that can terminate a child on an actual
  stalled tool; `bash`, custom tools, `bg_wait` and supervisor tools do not receive
  that implicit guard. Explicit per-tool limits remain authoritative.
- External runner contracts are not made unlimited: the Herdr Codex monitor
  defaults to a 90s placement wait capped at 30m; external-job bridge operations
  retain their 2m request timeout. These transports/adapters need separate
  validation before claiming unbounded external-runner behavior. Native supervisor
  requests retain their upstream 10m unanswered-request timeout; this is an ask
  deadline, not a child runtime deadline. Watchdog post-terminal tail/permission
  delivery guards also remain; idle activity thresholds alone do not kill a run.
- Unbounded work can consume resources indefinitely until cancellation. This is
  intentional; operators who need a finite deadline must configure one explicitly.
- Upstream changes, alternate Pi versions, and actual Telegram ingress/deployment
  require separate validation. No live installation is claimed by these tests.
