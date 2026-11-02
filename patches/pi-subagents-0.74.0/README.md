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
The unchanged headless branch and ordinary TUI behavior remain unchanged.

The integration uses existing package-owned session/run predicates and strict
30-minute auto-drain. It then drains native watcher/coalescer/completion batches
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

Draining and delivery are abort-aware. A captured real RPC operation signal also
prevents late completion notifications from starting a parent turn after abort;
results can still be appended. A fresh explicit operation resets that gate.
The prefixed `TAKOPI_RPC_LIFETIME:` error is deliberately terminal in Takopi's RPC
consumer, because Pi otherwise reports a handler error and can still settle.
Unrelated extension/naming errors retain their existing policy.

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
Preparation validates everything first, patches a staging copy, verifies output
hashes, and only then renames it into place. A repeated invocation verifies an
existing destination rather than changing it. Mismatch never overwrites either
package. `patch` and Python 3 are required. The copy links its `node_modules` to
the original package's dependency root; **retain that tree and the pinned Pi
installation**. Nothing runs npm, mutates original files, or rewrites settings.

For an approved deployment, retain the original release/package for rollback.
Exclude only the original npm package's **extension**, not its skills/prompts:
change its existing Pi package declaration to the object form with
`"extensions": []` (preserve its existing source and other resource filters).
Load exactly one patched index in the Takopi Pi runner's existing `extra_args`:

```toml
extra_args = ["--extension", "/mnt/victor_vector_volume/pi-subagents-rpc-0.74.0/index.js", "--takopi-rpc-lifetime"]
```

Merge those arguments with existing options; do not discard naming/extensions
or user intents. The custom flag is the startup capability proof: when the
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
batch delay, and a working subsequent explicit prompt. Tests bound their own
child/ask deadlines and reap only their unique scratch-path runners.

To reproduce the old regression, select an untouched original package and set
`LIFETIME_EXPECT_RED=1`, running only the `empty parent` test. It fails missing
synthesis after settlement. This switch deliberately does not pass the custom
flag. Helper tests also cover cancellation forwarding, no-work no-op, deadline,
failed wait, changed owner, strict attention, bounded delivery and resolved-reply
correlation. Python tests cover prefix-scoped host errors and all-or-nothing,
idempotent copy preparation. CI runs the pinned copy preparation and full Node
suite, in addition to normal Python checks.

## Limits

- This is not a durable detached-work service or Telegram end-to-end proof.
  Topic/session isolation continues to use package ownership and Takopi's
  existing exclusive session/path claims; no topic routing was changed.
- The late-completion gate covers native completion notifier wakes. Native
  supervisor/control channels have independent wake paths. Explicit cancellation
  in Takopi must still follow its normal abort/close cleanup; this patch does not
  promise a permanently retained aborted RPC host suppresses every other channel.
- Genuine empty stops with no owned work retain Takopi's existing empty-final
  presentation; provider errors and unrelated naming errors are not rewritten.
- Six existing compiled package JS files plus one new helper are patched. Their
  source maps/declarations remain upstream artifacts; this pinned compatibility
  copy is executed as JavaScript, not published as an upstream TypeScript build.
- Upstream changes, alternate Pi versions, and actual Telegram ingress/deployment
  require separate validation. No live installation is claimed by these tests.
