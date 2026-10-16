# Contextual Pi conversation (2026-10-15)

With `transports.telegram.pi_live_conversation = true`, ordinary messages in an
active Pi topic go to that topic's existing Pi RPC owner. Questions, corrections,
and messages containing both are preserved in full. `/update <text>` remains an
optional compatibility alias, not a requirement for understanding natural language.

The old question-mark/keyword classifier and phase/counts-only answer path are no
longer used. Understanding and answering belong to the agent that has the actual
conversation, task context, tools and results. No second model session is launched
for a status question. Cache investigation is out of scope.

## User-visible behavior

- Receipt persistence, RPC acknowledgement and observed delivery are silent.
- Main Pi answers the actual question or explains the applied correction. Prompt
  guidance explicitly forbids replacing an answer with receipt counts, tool phases,
  generic acknowledgements or promises to investigate.
- A main-agent consideration marker is processed only after positive observation
  of the matching user message in the same Pi session. Its answer is sent as a reply
  to the original Telegram message, without the technical marker/status label.
- The original task retains one initial progress/control message. Per-tool progress
  edits and repeated live-intake/startup notices are suppressed. Existing explicit
  `progress_updates = "none"` remains respected.
- Actual transport/RPC uncertainty and recovery errors remain visible: silence must
  not imply successful submission or invite unsafe duplicate tool effects.
- Pi may need to finish a current tool call before answering. This is not a separate
  concurrent read-only agent and does not promise an answer before a safe checkpoint.
  It does not abort tools or background children to answer a question.

## Output durability

Substantive receipt answers are persisted before sending. A service-level dispatcher
retries them even after the task owner has closed, and after process restart.
Successful transport return must include a message reference; a `None` response is
not delivery. The dispatcher does not put a short timeout around an outbox waiter:
cancelling that waiter would not cancel the already queued Telegram send.

Answers are split into <=2000-codepoint chunks (<=4000 UTF-16 units). Each confirmed
chunk advances a durable offset, so a later chunk failure does not resend earlier
confirmed chunks. A failed answer blocks later queued answers in its session, not
successful retries in other sessions. An ambiguous external send or crash between
send and persistence can still duplicate a chunk; this is not exactly-once delivery.

A completed main-task result must not disappear when a later question is consumed.
The RPC adapter therefore retains completed, non-tool assistant turns before
`agent_settled`, not only the last turn. Earlier results and settlement-turn results
are persisted and sent as replies to the original task message, including the final
result. Verified reconciliation consumes current-run assistant events in order;
completed task results and marked answers use the same ordered durable dispatcher.
A normal final cannot bypass a delayed earlier output, and a correction cannot be
followed by an older result presented as the latest state. Independently completed
outputs remain stored even if a later receipt cannot settle. Outputs are not cached
and relabelled as the final state after a correction. The ordinary bridge suppresses
only output already handed to durable delivery. A genuinely empty final without
such an answer keeps the existing failure behavior. Actionable error/recovery
notices retain their existing independent path.

Old inbox records decode with no pending reply; historical terminal receipts are not
reannounced. A fresh owner can still settle an unresolved historical receipt from
exact-session evidence: the matching user message must precede its assistant
acknowledgement. This does not replay old standalone outputs. Mixed assistant turns
are split around known markers so preceding text, the answer and following text
retain their order, not just the order between separate turns. Inbox/output compaction, cross-process session ownership and exactly-once
external delivery are not implemented by this change. Terminal records and completed
outputs currently remain in the JSON state file.

## Verification boundaries

Automated checks cover startup/active/settlement, natural and mixed messages,
durable intake while RPC/Telegram is busy, receipt and standalone-output retry,
partial chunks, restart, topic isolation and the existing media/forward/cancel matrix.
A real installed Pi RPC probe with tools/extensions disabled additionally checks
inference context and active steer with a synthetic task, conflicting correction
(10 → 20), and a subsequent question remembering the corrected value; its Telegram
callback is in-memory. This is not a production Telegram end-to-end or background-tool test.
Deployment/restart is a separate operation, not implied by opening the PR.

Successful durable publication can leave an intentionally empty final. The production
executor wraps runners to hide resume lines and to prepend unsupported-option
warnings; both wrappers must forward the live runner's dynamic
`suppress_empty_final` property. Otherwise a correctly sent answer is followed by a
false `error` card. Executor-level tests cover visible/hidden resume, warning on/off,
progress none/once and published/absent/failed terminal answers. Actual empty or
failed answers are still errors; the wrappers do not suppress them.
