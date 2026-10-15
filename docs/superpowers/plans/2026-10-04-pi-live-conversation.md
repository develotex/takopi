# Pi Live Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The user can get an immediate, truthful response in a busy Pi Telegram topic and submit durable task updates without aborting the main run.

**Architecture:** Keep the existing per-session FIFO for non-Pi and non-live routes. Implement a per-session Pi RPC owner with correlation/event streaming and a separate read-only quick responder, plus a durable per-topic inbox. Main session delivery uses queued steering; quick answers never enter the main session.

**Tech Stack:** Python 3.12+, anyio, msgspec/pydantic, Telegram transport, installed Pi RPC JSONL, pytest/ruff/ty.

**Spec:** `docs/superpowers/specs/2026-10-04-pi-live-conversation-design.md`

## Global Constraints

- Only `develotex/takopi`, base `origin/local-dev`, branch `victor/pi-live-conversation`; no changes to Pi core.
- Do not cancel an existing run or child just to process a question/update. Separate quick responder must have no write-capable tools.
- Existing runner/other engines and commands retain their routing. A queued RPC command does not imply delivery/consideration.
- Avoid concurrent session file/worktree writers; owner-controlled lifecycle, bounded timeouts, persistent recovery.
- `just check` before code commits; Russian PR description with Manual testing section; two independent reviews before PR.

## Review Focus

- A user sends a mixed question and constraint while the main tool call is active: reply promptly and deliver only the original constraint after the tool finishes (Task 3).
- Telegram retries a message after Pi accepts a steer, or Takopi restarts: do not claim duplicate-free consideration without evidence (Tasks 2, 3).
- A quick responder in topic A must not see transcript/resume of topic B or rebind A to its own session (Task 3).
- Pi emits `agent_end` then retries or receives a follow-up: do not close early (Task 1).
- `/cancel` or `/new` while a quick reply runs: cancel only explicitly addressed active work and preserve message association (Task 3).

---

### Task 1: Pi RPC transport and per-session lifecycle

**Files:** Create `src/takopi/runners/pi_rpc.py`, `tests/test_pi_rpc.py`; modify `src/takopi/runners/pi.py` and its tests only where needed.

**Interfaces:** `PiRpcClient(session_path: Path, cwd: Path, args: list[str])`; `start()`, `request(type: str, **fields: object) -> dict`, `events() -> AsyncIterator[dict]`, `close()`. `PiRpcRun` exposes `run(prompt, resume) -> AsyncIterator[TakopiEvent]`, `steer(text) -> disposition`, `interrupt() -> bool`, and `session_key`. Existing `PiRunner` remains the non-live fallback. Preserve `RunOptions` including model/provider/thinking.

- [ ] Write failing fake-process tests: command IDs matched out of order, LF framing (including U+2028), read stdout continuously, stderr drain, failure/exit, timeout and cleanup; agent_end followed by retry then agent_settled.
- [ ] Run `uv run pytest tests/test_pi_rpc.py -q` to confirm failures.
- [ ] Implement minimal RPC transport and event translator; emit Started with control on actual active run, Completed only at agent_settled; use Pi get_state for stable session ID/file (don't promote a filename to an 8-char ambiguous ID).
- [ ] Run focused tests; real Pi RPC smoke with a throwaway session and a blocked mock tool; verify steer accepted while tool active, delivered afterward, no abort.
- [ ] Commit implementation and tests after `just check`.

### Task 2: Durable per-scope inbox and delivery receipts

**Files:** Create `src/takopi/telegram/live_inbox.py`, `tests/test_telegram_live_inbox.py`; follow existing atomic-json state-store conventions.

**Interfaces:** `LiveInbox(path: Path)` with `receive(chat_id, thread_id, message_id, session_key, text) -> Receipt`, `mark_submitted`, `mark_delivered`, `mark_considered`, `pending(session_key) -> list[Receipt]`; per-record stable ID `(chat_id,thread_id,message_id)` and states `received/submitted/delivered/considered/deferred`. No auto-confirm on a successful RPC response.

- [ ] Write failing tests: duplicate Telegram delivery, FIFO, restart at each lifecycle state, race with settlement, uncertain delivery, malformed/truncated store, cross-topic isolation.
- [ ] Run `uv run pytest tests/test_telegram_live_inbox.py -q` to confirm failures.
- [ ] Implement atomic persistence under configured Takopi state directory; recovery reconciles main-session user entries (stable receipt marker) before retry, never automatically asserts consideration.
- [ ] Run focused tests; run `just check`, commit.

### Task 3: Telegram live route, quick responder and user-visible acknowledgement

**Files:** Create `src/takopi/telegram/live_conversation.py`, `tests/test_telegram_live_conversation.py`; modify `src/takopi/telegram/loop.py`, `src/takopi/runner_bridge.py`, `src/takopi/settings.py`, Pi runner integration, and relevant docs. Keep loop modifications limited to dispatch/ownership, not a second state machine in loop.py.

**Interfaces:** `LiveConversationService` receives the resolved `IncomingMessage`, exact topic/session key, active RPC owner, and Telegram transport. It stores updates before ACK, routes pure questions to an isolated no-tools responder with bounded snapshot and replies in the same topic; sends updates with `steer` while streaming or serialized `prompt` after settlement. Explicit `/update` (or equivalent documented command) bypasses ambiguous classification. The main Pi agent must explicitly acknowledge considered/deferred; surface distinct receipt states without pretending queued means read.

- [ ] Write failing integration tests for all Review Focus conditions, normal idle messages, topic/session reply overrides, status latency during foreground tool, detached subagent, status+constraint, fallback if classifier uncertain, quick-responder timeout, cancel/new, and non-Pi runner regression.
- [ ] Run `uv run pytest tests/test_telegram_live_conversation.py -q` to confirm failures.
- [ ] Implement opt-in settings, active owner routing, bounded quick responder, receipt progress/ack and safe checkpoint delivery; no two writable sessions on same path and no sidecar resume overwrite.
- [ ] Run focused tests and relevant existing queue/runner/topic suites; manual local smoke against installed Pi (no production bot restart).
- [ ] Update user-facing docs, run `just check`, commit.

### Task 4: Whole-branch verification and PR

- [ ] Check `git diff origin/local-dev`, branch/status and edge cases yourself; verify a question receives an answer during a long main task and a new constraint is later explicitly considered without interrupting it.
- [ ] Run `just check` and document smoke evidence; do not call it passing if tool unavailable or test fails.
- [ ] Fresh Viktor read-only review and separate Devin read-only review of implementation against spec and regressions. Assess each finding from code, fix valid findings, rerun tests and review high-risk changes.
- [ ] Push only `victor/pi-live-conversation` to `develotex/takopi`; PR base `local-dev`; Russian title/body ordered причина → продуктовое решение → техническая реализация, explicit limits and Manual testing checklist.
