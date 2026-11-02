import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, access } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const load = file => import(pathToFileURL(path.join(process.env.SUBAGENTS_ROOT, file)));
const { handleSubagentControlNotice } = await load('src/extension/control-notices.js');
const { handleSubagentSteeringNotice } = await load('src/extension/steering-notices.js');
const { createWaitSubscriptionManager } = await load('src/runs/background/wait-subscriptions.js');

for (const allowed of [false, true]) {
  test(`control and steering notices respect optional wake predicate=${allowed}`, () => {
    const sent = [];
    const pi = { sendMessage: (message, options) => sent.push({ message, options }) };
    handleSubagentControlNotice({ pi, canTriggerTurn: () => allowed, visibleControlNotices: new Set(), details: { source: 'async', noticeText: 'local attention', event: { type: 'needs_attention', runId: 'owned', agent: 'probe', reason: 'idle' } } });
    handleSubagentSteeringNotice({ pi, canTriggerTurn: () => allowed, state: { currentSessionId: 'owner' }, details: { currentSessionId: 'owner', runId: 'owned', state: 'failed', message: 'local steering', requestId: 'request' } });
    assert.equal(sent.length, 2);
    assert.ok(sent.every(item => item.options.triggerTurn === allowed));
  });
}

test('cancelled wait subscription retains its exact armed file until explicit fresh operation', async () => {
  const dir = await mkdtemp(path.join(process.env.LIFETIME_SCRATCH_ROOT ?? '/mnt/victor_vector_volume', 'rpc-wait-proof-'));
  const sent = [];
  const pi = { events: { on: () => () => {} }, sendMessage: (message, options) => sent.push({ message, options }) };
  const state = { currentSessionId: 'owner' };
  let allowed = false, now = 0;
  const manager = createWaitSubscriptionManager(pi, state, { subscriptionsDir: dir, now: () => now, canTriggerTurn: () => allowed });
  try {
    const record = manager.arm({ targetKind: 'async', runId: 'owned', requestedId: 'owned', timeoutMs: 10 });
    now = 20;
    manager.reconcile();
    assert.equal(sent.length, 0);
    assert.ok(state.waitSubscriptions.has(record.token));
    await access(path.join(dir, `${record.token}.json`));
    allowed = true;
    manager.reconcile();
    assert.equal(sent.length, 1);
    assert.equal(sent[0].options.triggerTurn, true);
    assert.equal(sent[0].message.details.token, record.token);
    assert.equal(state.waitSubscriptions.size, 0);
  } finally { manager.dispose(); await rm(dir, { recursive: true, force: true }); }
});
