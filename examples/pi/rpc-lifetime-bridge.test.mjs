import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const root = process.env.SUBAGENTS_ROOT;
const { drainRpcLifetime, rpcAttentionPredicate } = await import(pathToFileURL(path.join(root, 'src/extension/takopi-rpc-lifetime.js')));

function setup(overrides = {}) {
  const controller = new AbortController();
  let flushes = 0;
  const deps = {
    ctx: { signal: controller.signal, hasPendingMessages: () => false },
    state: { currentSessionId: 'owner' },
    supervisorChannel: { hasPendingRequests: () => false },
    completionNotifier: { flushPendingDelivery: () => flushes++, hasPendingDelivery: () => false },
    resultWatcher: { refreshResultDelivery() {}, hasPendingDelivery: () => false },
    drain: async () => {},
    ...overrides,
  };
  return { deps, controller, flushes: () => flushes };
}

test('flush only after successful owned drain', async () => {
  const good = setup();
  await drainRpcLifetime(good.deps);
  assert.equal(good.flushes(), 1);
  const bad = setup({ drain: async () => { throw Error('child failed'); } });
  await assert.rejects(drainRpcLifetime(bad.deps), /TAKOPI_RPC_LIFETIME: child failed/);
  assert.equal(bad.flushes(), 0);
});

test('cancel during drain never flushes or triggers continuation', async () => {
  const s = setup();
  s.deps.drain = async () => { s.controller.abort(); };
  await assert.rejects(drainRpcLifetime(s.deps), /TAKOPI_RPC_LIFETIME:.*abort/i);
  assert.equal(s.flushes(), 0);
});

test('supervisor must already have a native queued parent wake', async () => {
  const s = setup({ deliveryTimeoutMs: 20, supervisorChannel: { hasPendingRequests: () => true, hasQueuedSupervisorWake: () => false } });
  await assert.rejects(drainRpcLifetime(s.deps), /supervisor.*wake/i);
  s.deps.supervisorChannel.hasQueuedSupervisorWake = () => true;
  await drainRpcLifetime(s.deps);
});

test('resolved reply only defers its exact run/agent stale supervisor status, bounded beyond 500ms', () => {
  let now = 1000;
  const replies = new Map([['child', { replyTo: 'ask', runId: 'child', agent: 'probe', deadline: 6000 }]]);
  const channel = { hasResolvedReply: receipt => receipt.replyTo === 'ask' };
  const attention = rpcAttentionPredicate(replies, channel, () => now);
  const run = { id: 'child', activityState: 'needs_attention', currentTool: 'contact_supervisor', steps: [{ agent: 'probe', activityState: 'needs_attention', currentTool: 'contact_supervisor' }] };
  now = 1800;
  assert.equal(attention(run), false);
  assert.equal(attention({ ...run, id: 'wrong' }), true);
  assert.equal(attention({ ...run, steps: [{ agent: 'other', activityState: 'needs_attention', currentTool: 'contact_supervisor' }] }), true);
  assert.equal(attention({ ...run, currentTool: 'bash' }), true);
  channel.hasResolvedReply = () => false; // new/unknown ask cannot use old receipt
  assert.equal(attention(run), true);
  channel.hasResolvedReply = () => true;
  now = 6001;
  assert.equal(attention(run), true);
});

test('delivery wait is bounded and owner-isolated', async () => {
  const s = setup({ deliveryTimeoutMs: 20, resultWatcher: { refreshResultDelivery() {}, hasPendingDelivery: () => true } });
  await assert.rejects(drainRpcLifetime(s.deps), /delivery.*timed out/i);
  s.deps.resultWatcher = { refreshResultDelivery() { s.deps.state.currentSessionId = 'other'; }, hasPendingDelivery: () => false };
  await assert.rejects(drainRpcLifetime(s.deps), /session changed/i);
});
