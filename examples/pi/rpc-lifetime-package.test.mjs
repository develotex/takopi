import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = process.env.SUBAGENTS_ROOT;
assert.ok(root, 'SUBAGENTS_ROOT must select a package copy');
const load = file => import(pathToFileURL(path.join(root, file)));
const { drainOutstandingWork } = await load('src/runs/background/auto-drain.js');

// Removing signal forwarding from the production drain makes this fail.
test('auto-drain forwards actual operation cancellation to its wait', async () => {
  const controller = new AbortController();
  const state = { currentSessionId: 'owned' };
  let captured;
  let work = true;
  await assert.rejects(drainOutstandingWork({
    state, signal: controller.signal, hasWork: () => work,
    wait: async (_args, signal) => {
      captured = signal;
      work = false;
      controller.abort();
      return { content: [], details: {} };
    },
  }), { name: 'AbortError' });
  assert.equal(captured, controller.signal);
});

test('auto-drain rejects abort before entering a wait', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(drainOutstandingWork({
    state: { currentSessionId: 'owned' }, signal: controller.signal,
    hasWork: () => false,
  }), /abort/i);
});

test('no pending work remains a quiet no-op', async () => {
  await drainOutstandingWork({ state: { currentSessionId: 'owned' }, hasWork: () => false });
});

test('timeout, wait failure, changed session and supervisor yield stay explicit', async () => {
  const state = { currentSessionId: 'owned' };
  let now = 0;
  await assert.rejects(drainOutstandingWork({ state, timeoutMs: 1, now: () => now++, hasWork: () => true }), /timed out/);
  await assert.rejects(drainOutstandingWork({ state, hasWork: () => true, wait: async () => ({ isError: true, content: [{ type: 'text', text: 'needs attention' }] }) }), /needs attention/);
  await assert.rejects(drainOutstandingWork({ state, hasWork: () => true, wait: async () => { state.currentSessionId = 'wrong'; return { details: {}, content: [] }; } }), /session changed/);
  state.currentSessionId = 'owned';
  await drainOutstandingWork({ state, hasPendingSupervisorRequest: () => true, hasWork: () => { throw Error('must yield for supervisor'); } });
});
