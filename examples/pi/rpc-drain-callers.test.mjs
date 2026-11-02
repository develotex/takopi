// Exercise actual registered agent_end callers, not a direct drain helper.
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks, createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
const root = process.env.SUBAGENTS_ROOT, piRoot = process.env.PI_PACKAGE_ROOT;
assert.ok(root && piRoot);
const scratch = mkdtempSync(path.join(process.env.LIFETIME_SCRATCH_ROOT ?? '/mnt/victor_vector_volume', 'rpc-drain-callers-'));
process.env.TMPDIR = scratch;
process.env.PI_CODING_AGENT_DIR = path.join(scratch, 'agent');
process.env.PI_SUBAGENT_CHILD = '0';
mkdirSync(path.join(scratch, 'agent/extensions/subagent'), { recursive: true });
writeFileSync(path.join(scratch, 'agent/extensions/subagent/config.json'), JSON.stringify({ toolActivation: 'eager', waitTool: { enabled: false }, fleetView: false, asyncWidget: false }));
const peers = createRequire(path.join(piRoot, 'package.json'));
registerHooks({ resolve(specifier, context, next) {
  try { return next(specifier, context); }
  catch (error) {
    if (specifier === '@earendil-works/pi-coding-agent') return next(pathToFileURL(path.join(piRoot, 'dist/index.js')).href, context);
    if (/^@earendil-works\/pi-(agent-core|ai|tui)$/.test(specifier)) return next(pathToFileURL(path.join(piRoot, 'node_modules', specifier, 'dist/index.js')).href, context);
    if (!specifier.startsWith('.') && !specifier.startsWith('/')) {
      try { return next(pathToFileURL(peers.resolve(specifier)).href, context); } catch {}
    }
    throw error;
  }
} });
const load = file => import(pathToFileURL(path.join(root, file)));
const { default: registerChild } = await load('src/runs/shared/subagent-prompt-runtime.js');
const { default: registerMain } = await load('src/extension/index.js');
const { registerBackgroundWorkProvider } = await load('src/api/background-work.js');
function host() {
  const hooks = new Map(), bus = new EventEmitter();
  const api = {
    on(event, handler) { if (!hooks.has(event)) hooks.set(event, []); hooks.get(event).push(handler); },
    events: { on(event, handler) { bus.on(event, handler); return () => bus.off(event, handler); }, emit: (...args) => bus.emit(...args) },
    registerTool() {}, registerMessageRenderer() {}, registerEntryRenderer() {}, registerFlag() {}, registerCommand() {}, registerShortcut() {},
    getFlag: () => false, getThinkingLevel: () => 'off', getAllTools: () => [], getActiveTools: () => [],
    getCommands: () => [], sendMessage() {}, sendUserMessage() {}, appendEntry() {},
  };
  return { api, hooks };
}
async function assertCancelledCaller(hook, ctx, when, sessionId) {
  let active = true;
  const dispose = registerBackgroundWorkProvider({ name: `caller-${when}`, listActiveWork: () => active ? [{ id: 'work', sessionId }] : [] });
  const controller = new AbortController();
  if (when === 'already') controller.abort();
  const pending = hook({}, { ...ctx, signal: controller.signal });
  const result = pending.then(() => 'settled', error => error);
  if (when === 'during') {
    await new Promise(resolve => setTimeout(resolve, 10));
    controller.abort();
  }
  let timer;
  try {
    const outcome = await Promise.race([result, new Promise(resolve => { timer = setTimeout(() => resolve('still-waiting'), 100); })]);
    assert.equal(outcome?.name, 'AbortError', `agent_end must reject cancellation; got ${String(outcome)}`);
  } finally {
    clearTimeout(timer);
    active = false;
    // Reap the old-code RED wait before unregistering its provider.
    await result;
    dispose();
  }
}
for (const when of ['already', 'during']) {
  test(`child runtime forwards ${when}-aborted operation signal with active provider`, async () => {
    const { api, hooks } = host();
    const sessionId = path.join(scratch, `child-${when}.jsonl`);
    const held = [];
    registerChild(api, { runtimeState: { currentSessionId: sessionId }, waitTool: { enabled: false }, holdFinalDrain: value => held.push(value) });
    await assertCancelledCaller(hooks.get('agent_end')[0], { hasUI: false }, when, sessionId);
    assert.deepEqual(held, [true, false]);
  });
}

test('non-UI main caller forwards operation cancellation with active provider', async () => {
  const { api, hooks } = host();
  registerMain(api);
  const sessionId = path.join(scratch, 'main.jsonl');
  const ctx = { cwd: scratch, hasUI: false, mode: 'text', sessionManager: {
    getSessionId: () => 'caller-test', getSessionFile: () => sessionId,
    getEntries: () => [], getBranch: () => [], getLeafId: () => null,
  }, ui: {} };
  try {
    // Actual session setup supplies package-owned identity to the main caller.
    for (const hook of hooks.get('session_start') ?? []) await hook({ reason: 'startup' }, ctx);
    const agentEnd = async (event, context) => { for (const hook of hooks.get('agent_end')) await hook(event, context); };
    for (const when of ['already', 'during']) await assertCancelledCaller(agentEnd, ctx, when, sessionId);
  } finally {
    for (const hook of hooks.get('session_shutdown') ?? []) await hook({}, ctx);
  }
});

test.after(() => rmSync(scratch, { recursive: true, force: true }));
