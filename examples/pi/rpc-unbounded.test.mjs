// Regression: implicit runtime deadlines must not terminate healthy owned work.
import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks, createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
const root = process.env.SUBAGENTS_ROOT;
const pi = process.env.PI_PACKAGE_ROOT;
assert.ok(root && pi, 'select pinned package and Pi');
const peers = createRequire(path.join(pi, 'package.json'));
registerHooks({ resolve(specifier, context, next) {
  try { return next(specifier, context); }
  catch (error) {
    if (specifier === '@earendil-works/pi-coding-agent') return next(pathToFileURL(path.join(pi, 'dist/index.js')).href, context);
    if (/^@earendil-works\/pi-(agent-core|ai|tui)$/.test(specifier)) return next(pathToFileURL(path.join(pi, 'node_modules', specifier, 'dist/index.js')).href, context);
    if (!specifier.startsWith('.') && !specifier.startsWith('/')) {
      try { return next(pathToFileURL(peers.resolve(specifier)).href, context); } catch {}
    }
    throw error;
  }
} });
const load = file => import(pathToFileURL(path.join(root, file)));
const { drainOutstandingWork } = await load('src/runs/background/auto-drain.js');
const { resolveSingleAgentLaunchTimeout, resolveForegroundTimeout, prepareWorkflowLaunchParams, createSubagentExecutor } = await load('src/runs/foreground/subagent-executor.js');
const { buildAsyncRunnerSteps } = await load('src/runs/background/async-execution.js');

test('auto-drain re-arms finite windows for four virtual hours and completes exactly once', async () => {
  let now = 0, waits = 0, completions = 0;
  await drainOutstandingWork({ state: { currentSessionId: 'owner' }, now: () => now,
    hasWork: id => id === 'owner' && now < 4 * 60 * 60 * 1000,
    wait: async (args, signal, deps) => {
      assert.equal(args.all, true);
      assert.ok(args.timeoutMs > 0 && args.timeoutMs <= 1800000);
      assert.equal(deps.failOnFailedRuns, true);
      now += args.timeoutMs; waits++;
      return { content: [], details: { wait: { reason: 'window_elapsed' } } };
    },
  }, { begin() {}, predicate() {}, complete() { completions++; }, deny() { assert.fail('healthy work denied'); } });
  assert.equal(waits, 8);
  assert.equal(completions, 1);
});

test('late cancellation, owner change, real errors and supervisor asks retain their barriers', async () => {
  for (const outcome of ['abort', 'session', 'error', 'supervisor']) {
    const controller = new AbortController();
    const state = { currentSessionId: 'owner' };
    let now = 0, waits = 0, pending = false;
    const promise = drainOutstandingWork({ state, signal: controller.signal, now: () => now,
      hasWork: () => true, hasPendingSupervisorRequest: () => pending,
      wait: async args => {
        now += args.timeoutMs;
        if (++waits === 4) {
          if (outcome === 'abort') controller.abort();
          if (outcome === 'session') state.currentSessionId = 'foreign';
          if (outcome === 'error') return { isError: true, content: [{ type: 'text', text: 'actual child failure' }], details: {} };
          if (outcome === 'supervisor') pending = true;
        }
        return { content: [], details: {} };
      },
    });
    if (outcome === 'supervisor') await promise;
    else await assert.rejects(promise, outcome === 'abort' ? /abort/i : outcome === 'session' ? /session changed/ : /actual child failure/);
    assert.equal(waits, 4);
    assert.equal(now, 7200000);
  }
});

test('explicit drain deadline remains enforced across windows', async () => {
  let now = 0;
  await assert.rejects(drainOutstandingWork({ state: { currentSessionId: 'owner' }, timeoutMs: 3600001,
    now: () => now, hasWork: () => true,
    wait: async args => { assert.ok(args.timeoutMs <= 1800000); now += args.timeoutMs; return { content: [], details: {} }; },
  }), /timed out after 3600001ms/);
  assert.equal(now, 3600001);
});

test('foreground, async single, chain, tasks and workflow omission serialize without deadline', () => {
  for (const async of [false, true]) for (const params of [{}, { chain: [{ agent: 'probe', task: 'x' }] }, { tasks: [{ agent: 'probe', task: 'x' }] }, { workflowScript: 'x.js' }]) {
    const resolved = resolveSingleAgentLaunchTimeout(params, async);
    assert.deepEqual(JSON.parse(JSON.stringify(resolved)), {});
  }
});

test('explicit aliases and config defaults retain finite deadlines and validation', () => {
  for (const async of [false, true]) {
    assert.deepEqual(resolveSingleAgentLaunchTimeout({ timeoutMs: 17 }, async), { timeoutMs: 17 });
    assert.deepEqual(resolveSingleAgentLaunchTimeout({ maxRuntimeMs: 19 }, async), { timeoutMs: 19 });
    assert.deepEqual(resolveSingleAgentLaunchTimeout({}, async, 23), { timeoutMs: 23 });
  }
  assert.match(resolveForegroundTimeout({ timeoutMs: 1, maxRuntimeMs: 2 }).error, /aliases/);
  assert.match(resolveForegroundTimeout({ timeoutMs: 2147483648 }).error, /no larger/);
});

test('async chain child builder and JSON round trip preserve absent and agent-explicit deadline', () => {
  for (const defaultTimeoutMs of [undefined, 71]) {
    const result = buildAsyncRunnerSteps('unbounded-test', {
      chain: [{ agent: 'probe', task: 'Inspect the task' }],
      agents: [{ name: 'probe', description: 'test', systemPrompt: 'test', source: 'user', filePath: '/tmp/probe.md', ...(defaultTimeoutMs ? { defaultTimeoutMs } : {}) }],
      ctx: { cwd: '/tmp', currentSessionId: 'owner' }, asyncDir: '/tmp/unbounded-builder',
      unknownAgentDiagnosticContext: {},
    });
    assert.equal(result.error, undefined);
    const serialized = JSON.parse(JSON.stringify(result.steps));
    assert.equal(serialized[0].timeoutMs, defaultTimeoutMs);
    assert.equal(Object.hasOwn(serialized[0], 'timeoutMs'), defaultTimeoutMs !== undefined);
  }
});

test('async chain validates configured agent deadlines before constructing any runner step', () => {
  const build = (defaultTimeoutMs, parallel = false) => buildAsyncRunnerSteps('deadline-boundary', {
    chain: parallel ? [{ parallel: [{ agent: 'probe', task: 'Inspect task' }] }] : [{ agent: 'probe', task: 'Inspect task' }],
    agents: [{ name: 'probe', description: 'test', systemPrompt: 'test', source: 'user', filePath: '/tmp/probe.md', defaultTimeoutMs }],
    ctx: { cwd: '/tmp', currentSessionId: 'owner' }, asyncDir: '/tmp/unbounded-builder', unknownAgentDiagnosticContext: {},
  });
  for (const parallel of [false, true]) {
    for (const timeout of [2147483648, 0, -1, 1.5, NaN, Infinity, '100']) {
      const result = build(timeout, parallel);
      assert.match(result.error ?? '', /timeoutMs must be a positive integer no larger than 2147483647/, `agent deadline ${String(timeout)} must be rejected before runner construction`);
      assert.equal(result.steps, undefined);
    }
    for (const timeout of [undefined, 1, 2147483647]) {
      const result = build(timeout, parallel);
      assert.equal(result.error, undefined);
      const step = parallel ? result.steps[0].parallel[0] : result.steps[0];
      assert.equal(step.timeoutMs, timeout);
    }
  }
});

test('actual workflow root rejects contradictory, nonpositive, fractional and overflowing runtime aliases before discovery', async () => {
  for (const async of [false, true]) {
    for (const [limits, expected] of [
      [{ timeoutMs: 1000, maxRuntimeMs: 2000 }, /aliases/],
      [{ timeoutMs: 0 }, /positive integer/],
      [{ maxRuntimeMs: -1 }, /positive integer/],
      [{ timeoutMs: 1.5 }, /positive integer/],
      [{ timeoutMs: 2147483648 }, /no larger/],
    ]) {
      const executor = createSubagentExecutor({ config: {}, state: {}, asyncByDefault: true,
        discoverAgents() { assert.fail('invalid workflow deadline reached agent discovery'); },
      });
      const result = await executor.execute('deadline-root', { workflowScript: 'return 1;', async, ...limits }, undefined, undefined, {
        cwd: '/tmp', sessionManager: { getSessionFile: () => '/tmp/deadline-test.jsonl', getSessionId: () => 'deadline-test' },
      });
      assert.equal(result.isError, true);
      assert.match(result.content.map(part => part.text ?? '').join(' '), expected);
    }
  }
});

const { waitForSubagents } = await load('src/runs/background/subagent-wait.js');
const { updateActiveRunIndex } = await load('src/runs/background/active-run-index.js');

// Only clock/sleep vary; real status reconciliation and attention handling run.
test('real async waiter crosses four hours of idle attention without failure or spin; foreign work untouched', async () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'rpc-unbounded-'));
  const owned = path.join(dir, 'owned'), foreign = path.join(dir, 'foreign');
  mkdirSync(owned); mkdirSync(foreign);
  let now = Date.now(), sleeps = 0, waits = 0;
  const start = now;
  const status = { runId: 'owned', sessionId: 'owner', state: 'running', mode: 'single', pid: process.pid,
    startedAt: now, updatedAt: now, activityState: 'needs_attention', currentTool: 'bash',
    steps: [{ agent: 'probe', state: 'running', activityState: 'needs_attention', currentTool: 'bash' }] };
  const save = () => writeFileSync(path.join(owned, 'status.json'), JSON.stringify(status));
  save(); updateActiveRunIndex(owned, 'running');
  writeFileSync(path.join(foreign, 'status.json'), JSON.stringify({ ...status, runId: 'foreign', sessionId: 'other' }));
  updateActiveRunIndex(foreign, 'running');
  const foreignBefore = readFileSync(path.join(foreign, 'status.json'), 'utf8');
  const state = { currentSessionId: 'owner' };
  try {
    await drainOutstandingWork({ state, asyncDirRoot: dir, resultsDir: path.join(dir, 'results'), now: () => now,
      wait: async (args, signal, deps) => {
        waits++;
        return waitForSubagents(args, signal, { ...deps, pollIntervalMs: 1800000,
          sleep: async ms => {
            assert.ok(ms > 0); sleeps++; now += ms;
            if (now - start >= 14400000) { status.state = 'completed'; status.steps[0].state = 'completed'; }
            status.updatedAt = now; save();
          },
        });
      },
    });
    assert.equal(waits, 8);
    assert.equal(sleeps, 8);
    assert.equal(now - start, 14400000);
    assert.equal(readFileSync(path.join(foreign, 'status.json'), 'utf8'), foreignBefore);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('workflow/nested serialization never invents a deadline; explicit parent and child defaults survive', () => {
  const plain = prepareWorkflowLaunchParams({}, { agent: 'probe', task: 'x' }, 'parent', 'leaf');
  const json = JSON.parse(JSON.stringify(plain));
  assert.equal(Object.hasOwn(json, 'timeoutMs'), false);
  assert.equal(Object.hasOwn(json, 'workflowParentDeadlineAt'), false);
  const deadline = Date.now() + 5000;
  const inherited = prepareWorkflowLaunchParams({}, { agent: 'probe', task: 'x' }, 'parent', 'leaf', { parentDeadlineAt: deadline });
  assert.equal(inherited.workflowParentDeadlineAt, deadline);
  const explicit = prepareWorkflowLaunchParams({ timeoutMs: 41 }, { agent: 'probe', task: 'x' }, 'parent', 'leaf');
  assert.equal(JSON.parse(JSON.stringify(explicit)).timeoutMs, 41);
});
