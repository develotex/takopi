import assert from 'node:assert/strict';
import { test } from 'node:test';

const { default: extension } = await import(new URL('./session-title-rpc.ts', import.meta.url));

function harness({ mode = 'rpc', entries = [], allEntries = entries, name, response = '💻 Исправление автоматических названий топиков', fail, auth = true } = {}) {
  const handlers = new Map();
  const requests = [];
  const statuses = [];
  const state = { name, sets: 0 };
  let result = async () => {
    if (fail) throw new Error('upstream secret diagnostic');
    return { content: [{ type: 'text', text: response }], stopReason: 'stop' };
  };
  const pi = {
    on: (type, handler) => handlers.set(type, handler),
    getSessionName: () => state.name,
    setSessionName: value => { state.name = value; state.sets++; },
  };
  const abortController = new AbortController();
  const ctx = {
    mode,
    signal: abortController.signal,
    sessionManager: { getBranch: () => entries, getEntries: () => allEntries, getSessionId: () => 'test-session' },
    ui: { notify: (...args) => statuses.push(args) },
    modelRegistry: {
      find: () => ({ id: 'test', provider: 'test' }),
      hasConfiguredAuth: () => auth,
      streamSimple: (model, context, options) => {
        requests.push({ model, context, options });
        return { result: () => result() };
      },
    },
  };
  extension(pi);
  return { state, requests, statuses, ctx, setResult: value => { result = value; },
    event: (type, event = {}) => handlers.get(type)?.(event, ctx),
    start: () => handlers.get('session_start')?.({}, ctx),
    abortController,
    capture: text => handlers.get('before_agent_start')?.({ prompt: text }, ctx),
    prompt: async text => {
      await handlers.get('before_agent_start')?.({ prompt: text }, ctx);
      await handlers.get('turn_start')?.({}, ctx);
    },
  };
}

test('names a fresh RPC session once and leaves the main prompt intact', async () => {
  const h = harness(); await h.start();
  await h.prompt('Исправь автоматические названия топиков');
  assert.equal(h.state.name, '💻 Исправление автоматических названий топиков');
  await h.prompt('А теперь другое задание');
  assert.equal(h.state.sets, 1);
  assert.equal(h.requests.length, 1);
});

test('does not call the provider in JSON, print or TUI modes', async () => {
  for (const mode of ['json', 'text', 'tui']) {
    const h = harness({ mode }); await h.start(); await h.prompt('Task');
    assert.equal(h.requests.length, 0); assert.equal(h.state.sets, 0);
  }
});

test('never renames named or historical unnamed sessions', async () => {
  for (const options of [{ name: 'Manual title' }, { entries: [{ type: 'message', message: { role: 'user' } }] }]) {
    const h = harness(options); await h.start(); await h.prompt('Another task');
    assert.equal(h.requests.length, 0); assert.equal(h.state.sets, 0);
  }
});

test('removes only a known leading Takopi guidance paragraph before truncation', async () => {
  const prefix = 'Messages marked [takopi-live:<marker>] are conversation from the user in this task, including questions and corrections. ' + 'context '.repeat(320) + 'Continue the main task.\n\n';
  const h = harness(); await h.start(); await h.prompt(prefix + 'Верни корректные названия тем');
  const input = h.requests[0].context.messages[0].content[0].text;
  assert.match(input, /Верни корректные названия тем/);
  assert.doesNotMatch(input, /context context|Messages marked/);
  const other = harness(); await other.start(); await other.prompt('Explain RPC\n\nKeep this paragraph');
  assert.match(other.requests[0].context.messages[0].content[0].text, /Explain RPC\n\nKeep this paragraph/);
});

test('does not overwrite a manual title chosen during generation', async () => {
  const h = harness(); await h.start();
  h.setResult(async () => { h.state.name = 'My manual title'; return { content: [{ type: 'text', text: '💻 Generated title' }], stopReason: 'stop' }; });
  await h.prompt('Task');
  assert.equal(h.state.name, 'My manual title'); assert.equal(h.state.sets, 0);
});

test('failure is nonfatal, reports only a safe category, and is not retried every turn', async () => {
  const h = harness({ fail: true }); await h.start();
  await assert.doesNotReject(() => h.prompt('Task'));
  await h.prompt('Next');
  assert.equal(h.state.sets, 0); assert.equal(h.requests.length, 1);
  assert.equal(h.statuses.length, 1);
  assert.doesNotMatch(JSON.stringify(h.statuses), /upstream secret/);
});

test('rejects empty and multiline provider output rather than publishing it', async () => {
  for (const response of ['', '   ', '💻 One title\nAnother unwanted title']) {
    const h = harness({ response }); await h.start(); await h.prompt('Task');
    assert.equal(h.state.sets, 0);
  }
});

test('trims quotes and code fences and bounds Telegram title length', async () => {
  for (const response of ['"💻 Исправление автоматических названий топиков"', '```text\n💻 Исправление автоматических названий топиков\n```']) {
    const h = harness({ response }); await h.start(); await h.prompt('Task');
    assert.equal(h.state.name, '💻 Исправление автоматических названий топиков');
  }
  const h = harness({ response: '💻 ' + 'Проверка '.repeat(30) }); await h.start(); await h.prompt('Task');
  assert.ok([...h.state.name].length <= 80);
});

test('does not publish a stale result after session switch', async () => {
  const h = harness(); await h.start();
  h.setResult(async () => {
    h.ctx.sessionManager.getSessionId = () => 'other-session';
    await h.start();
    return { content: [{ type: 'text', text: '💻 Old session title' }], stopReason: 'stop' };
  });
  await h.prompt('Old task');
  assert.equal(h.state.sets, 0);
});

test('captures preflight without awaiting the network; abort in active turn cancels naming', async () => {
  const h = harness(); await h.start();
  await h.capture('Task');
  assert.equal(h.requests.length, 0);
  let entered;
  const started = new Promise(resolve => { entered = resolve; });
  h.setResult(() => new Promise(resolve => {
    const signal = h.requests[0].options.signal;
    entered();
    signal.addEventListener('abort', () => resolve({ content: [{ type: 'text', text: '💻 Canceled task title' }], stopReason: 'aborted' }), { once: true });
  }));
  const naming = h.event('turn_start');
  await started; h.abortController.abort(); await naming;
  assert.equal(h.state.sets, 0);
  assert.equal(h.requests[0].options.signal.aborted, true);
});

test('shutdown cancels the naming request without publishing a late result', async () => {
  const h = harness(); await h.start(); await h.capture('Task');
  let entered;
  const started = new Promise(resolve => { entered = resolve; });
  h.setResult(() => new Promise(resolve => {
    entered();
    h.requests[0].options.signal.addEventListener('abort', () => resolve({ content: [{ type: 'text', text: '💻 Late title' }], stopReason: 'stop' }), { once: true });
  }));
  const naming = h.event('turn_start'); await started;
  await h.event('session_shutdown'); await naming;
  assert.equal(h.state.sets, 0); assert.equal(h.requests[0].options.signal.aborted, true);
});

test('provider timeout finishes metadata work without interrupting the task', { timeout: 12000 }, async () => {
  const h = harness(); await h.start();
  h.setResult(() => new Promise((_resolve, reject) => {
    h.requests[0].options.signal.addEventListener('abort', () => reject(new Error('provider timeout')), { once: true });
  }));
  const keepAlive = setTimeout(() => {}, 11000);
  try {
    await assert.doesNotReject(() => h.prompt('Task'));
    assert.equal(h.state.sets, 0);
    assert.equal(h.requests[0].options.signal.aborted, true);
    assert.equal(h.statuses.length, 1);
  } finally { clearTimeout(keepAlive); }
});

test('does not name a historical session whose current branch has only a summary', async () => {
  const h = harness({ entries: [{ type: 'branch_summary' }], allEntries: [{ type: 'message', message: { role: 'user', content: 'Old task' } }, { type: 'branch_summary' }] });
  await h.start(); await h.prompt('A later unrelated task');
  assert.equal(h.requests.length, 0); assert.equal(h.state.sets, 0);
});

test('does not publish a bare emoji after truncating a long unbroken word', async () => {
  const h = harness({ response: '💻 ' + 'x'.repeat(150) });
  await h.start(); await h.prompt('Task');
  assert.equal(h.state.sets, 0);
});

test('abort, missing credentials and provider error messages are nonfatal', async () => {
  const noAuth = harness({ auth: false }); await noAuth.start(); await noAuth.prompt('Task');
  assert.equal(noAuth.requests.length, 0);
  const h = harness(); await h.start();
  h.setResult(async () => ({ content: [{ type: 'text', text: '💻 Partial title' }], stopReason: 'error', errorMessage: 'sensitive provider error' }));
  await h.prompt('Task'); assert.equal(h.state.sets, 0);
});
