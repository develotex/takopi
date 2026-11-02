import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm, readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.SUBAGENTS_ROOT;
const pi = process.env.PI_PACKAGE_ROOT;
assert.ok(root && pi, 'set SUBAGENTS_ROOT (package copy), PI_PACKAGE_ROOT (0.99.2)');

async function probe({ text = false, abort = false, supervisor = false, noWork = false, unpatched = false } = {}) {
  const dir = await mkdtemp(path.join(process.env.LIFETIME_SCRATCH_ROOT ?? '/mnt/victor_vector_volume', 'rpc-proof-'));
  const agentDir = path.join(dir, 'agent');
  await mkdir(path.join(agentDir, 'extensions/subagent'), { recursive: true });
  await writeFile(path.join(agentDir, 'extensions/subagent/config.json'), JSON.stringify({ toolActivation: 'eager', completionBatch: { enabled: true, debounceMs: 1000 } }));
  await mkdir(path.join(dir, '.pi/agents'), { recursive: true });
  await writeFile(path.join(dir, '.pi/agents/probe.md'), `---\nname: probe\ndescription: Local deterministic lifecycle child\nmodel: inherit\ntools: ${supervisor ? 'contact_supervisor' : ''}\nextensions: ${path.join(here, 'rpc-lifetime-provider.mjs')}\n---\nReturn the deterministic child result.\n`);
  await writeFile(path.join(dir, 'probe-workflow.js'), 'const child = await runs.run("leaf", { agent: "probe", task: "LEAF_ONLY_MARKER", timeoutMs: 8000 }); return child.output;');
  const args = [path.join(pi, 'dist/cli.js'), '--mode', 'rpc', '--no-extensions', '-e', path.join(root, 'index.js'), '-e', path.join(here, 'rpc-lifetime-provider.mjs'), '--provider', 'lifetime-probe', '--model', 'script', '--session', path.join(dir, 'session.jsonl')];
  if (!unpatched) args.push('--takopi-rpc-lifetime');
  const child = spawn(process.execPath, args, { cwd: dir, env: { ...process.env, PI_CODING_AGENT_DIR: agentDir, LIFETIME_TEXT_STOP: text ? '1' : '0', LIFETIME_SUPERVISOR: supervisor ? '1' : '0', LIFETIME_NO_WORK: noWork ? '1' : '0', PI_SUBAGENT_CHILD: '0', TMPDIR: dir, PI_INTERCOM_ASK_TIMEOUT_MS: '6000' } });
  const records = [];
  let stderr = '', buffer = '', resolveSettled, rejectSettled, abortAt;
  const settled = new Promise((resolve, reject) => { resolveSettled = resolve; rejectSettled = reject; });
  const timer = setTimeout(() => rejectSettled(Error(`proof timed out: ${JSON.stringify(records)}\n${stderr}`)), 20000);
  child.stderr.on('data', b => { stderr += b; });
  child.on('exit', code => rejectSettled(Error(`Pi exited ${code}: ${stderr}\n${JSON.stringify(records)}`)));
  child.stdout.on('data', bytes => {
    buffer += bytes.toString();
    while (buffer.includes('\n')) {
      const n = buffer.indexOf('\n'), line = buffer.slice(0, n); buffer = buffer.slice(n + 1);
      if (!line) continue;
      const record = JSON.parse(line); records.push(record);
      if (abort && record.type === 'message_end' && record.message?.role === 'assistant' && record.message.stopReason === 'stop' && !abortAt) {
        abortAt = Date.now();
        setTimeout(() => child.stdin.write(JSON.stringify({ type: 'clear_queue', id: 'clear-before-abort' }) + '\n'), 30);
      }
      if (record.type === 'response' && record.id === 'clear-before-abort' && record.success)
        child.stdin.write(JSON.stringify({ type: 'abort', id: 'abort' }) + '\n');
      if (record.type === 'agent_settled') resolveSettled();
    }
  });
  child.stdin.write(JSON.stringify({ type: 'prompt', id: 'start', message: 'Execute the supplied local lifecycle workflow, then synthesize its result.' }) + '\n');
  try {
    await settled;
    if (abort) {
      // agent_settled precedes the abort response in Pi. Give the bounded ACK
      // and any accidental completion-induced restart a chance to appear.
      const ackDeadline = Date.now() + 1500;
      while (!records.some(r => r.type === 'response' && r.id === 'abort') && Date.now() < ackDeadline)
        await new Promise(resolve => setTimeout(resolve, 10));
      assert.ok(records.some(r => r.type === 'response' && r.id === 'abort' && r.success), stderr);
      assert.ok(Date.now() - abortAt < 2000, 'abort must not wait for 30-minute drain');
      await new Promise(resolve => setTimeout(resolve, 4500));
      const firstSettle = records.findIndex(r => r.type === 'agent_settled');
      assert.equal(records.slice(firstSettle + 1).filter(r => r.type === 'agent_start').length, 0, 'late completion must not restart aborted parent');
      const boundary = records.length;
      child.stdin.write(JSON.stringify({ type: 'prompt', id: 'manual', message: 'Explicitly continue and synthesize available child results.' }) + '\n');
      const manualDeadline = Date.now() + 10000;
      while (!records.slice(boundary).some(r => r.type === 'agent_settled') && Date.now() < manualDeadline)
        await new Promise(resolve => setTimeout(resolve, 10));
      assert.ok(records.slice(boundary).some(r => r.type === 'agent_settled'), stderr);
      assert.equal(records.slice(boundary).filter(r => r.type === 'extension_error' && r.error?.startsWith('TAKOPI_RPC_LIFETIME:')).length, 0, 'fresh continuation must not hide a lifecycle error');
      if (supervisor) assert.ok(records.slice(boundary).some(r => r.type === 'tool_execution_end' && r.toolName === 'subagent_supervisor' && !r.isError), 'explicit continuation must answer retained ask');
      assert.ok(records.slice(boundary).some(r => r.type === 'message_end' && JSON.stringify(r.message?.content).includes('SYNTHESIZED_REAL_CHILD_RESULT_42')), 'fresh explicit operation must work after abort');
      return { records: records.slice(0, boundary), stderr };
    }
    return { records, stderr };
  } finally {
    clearTimeout(timer);
    child.stdin.end();
    const exited = child.exitCode !== null || child.signalCode !== null ? Promise.resolve() : new Promise(resolve => child.once('exit', resolve));
    const kill = setTimeout(() => child.kill('SIGKILL'), 3000);
    await exited; clearTimeout(kill);
    // Detached package runners own separate process groups. Reap only this
    // probe's exact cmdline-scoped runners, even if a failing ask outlives Pi.
    for (const signal of ['SIGTERM', 'SIGKILL']) {
      for (const pid of await readdir('/proc')) {
        if (!/^\d+$/.test(pid)) continue;
        try {
          const command = await readFile(`/proc/${pid}/cmdline`, 'utf8');
          if (command.includes(dir + '/') && command.includes('subagent-runner'))
            process.kill(Number(pid), signal);
        } catch (error) { if (!['ENOENT', 'ESRCH', 'EACCES'].includes(error.code)) throw error; }
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    await rm(dir, { recursive: true, force: true });
  }
}

test('absent compatibility extension fails startup on required capability flag', async () => {
  const dir = await mkdtemp(path.join(process.env.LIFETIME_SCRATCH_ROOT ?? '/mnt/victor_vector_volume', 'rpc-no-extension-'));
  try {
    const result = spawnSync(process.execPath, [path.join(pi, 'dist/cli.js'), '--no-extensions', '--mode', 'rpc', '--takopi-rpc-lifetime'], { cwd: dir, env: { ...process.env, PI_CODING_AGENT_DIR: dir }, encoding: 'utf8', timeout: 10000, input: '' });
    assert.equal(result.error, undefined);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Unknown option: --takopi-rpc-lifetime/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('genuine empty stop without child remains one ordinary settlement', { timeout: 25000 }, async () => {
  const { records } = await probe({ noWork: true });
  assert.equal(records.filter(r => r.type === 'tool_execution_start').length, 0);
  assert.equal(records.filter(r => r.type === 'extension_error').length, 0);
  assert.equal(records.filter(r => r.type === 'agent_settled').length, 1);
  const answers = records.filter(r => r.type === 'message_end' && r.message?.role === 'assistant');
  assert.equal(answers.length, 1);
  assert.deepEqual(answers[0].message.content, []);
});

for (const text of [false, true]) {
  test(`actual Pi + real async workflow + child survives ${text ? 'text' : 'empty'} parent stop and synthesizes`, { timeout: 25000 }, async () => {
    const { records, stderr } = await probe({ text, unpatched: process.env.LIFETIME_EXPECT_RED === '1' });
    const answers = records.filter(r => r.type === 'message_end' && r.message?.role === 'assistant').map(r => JSON.stringify(r.message.content));
    assert.ok(answers.some(s => s.includes('SYNTHESIZED_REAL_CHILD_RESULT_42')), `${JSON.stringify(records)}\n${stderr}`);
    assert.equal(records.filter(r => r.type === 'agent_settled').length, 1);
    assert.equal(records.filter(r => r.type === 'extension_error' && r.error?.startsWith('TAKOPI_RPC_LIFETIME:')).length, 0);
  });
}

test('real pending supervisor request wakes parent to reply and then synthesize', { timeout: 25000 }, async () => {
  const { records, stderr } = await probe({ supervisor: true });
  assert.ok(records.some(r => r.type === 'tool_execution_end' && r.toolName === 'subagent_supervisor' && !r.isError), stderr);
  assert.ok(records.some(r => r.type === 'message_end' && r.message?.role === 'assistant' && JSON.stringify(r.message.content).includes('SYNTHESIZED_REAL_CHILD_RESULT_42')), JSON.stringify(records.filter(r => ['extension_error', 'message_end'].includes(r.type) && r.message?.role !== 'system')) + stderr);
});

test('late native supervisor ask cannot restart aborted parent; explicit continuation answers retained ask', { timeout: 25000 }, async () => {
  const { records } = await probe({ abort: true, supervisor: true });
  assert.ok(records.some(r => r.type === 'response' && r.id === 'clear-before-abort' && r.success));
  assert.equal(records.filter(r => r.type === 'tool_execution_start' && r.toolName === 'subagent_supervisor').length, 0, 'no supervisor action before explicit continuation');
});

test('actual operation abort during RPC drain ACKs promptly without parent re-prompt', { timeout: 25000 }, async () => {
  const { records, stderr } = await probe({ abort: true });
  assert.ok(records.some(r => r.type === 'extension_error' && r.event === 'agent_end' && r.error.includes('TAKOPI_RPC_LIFETIME:')), stderr);
  const lastStop = records.findIndex(r => r.type === 'message_end' && r.message?.role === 'assistant' && r.message.stopReason === 'stop');
  assert.equal(records.slice(lastStop + 1).filter(r => r.type === 'agent_start').length, 0, JSON.stringify(records));
});
