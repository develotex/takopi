// Deterministic, no-network provider used only by rpc-lifetime-real.test.mjs.
import { AssistantMessageEventStream } from '@earendil-works/pi-ai';

export default function (pi) {
  pi.registerProvider('lifetime-probe', {
    api: 'lifetime-script', apiKey: 'local-test-only', baseUrl: 'http://no-network.invalid',
    models: [{ id: 'script', name: 'Local lifecycle script', api: 'lifetime-script',
      reasoning: false, input: ['text'], contextWindow: 128000, maxTokens: 4000,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    streamSimple(model, context, options) {
      const stream = new AssistantMessageEventStream();
      void (async () => {
        const message = { role: 'assistant', api: model.api, provider: model.provider,
          model: model.id, timestamp: Date.now(), content: [], stopReason: 'stop',
          usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
        const text = m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
        const child = context.messages.some(m => m.role === 'user' && text(m).includes('LEAF_ONLY_MARKER'));
        if (!child && process.env.LIFETIME_NO_WORK === '1') {
          message.content = [];
        } else if (child) {
          await new Promise(resolve => setTimeout(resolve, 650));
          if (process.env.LIFETIME_SUPERVISOR === '1' && !context.messages.some(m => m.role === 'toolResult' && m.toolName === 'contact_supervisor')) {
            message.content = [{ type: 'toolCall', id: 'ask-parent', name: 'contact_supervisor', arguments: { reason: 'need_decision', message: 'PROBE_SUPERVISOR_QUESTION: approve local result?' } }];
            message.stopReason = 'toolUse';
          } else message.content = [{ type: 'text', text: 'REAL_CHILD_RESULT_42' }];
        } else if (!context.messages.some(m => m.role === 'toolResult' && m.toolName === 'subagent')) {
          message.content = [{ type: 'toolCall', id: 'launch-owned-workflow', name: 'subagent',
            arguments: { workflow: './probe-workflow.js', async: true } }];
          message.stopReason = 'toolUse';
        } else if (context.messages.some(m => text(m).includes('PROBE_SUPERVISOR_QUESTION')) && !context.messages.some(m => m.role === 'toolResult' && m.toolName === 'subagent_supervisor')) {
          message.content = [{ type: 'toolCall', id: 'answer-child', name: 'subagent_supervisor', arguments: { action: 'reply', message: 'Approved by scripted supervisor.' } }];
          message.stopReason = 'toolUse';
        } else if (context.messages.some(m => text(m).includes('REAL_CHILD_RESULT_42'))) {
          message.content = [{ type: 'text', text: 'SYNTHESIZED_REAL_CHILD_RESULT_42' }];
        } else {
          message.content = process.env.LIFETIME_TEXT_STOP === '1' ? [{ type: 'text', text: 'Waiting for owned child.' }] : [];
        }
        if (options?.signal?.aborted) {
          message.stopReason = 'aborted';
          message.errorMessage = 'Local lifecycle script aborted';
          stream.push({ type: 'error', reason: 'aborted', error: message });
        } else {
          stream.push({ type: 'start', partial: message });
          stream.push({ type: 'done', reason: message.stopReason, message });
        }
        stream.end();
      })();
      return stream;
    },
  });
}
