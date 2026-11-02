// Compatibility only: Pi 0.99.2 keeps its operation signal alive while awaiting
// agent_end handlers, but NOT at agent_before_settle. Never move this wait there.
import { setTimeout as delay } from "node:timers/promises";
import { drainOutstandingWork } from "../runs/background/auto-drain.js";

// Status snapshots can lag an acknowledged supervisor reply. Only that exact
// replied run/agent may wait for actual status clearance, for at most 5 seconds.
// New pending asks are checked by auto-drain before this attention predicate.
export function rpcAttentionPredicate(replies = new Map(), channel, now = Date.now) {
    return (run) => {
        const attention = run.activityState === "needs_attention" || run.steps.some(step => step.activityState === "needs_attention");
        if (!attention) return false;
        const receipt = replies.get(run.id);
        if (!receipt || now() >= receipt.deadline || !channel.hasResolvedReply(receipt)) return true;
        if (run.currentTool && run.currentTool !== "contact_supervisor") return true;
        const steps = run.steps.filter(step => step.activityState === "needs_attention");
        return !steps.length || steps.some(step => step.agent !== receipt.agent || step.currentTool !== "contact_supervisor");
    };
}

export async function drainRpcLifetime(deps) {
    const { ctx, state, events, supervisorChannel, completionNotifier, resultWatcher } = deps;
    const signal = ctx.signal;
    const sessionId = state.currentSessionId;
    try {
        if (!signal)
            throw new Error("RPC operation signal unavailable; expected Pi 0.99.2 agent_end");
        signal.throwIfAborted();
        await (deps.drain ?? drainOutstandingWork)({ state, events, signal, hasPendingSupervisorRequest: supervisorChannel.hasPendingRequests, isAttention: rpcAttentionPredicate(deps.supervisorReplies, supervisorChannel) });
        signal.throwIfAborted();
        // Terminal status can precede the watcher/coalescer and notification
        // batch timers. Drain delivery too, without creating a second notifier
        // or claiming results owned by another session/runtime.
        const deadline = Date.now() + (deps.deliveryTimeoutMs ?? 30000);
        resultWatcher.refreshResultDelivery();
        while (true) {
            signal.throwIfAborted();
            if (state.currentSessionId !== sessionId)
                throw new Error("RPC delivery stopped because the active session changed");
            completionNotifier.flushPendingDelivery();
            if (!resultWatcher.hasPendingDelivery() && !completionNotifier.hasPendingDelivery())
                break;
            if (Date.now() >= deadline)
                throw new Error("RPC completion delivery timed out with pending results");
            await delay(10, undefined, { signal });
        }
        // The native supervisor channel queues its ask with triggerTurn. A
        // pending ask without that queue is not permission to tear down work.
        while (supervisorChannel.hasPendingRequests() && !supervisorChannel.hasQueuedSupervisorWake()) {
            signal.throwIfAborted();
            if (state.currentSessionId !== sessionId)
                throw new Error("RPC supervisor delivery stopped because the active session changed");
            if (Date.now() >= deadline)
                throw new Error("Pending supervisor request has no native parent wake before delivery deadline");
            await delay(10, undefined, { signal });
        }
        signal.throwIfAborted();
    }
    catch (error) {
        throw new Error(`TAKOPI_RPC_LIFETIME: ${error instanceof Error ? error.message : String(error)}`, { cause: error });
    }
}
