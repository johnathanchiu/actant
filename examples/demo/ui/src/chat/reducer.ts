/**
 * Top-level reducer: (entries, event) → entries.
 *
 * Folds wire events into the UI's ConsoleEntry shape. Pure — no
 * React, no I/O, no refs. Per-turn / per-tool-call mutation logic
 * lives in `turnLogic.ts` and is shared with the sub-thread reducer
 * (`subThreads.ts`) so both paths fold events identically.
 *
 * Top-level events (no `parent_thread_id` set) land here. Sub-thread
 * events are routed to `subThreads.ts` by the orchestration layer.
 */

import type { ConsoleEntry, TurnEntry } from './state'
import { applyEventToToolCall, applyEventToTurn } from './turnLogic'
import type { ActantEvent } from './wire'

export function reduce(entries: ConsoleEntry[], event: ActantEvent): ConsoleEntry[] {
  // Sub-thread events are routed elsewhere; ignore them here.
  if (event.parent_thread_id) return entries

  switch (event.type) {
    case 'turn_start':
      return ensureTurn(entries, event.thread_id, turnUidOf(event))

    case 'text_delta':
    case 'thinking_delta':
    case 'tool_call_start':
    case 'tool_call_args_delta':
    case 'tool_call_args_complete':
    case 'tool_call':
    case 'assistant_message':
      return updateCurrentTurn(entries, event.thread_id, (t) =>
        applyEventToTurn(t, event),
      )

    case 'tool_result':
    case 'tool_waiting':
    case 'tool_resolved':
      return mapToolCall(entries, event.data.tool_call_id, (c) =>
        applyEventToToolCall(c, event),
      )

    case 'complete': {
      let mutated = false
      const next = entries.map((e) => {
        if (e.kind !== 'turn') return e
        if (e.threadId !== event.thread_id) return e
        if (!e.isStreaming) return e
        mutated = true
        return { ...e, isStreaming: false }
      })
      const closed = mutated ? next : entries
      if (event.data.success) return closed
      return [
        ...closed,
        errorEntry(event.thread_id, `Run ended: ${event.data.reason} — ${event.data.message}`),
      ]
    }

    case 'error':
      return [...entries, errorEntry(event.thread_id, event.data.message)]
  }
}

// ─── Helpers ────────────────────────────────────────────────────────

function currentStreamingTurnIndex(entries: ConsoleEntry[], threadId: string): number {
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i]
    if (e.kind !== 'turn') return -1
    if (e.threadId !== threadId) continue
    if (e.isStreaming) return i
    return -1
  }
  return -1
}

function ensureTurn(entries: ConsoleEntry[], threadId: string, turnUid: string): ConsoleEntry[] {
  if (entries.some((e) => e.kind === 'turn' && e.turnUid === turnUid)) return entries
  // Close any prior streaming turn on the same thread. Only one turn
  // per thread is ever actively streaming. assistant_message normally
  // does this, but defensive: if the server drops one (mid-turn
  // error, dropped frame) the prior turn would otherwise stay
  // isStreaming=true forever, showing a stuck pulse dot AND
  // double-dotting alongside the new turn's pulse.
  const closed = entries.map((e) =>
    e.kind === 'turn' && e.threadId === threadId && e.isStreaming
      ? { ...e, isStreaming: false }
      : e,
  )
  const turn: TurnEntry = {
    kind: 'turn',
    id: `turn_${turnUid}`,
    turnUid,
    threadId,
    text: '',
    thinking: '',
    toolCalls: [],
    isStreaming: true,
    timestamp: Date.now(),
  }
  return [...closed, turn]
}

function updateCurrentTurn(
  entries: ConsoleEntry[],
  threadId: string,
  mutate: (turn: TurnEntry) => TurnEntry,
): ConsoleEntry[] {
  const idx = currentStreamingTurnIndex(entries, threadId)
  if (idx >= 0) {
    const next = entries.slice()
    next[idx] = mutate(next[idx] as TurnEntry)
    return next
  }
  // No streaming turn — synthesize one. Happens when a delta arrives
  // before its turn_start (rare; defensive).
  const turnUid = `client_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  const seeded = ensureTurn(entries, threadId, turnUid)
  const idx2 = currentStreamingTurnIndex(seeded, threadId)
  if (idx2 < 0) return seeded
  const next = seeded.slice()
  next[idx2] = mutate(next[idx2] as TurnEntry)
  return next
}

function mapToolCall(
  entries: ConsoleEntry[],
  toolCallId: string,
  mutate: (call: import('./state').ToolCallEntry) => import('./state').ToolCallEntry,
): ConsoleEntry[] {
  let mutated = false
  const next = entries.map((e) => {
    if (e.kind !== 'turn') return e
    if (!e.toolCalls.some((c) => c.id === toolCallId)) return e
    mutated = true
    return {
      ...e,
      toolCalls: e.toolCalls.map((c) => (c.id === toolCallId ? mutate(c) : c)),
    }
  })
  return mutated ? next : entries
}

function turnUidOf(event: Extract<ActantEvent, { type: 'turn_start' }>): string {
  return (
    event.data.turn_uid ??
    event.data.turn_id ??
    `client_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  )
}

function errorEntry(threadId: string, text: string): ConsoleEntry {
  return {
    kind: 'event',
    id: `event_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    threadId,
    level: 'error',
    text,
    timestamp: Date.now(),
  }
}
