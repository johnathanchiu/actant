/**
 * Sub-thread state map + reducer.
 *
 * When the parent thread invokes `task()`, the server's coordinator
 * spawns a sub-thread and dual-publishes its events to the parent's
 * SSE channel (via the actant coordinator primitives —
 * `make_sub_thread_aware_hooks_factory`). Those events carry
 * `parent_thread_id` set to the parent.
 *
 * This module:
 *   - Owns the `SubThreadMap` (one entry per active sub-thread).
 *   - Reduces sub-thread events into the right activity's turn list.
 *   - Uses the same per-turn / per-tool-call logic as the top-level
 *     reducer (via `turnLogic.ts`) so live and rehydration paths
 *     produce identical shapes.
 *
 * The chat UI reads `subThreads[subThreadId]` to render a
 * `NestedTranscript` inside the parent's `task()` tool-call card.
 */

import type { TurnEntry } from './state'
import { applyEventToToolCall, applyEventToTurn } from './turnLogic'
import type { ActantEvent } from './wire'

export type SubThreadActivity = {
  subThreadId: string
  parentThreadId: string
  parentToolCallId: string
  subagent: string | null
  turns: TurnEntry[]
  isStreaming: boolean
}

export type SubThreadMap = Record<string, SubThreadActivity>

/** Apply one event to the sub-thread map. Returns a new map.
 *
 * @param map               current sub-thread map
 * @param event             the event to apply
 * @param parentThreadId    the parent thread this UI is currently
 *                          rendering. Events whose `parent_thread_id`
 *                          doesn't match are ignored (defensive).
 */
export function reduceSubThread(
  map: SubThreadMap,
  event: ActantEvent,
  parentThreadId: string,
): SubThreadMap {
  if (!event.parent_thread_id) return map
  if (event.parent_thread_id !== parentThreadId) return map

  const subThreadId = event.thread_id
  const existing = map[subThreadId]
  const activity: SubThreadActivity =
    existing ?? {
      subThreadId,
      parentThreadId: event.parent_thread_id,
      parentToolCallId: event.parent_tool_call_id ?? '',
      subagent: event.subagent ?? null,
      turns: [],
      isStreaming: true,
    }

  const nextActivity = applyEventToActivity(activity, event)
  if (nextActivity === activity && existing) return map
  return { ...map, [subThreadId]: nextActivity }
}

function applyEventToActivity(
  activity: SubThreadActivity,
  event: ActantEvent,
): SubThreadActivity {
  switch (event.type) {
    case 'turn_start': {
      const turnUid =
        event.data.turn_uid ??
        event.data.turn_id ??
        `client_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      if (activity.turns.some((t) => t.turnUid === turnUid)) return activity
      // Close any prior streaming turn — only one active streamer per
      // sub-thread at a time. Mirrors the top-level reducer's
      // ensureTurn behavior; without this, two empty+streaming sub
      // turns can render back-to-back pulse dots inside the nested
      // transcript.
      const closedTurns = activity.turns.map((t) =>
        t.isStreaming ? { ...t, isStreaming: false } : t,
      )
      const turn: TurnEntry = {
        kind: 'turn',
        id: `subturn_${activity.subThreadId}_${turnUid}`,
        turnUid,
        threadId: activity.subThreadId,
        text: '',
        thinking: '',
        toolCalls: [],
        isStreaming: true,
        timestamp: Date.now(),
      }
      return { ...activity, turns: [...closedTurns, turn], isStreaming: true }
    }

    case 'text_delta':
    case 'thinking_delta':
    case 'tool_call_start':
    case 'tool_call_args_delta':
    case 'tool_call_args_complete':
    case 'tool_call':
    case 'assistant_message': {
      const idx = currentStreamingTurnIdx(activity.turns)
      if (idx >= 0) {
        const turns = activity.turns.slice()
        turns[idx] = applyEventToTurn(turns[idx], event)
        return { ...activity, turns }
      }
      // No streaming turn → synthesize one (delta arrived before
      // turn_start). Match the top-level reducer's defensive behavior.
      const turnUid = `client_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      const seed: TurnEntry = {
        kind: 'turn',
        id: `subturn_${activity.subThreadId}_${turnUid}`,
        turnUid,
        threadId: activity.subThreadId,
        text: '',
        thinking: '',
        toolCalls: [],
        isStreaming: true,
        timestamp: Date.now(),
      }
      return {
        ...activity,
        turns: [...activity.turns, applyEventToTurn(seed, event)],
      }
    }

    case 'tool_result':
    case 'tool_waiting':
    case 'tool_resolved': {
      let mutated = false
      const turns = activity.turns.map((t) => {
        if (!t.toolCalls.some((c) => c.id === event.data.tool_call_id)) return t
        mutated = true
        return {
          ...t,
          toolCalls: t.toolCalls.map((c) =>
            c.id === event.data.tool_call_id ? applyEventToToolCall(c, event) : c,
          ),
        }
      })
      return mutated ? { ...activity, turns } : activity
    }

    case 'complete': {
      const turns = activity.turns.map((t) =>
        t.isStreaming ? { ...t, isStreaming: false } : t,
      )
      return { ...activity, turns, isStreaming: false }
    }

    case 'error':
      // Sub-thread errors surface via the parent's resolve flow (the
      // failed result lands on the parent's task() tool call). No need
      // to mirror them in the nested transcript.
      return activity
  }
}

function currentStreamingTurnIdx(turns: TurnEntry[]): number {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].isStreaming) return i
    return -1
  }
  return -1
}

/** Backfill a sub-thread's activity from its persisted messages
 * (returned by `/api/threads/:sub_thread_id/messages`). Used after
 * page refresh — live SSE for a finished sub-thread is gone, so we
 * reconstruct the activity from history.
 *
 * `historyTurns` is what `historyToEntries` returned for the sub-thread,
 * filtered to `TurnEntry`s only. We trust those over any partial
 * live state for the same sub-thread.
 */
export function backfillSubThread(
  map: SubThreadMap,
  link: { sub_thread_id: string; parent_thread_id: string; parent_tool_call_id: string },
  historyTurns: TurnEntry[],
  subagent: string | null,
): SubThreadMap {
  // Tag each turn with the sub-thread's threadId and matching parent.
  const turns: TurnEntry[] = historyTurns.map((t) => ({
    ...t,
    threadId: link.sub_thread_id,
  }))
  return {
    ...map,
    [link.sub_thread_id]: {
      subThreadId: link.sub_thread_id,
      parentThreadId: link.parent_thread_id,
      parentToolCallId: link.parent_tool_call_id,
      subagent,
      turns,
      isStreaming: false,
    },
  }
}
