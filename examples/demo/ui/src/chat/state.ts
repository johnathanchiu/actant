/**
 * Chat-surface entry types.
 *
 * These are the SHAPES THE UI RENDERS — they're derived from the
 * wire-level ActantEvents by the reducer, but they're not the same
 * thing. UI components only depend on these types; they have no
 * idea SSE exists.
 *
 * The split lets the reducer change its internal logic, or the wire
 * protocol change shape, without rippling into the components.
 */

export type ToolCallState =
  | 'streaming'
  | 'pending'
  | 'running'
  | 'waiting'
  | 'resolved'
  | 'ok'
  | 'error'

export type ToolCallEntry = {
  id: string
  name: string
  argsText: string
  args: Record<string, unknown> | null
  state: ToolCallState
  result: string | null
  error: string | null
  waitPrompt: string | null
  waitKind: string | null
  /** For multiple-choice `ask_user` calls, the option strings the
   * agent provided. DeferredPanel renders these as buttons. */
  waitOptions: string[] | null
  /** Populated for `task()` tool calls when the spawned sub-thread is
   * known (live: from sub-thread SSE events with matching
   * parent_tool_call_id; rehydration: from `/api/threads/:id/sub_threads`).
   * Components use this to render a NestedTranscript inside the
   * tool-call card. */
  subThreadId: string | null
  subagent: string | null
  startedAt: number
}

export type UserEntry = {
  kind: 'user'
  id: string
  threadId: string
  text: string
  timestamp: number
}

export type TurnEntry = {
  kind: 'turn'
  id: string
  /** Server-stamped per-turn uid. All events with the same turnUid
   * collapse into one TurnEntry — this is the dedup key the reducer
   * uses. */
  turnUid: string
  threadId: string
  text: string
  thinking: string
  toolCalls: ToolCallEntry[]
  isStreaming: boolean
  timestamp: number
}

export type EventEntry = {
  kind: 'event'
  id: string
  threadId: string
  level: 'info' | 'error'
  text: string
  timestamp: number
}

export type ConsoleEntry = UserEntry | TurnEntry | EventEntry
