/**
 * Per-turn + per-tool-call event-application helpers.
 *
 * Pure functions, shared between the top-level reducer (`reducer.ts`)
 * and the sub-thread reducer (`subThreads.ts`). Extracted because the
 * logic for "fold a single event into a single TurnEntry / ToolCallEntry"
 * is identical regardless of which thread scope the event belongs to.
 *
 * Top-level vs sub-thread differs only in WHERE the turn lives
 * (`entries: ConsoleEntry[]` vs `activity.turns: TurnEntry[]`) and
 * HOW the current turn is found (most-recent-streaming on the right
 * thread). The MUTATION of a turn given an event is the same.
 */

import type {
  ToolCallEntry,
  ToolCallState,
  TurnEntry,
} from './state'
import type { ActantEvent } from './wire'

/** Apply one event to one turn. Returns a new TurnEntry. The event's
 * `parent_thread_id`/`thread_id` fields are IGNORED — the caller is
 * responsible for routing the event to the right turn. */
export function applyEventToTurn(turn: TurnEntry, event: ActantEvent): TurnEntry {
  switch (event.type) {
    case 'text_delta':
      return { ...turn, text: turn.text + event.data.delta }
    case 'thinking_delta':
      return { ...turn, thinking: turn.thinking + event.data.delta }
    case 'tool_call_start': {
      if (turn.toolCalls.some((c) => c.id === event.data.tool_call_id)) return turn
      const call: ToolCallEntry = {
        id: event.data.tool_call_id,
        name: event.data.name,
        argsText: '',
        args: null,
        state: 'streaming',
        result: null,
        error: null,
        waitPrompt: null,
        waitKind: null,
        waitOptions: null,
        subThreadId: null,
        subagent: null,
        startedAt: Date.now(),
      }
      return { ...turn, toolCalls: [...turn.toolCalls, call] }
    }
    case 'tool_call_args_delta':
      return {
        ...turn,
        toolCalls: turn.toolCalls.map((c) =>
          c.id === event.data.tool_call_id
            ? { ...c, argsText: c.argsText + event.data.delta }
            : c,
        ),
      }
    case 'tool_call_args_complete':
      return {
        ...turn,
        toolCalls: turn.toolCalls.map((c) =>
          c.id === event.data.tool_call_id && c.state === 'streaming'
            ? { ...c, state: 'pending' as ToolCallState }
            : c,
        ),
      }
    case 'tool_call':
      return {
        ...turn,
        toolCalls: turn.toolCalls.map((c) =>
          c.id === event.data.tool_call_id
            ? {
                ...c,
                args: event.data.args,
                state:
                  c.state === 'streaming' ? ('pending' as ToolCallState) : c.state,
              }
            : c,
        ),
      }
    case 'assistant_message': {
      const finalContent =
        typeof event.data.content === 'string' ? event.data.content : ''
      const reconciled: ToolCallEntry[] = event.data.tool_calls.map((tc) => {
        const existing = turn.toolCalls.find((c) => c.id === tc.id)
        return {
          ...(existing ??
            ({
              id: tc.id,
              name: tc.function.name,
              argsText: '',
              args: null,
              state: 'pending',
              result: null,
              error: null,
              waitPrompt: null,
              waitKind: null,
              waitOptions: null,
              subThreadId: null,
              subagent: null,
              startedAt: Date.now(),
            } as ToolCallEntry)),
          argsText: tc.function.arguments,
          args: safeParseObject(tc.function.arguments),
          state:
            existing?.state === 'streaming'
              ? 'pending'
              : (existing?.state ?? 'pending'),
        }
      })
      return {
        ...turn,
        text: finalContent || turn.text,
        thinking: event.data.thought_summary ?? turn.thinking,
        toolCalls: reconciled.length ? reconciled : turn.toolCalls,
        isStreaming: false,
      }
    }
    default:
      return turn
  }
}

/** Apply a tool_result / tool_waiting / tool_resolved event to a
 * single ToolCallEntry. The caller is responsible for finding the
 * right tool call by id and rebuilding the containing entry. */
export function applyEventToToolCall(
  call: ToolCallEntry,
  event: ActantEvent,
): ToolCallEntry {
  switch (event.type) {
    case 'tool_result':
      return {
        ...call,
        state: event.data.error ? 'error' : 'ok',
        result: event.data.output,
        error: event.data.error,
      }
    case 'tool_waiting': {
      const rawOptions = (event.data.wait_payload as { options?: unknown } | undefined)
        ?.options
      const options =
        Array.isArray(rawOptions)
          ? rawOptions.filter((o): o is string => typeof o === 'string')
          : null
      return {
        ...call,
        state: 'waiting',
        waitPrompt: event.data.prompt,
        waitKind: event.data.wait_kind ?? null,
        waitOptions: options && options.length > 0 ? options : null,
      }
    }
    case 'tool_resolved':
      return {
        ...call,
        state: 'resolved',
        result: event.data.output,
      }
    default:
      return call
  }
}

function safeParseObject(text: string): Record<string, unknown> | null {
  if (!text) return null
  try {
    const parsed = JSON.parse(text) as unknown
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}
