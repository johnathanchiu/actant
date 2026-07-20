/**
 * Wire types for the actant demo server's SSE stream.
 *
 * Mirrors the JSON shape emitted by
 * ``actant.runtime.events.PublishingThreadHooks`` /
 * ``PublishingStreamListener`` — see
 * https://github.com/.../actant/runtime/events/lifecycle.py.
 *
 * Every event lands as an SSE frame of the form:
 *
 *   event: <type>
 *   data: <json envelope>
 *
 * Where ``<json envelope>`` is `{ type, thread_id, data, ... }`.
 * For sub-thread events on a parent's channel, the envelope ALSO
 * carries ``parent_thread_id``, ``parent_tool_call_id``, and
 * ``subagent`` — the publishing hooks stamp these when wired with
 * the coordinator primitives' `publishing_hooks_factory(..., registry)`.
 *
 * This module is pure types + a tiny SSE frame parser. No React, no
 * state, no network. Higher layers (sseClient, reducer) consume the
 * shapes defined here.
 */

export type ActantEnvelopeBase = {
  thread_id: string
  /** Set when this event came from a sub-thread; the FE attributes the
   * event to the parent's task() row instead of the parent's own
   * transcript. */
  parent_thread_id?: string
  parent_tool_call_id?: string
  subagent?: string | null
}

export type AssistantMessagePayload = {
  content: string | Array<Record<string, unknown>> | null
  thought_summary: string | null
  tool_calls: Array<{
    id: string
    function: { name: string; arguments: string }
  }>
}

export type ActantEvent =
  | (ActantEnvelopeBase & { type: 'text_delta'; data: { delta: string } })
  | (ActantEnvelopeBase & { type: 'thinking_delta'; data: { delta: string } })
  | (ActantEnvelopeBase & {
      type: 'tool_call_start'
      data: { tool_call_id: string; name: string }
    })
  | (ActantEnvelopeBase & {
      type: 'tool_call_args_delta'
      data: { tool_call_id: string; delta: string }
    })
  | (ActantEnvelopeBase & {
      type: 'tool_call_args_complete'
      data: { tool_call_id: string }
    })
  | (ActantEnvelopeBase & {
      type: 'turn_start'
      data: { turn: number; turn_id?: string; turn_uid?: string }
    })
  | (ActantEnvelopeBase & {
      type: 'assistant_message'
      data: AssistantMessagePayload
    })
  | (ActantEnvelopeBase & {
      type: 'tool_call'
      data: { tool_call_id: string; name: string; args: Record<string, unknown> }
    })
  | (ActantEnvelopeBase & {
      type: 'tool_result'
      data: {
        tool_call_id: string
        output: string | null
        error: string | null
        turn_id?: string
        turn_uid?: string
      }
    })
  | (ActantEnvelopeBase & {
      type: 'tool_waiting'
      data: {
        tool_call_id: string
        prompt: string
        wait_kind?: string
        wait_payload?: Record<string, unknown>
      }
    })
  | (ActantEnvelopeBase & {
      type: 'tool_resolved'
      data: { tool_call_id: string; output: string | null }
    })
  | (ActantEnvelopeBase & {
      type: 'complete'
      data: { success: boolean; reason: string; message: string }
    })
  | (ActantEnvelopeBase & { type: 'error'; data: { message: string } })

export type ActantEventType = ActantEvent['type']

export const ACTANT_EVENT_TYPES: readonly ActantEventType[] = [
  'text_delta',
  'thinking_delta',
  'tool_call_start',
  'tool_call_args_delta',
  'tool_call_args_complete',
  'turn_start',
  'assistant_message',
  'tool_call',
  'tool_result',
  'tool_waiting',
  'tool_resolved',
  'complete',
  'error',
] as const

const KNOWN_TYPES = new Set<string>(ACTANT_EVENT_TYPES)

// ─── SSE frame parser ───────────────────────────────────────────────

/** One parsed SSE frame. The transport layer accumulates lines until
 * a blank line, then calls `parseSseFrame` on the buffered text. */
export type SseFrame = {
  event: string | null
  data: string | null
}

/** Parse the raw text of one SSE frame (lines between two blank
 * lines, NOT INCLUDING the trailing blank). Returns the event name
 * and the raw data string. Comments (`:`-prefixed lines, used for
 * keepalive) yield `{event: null, data: null}` — the transport
 * layer should ignore those.
 *
 * Per SSE spec: a frame can have multiple `data:` lines; they're
 * concatenated with `\n` between them. Only the first `event:` line
 * is honored.
 */
export function parseSseFrame(raw: string): SseFrame {
  let event: string | null = null
  const dataParts: string[] = []
  for (const line of raw.split('\n')) {
    if (line === '' || line.startsWith(':')) continue
    if (event === null && line.startsWith('event:')) {
      event = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      dataParts.push(line.slice(5).replace(/^ /, ''))
      continue
    }
    // Unrecognised field (id:, retry:, etc) — silently ignored. Our
    // server doesn't emit these.
  }
  if (dataParts.length === 0) return { event, data: null }
  return { event, data: dataParts.join('\n') }
}

/** Decode one SSE frame into a typed ActantEvent. Returns null for:
 *   - keepalive comments (no event or data)
 *   - unrecognised event types
 *   - malformed JSON
 *
 * Throws nothing — caller decides whether to log / skip / surface.
 */
export function decodeActantFrame(frame: SseFrame): ActantEvent | null {
  if (frame.event === null || frame.data === null) return null
  if (!KNOWN_TYPES.has(frame.event)) return null
  try {
    const parsed = JSON.parse(frame.data) as ActantEvent
    // Quick sanity check: the inner `type` field of the envelope MUST
    // match the SSE event name. If they differ, the server is sending
    // mismatched frames and we shouldn't trust the payload.
    if ((parsed as { type?: string }).type !== frame.event) return null
    return parsed
  } catch {
    return null
  }
}
