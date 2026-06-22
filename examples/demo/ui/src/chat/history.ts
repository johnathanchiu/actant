/**
 * Persisted-messages -> ConsoleEntry converter.
 *
 * Reads what `/api/threads/:id/messages` returns and produces the
 * same `ConsoleEntry` shape the live reducer produces. Assistant and
 * tool messages are synthesized into reducer events, so refresh and
 * live SSE share the same turn/tool mutation logic.
 *
 * Pure. No fetching, no React.
 *
 * Pairing rule:
 *   - Each persisted `role=assistant` message becomes its own reducer
 *     `turn_start` + `assistant_message` pair, mirroring live SSE where
 *     every LLM round has a separate turn. `role=tool` messages then
 *     patch the matching tool call by id via the reducer.
 */

import type { PersistedMessage } from './api'
import { reduce } from './reducer'
import type { ConsoleEntry, UserEntry } from './state'
import type { ActantEvent } from './wire'

export function historyToEntries(
  messages: PersistedMessage[],
  threadId: string,
): ConsoleEntry[] {
  let entries: ConsoleEntry[] = []
  let counter = 0
  const nextId = (prefix: string) => {
    counter += 1
    return `${prefix}_hist_${counter}`
  }

  for (const msg of messages) {
    if (msg.role === 'user') {
      const entry: UserEntry = {
        kind: 'user',
        id: nextId('user'),
        threadId,
        text: stringContent(msg.content),
        timestamp: Date.now(),
      }
      entries = [...entries, entry]
      continue
    }

    if (msg.role === 'assistant') {
      const turnUid = nextId('turn')
      entries = reduce(entries, {
        type: 'turn_start',
        thread_id: threadId,
        data: { turn: counter, turn_uid: turnUid },
      })
      entries = reduce(entries, {
        type: 'assistant_message',
        thread_id: threadId,
        data: {
          content: stringContent(msg.content),
          thought_summary: msg.thought_summary ?? null,
          tool_calls: msg.tool_calls ?? [],
        },
      })
      continue
    }

    if (msg.role === 'tool' && msg.tool_call_id) {
      const resultText = stringContent(msg.content)
      const isError = looksLikeErrorPayload(resultText)
      const event: ActantEvent = {
        type: 'tool_result',
        thread_id: threadId,
        data: {
          tool_call_id: msg.tool_call_id,
          output: isError ? null : resultText,
          error: isError ? extractErrorMessage(resultText) : null,
        },
      }
      entries = reduce(entries, event)
    }
    // system messages: ignored for now.
  }

  return entries
}

// Helpers

function stringContent(content: PersistedMessage['content']): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        const text = (block as Record<string, unknown>).text
        return typeof text === 'string' ? text : ''
      })
      .filter(Boolean)
      .join('\n')
  }
  return ''
}

/** The actant message store records tool results as JSON-encoded
 * dicts. Failed tools produce `{"error": "...", "tool_call_id": "..."}`
 * and successful tools produce `{"result": "...", "tool_call_id": "..."}`.
 */
function looksLikeErrorPayload(text: string): boolean {
  if (!text || !text.startsWith('{')) return false
  try {
    const parsed = JSON.parse(text) as { error?: unknown }
    return typeof parsed.error === 'string' && parsed.error.length > 0
  } catch {
    return false
  }
}

function extractErrorMessage(text: string): string | null {
  try {
    const parsed = JSON.parse(text) as { error?: unknown }
    return typeof parsed.error === 'string' ? parsed.error : null
  } catch {
    return null
  }
}
