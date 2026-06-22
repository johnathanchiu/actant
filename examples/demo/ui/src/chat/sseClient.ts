/**
 * Fetch-streams SSE transport.
 *
 * Opens a persistent HTTP connection, reads the body as a stream,
 * splits on SSE frame boundaries (blank lines), parses each frame
 * via `wire.parseSseFrame`, decodes it via `wire.decodeActantFrame`,
 * and yields typed `ActantEvent`s.
 *
 * Uses the Fetch Streams API rather than the built-in `EventSource`
 * because:
 *   - `EventSource` had reliability issues in React 19 StrictMode
 *     in our earlier port — events would drop between mounts.
 *   - Fetch streams let us cleanly abort via AbortController.
 *   - The same code path is testable end-to-end in Node (the
 *     integration test exercises it against a live server).
 *
 * The function is a plain async generator. No React, no state, no
 * singletons. Callers own the AbortController and decide when to
 * cancel.
 */

import { decodeActantFrame, parseSseFrame, type ActantEvent } from './wire'

/** Open an SSE stream and yield decoded events.
 *
 * @param url       URL to GET. Must accept `text/event-stream`.
 * @param signal    AbortSignal. Aborting closes the underlying fetch.
 * @returns Async iterator of `ActantEvent`s. Yields nothing if the
 *          stream is empty; rethrows if `fetch` fails (after the
 *          response is received, malformed frames are silently
 *          skipped via `decodeActantFrame` returning null).
 */
export async function* openActantStream(
  url: string,
  signal: AbortSignal,
): AsyncGenerator<ActantEvent> {
  const resp = await fetch(url, {
    headers: { Accept: 'text/event-stream' },
    signal,
  })
  if (!resp.ok || !resp.body) {
    throw new Error(`SSE failed: ${resp.status} ${resp.statusText}`)
  }
  yield* readSseFromStream(resp.body, signal)
}

/** Split a byte stream into SSE frames and yield decoded events.
 * Exported separately so tests can feed in fake byte chunks without
 * a real fetch.
 */
export async function* readSseFromStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<ActantEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  try {
    for (;;) {
      if (signal?.aborted) break
      const { value, done } = await reader.read()
      if (done) break
      // Normalize CRLF → LF so the frame splitter works regardless
      // of whether the server emits one or the other.
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
      // SSE frames are terminated by a blank line (\n\n).
      while (true) {
        const sep = buf.indexOf('\n\n')
        if (sep < 0) break
        const raw = buf.slice(0, sep)
        buf = buf.slice(sep + 2)
        const decoded = decodeActantFrame(parseSseFrame(raw))
        if (decoded !== null) yield decoded
      }
    }
  } finally {
    // Best-effort cancel — releases the lock on the underlying body
    // so the connection can be torn down cleanly.
    await reader.cancel().catch(() => undefined)
  }
}
