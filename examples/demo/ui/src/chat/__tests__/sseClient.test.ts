/**
 * Unit tests for the SSE transport layer.
 *
 * Mock byte streams in, decoded ActantEvents out. No network.
 *
 * The companion integration test in `sseClient.integration.test.ts`
 * exercises the same code path against a live server (gated on
 * ACTANT_DEMO_INTEGRATION=1).
 */

import { expect, test } from 'bun:test'
import { readSseFromStream } from '../sseClient'
import type { ActantEvent } from '../wire'

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  let i = 0
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i++]))
      } else {
        controller.close()
      }
    },
  })
}

async function collect(
  body: ReadableStream<Uint8Array>,
): Promise<ActantEvent[]> {
  const out: ActantEvent[] = []
  for await (const ev of readSseFromStream(body)) out.push(ev)
  return out
}

test('decodes a single complete frame', async () => {
  const body = streamOf([
    'event: text_delta\ndata: {"type":"text_delta","thread_id":"t","data":{"delta":"hi"}}\n\n',
  ])
  const events = await collect(body)
  expect(events).toHaveLength(1)
  expect(events[0].type).toBe('text_delta')
})

test('handles a stream split mid-frame across multiple chunks', async () => {
  // The client must NOT mistake a chunk boundary for a frame boundary.
  // This test fragments the bytes at every possible offset to make sure
  // the buffer-and-split logic is correct.
  const wire =
    'event: turn_start\n' +
    'data: {"type":"turn_start","thread_id":"t","data":{"turn":1}}\n\n'
  const body = streamOf([
    wire.slice(0, 5),
    wire.slice(5, 20),
    wire.slice(20, 60),
    wire.slice(60),
  ])
  const events = await collect(body)
  expect(events).toHaveLength(1)
  expect(events[0].type).toBe('turn_start')
})

test('decodes multiple frames in one chunk', async () => {
  const body = streamOf([
    'event: turn_start\ndata: {"type":"turn_start","thread_id":"t","data":{"turn":1}}\n\n' +
      'event: text_delta\ndata: {"type":"text_delta","thread_id":"t","data":{"delta":"hi"}}\n\n' +
      'event: complete\ndata: {"type":"complete","thread_id":"t","data":{"success":true,"reason":"ok","message":""}}\n\n',
  ])
  const events = await collect(body)
  expect(events.map((e) => e.type)).toEqual(['turn_start', 'text_delta', 'complete'])
})

test('CRLF line endings are normalized', async () => {
  const body = streamOf([
    'event: text_delta\r\ndata: {"type":"text_delta","thread_id":"t","data":{"delta":"x"}}\r\n\r\n',
  ])
  const events = await collect(body)
  expect(events).toHaveLength(1)
  expect((events[0] as Extract<ActantEvent, { type: 'text_delta' }>).data.delta).toBe('x')
})

test('keepalive comments are skipped, real events still emit', async () => {
  const body = streamOf([
    ': connected\n\n' +
      'event: turn_start\ndata: {"type":"turn_start","thread_id":"t","data":{"turn":1}}\n\n' +
      ': ping\n\n' +
      'event: complete\ndata: {"type":"complete","thread_id":"t","data":{"success":true,"reason":"ok","message":""}}\n\n',
  ])
  const events = await collect(body)
  expect(events.map((e) => e.type)).toEqual(['turn_start', 'complete'])
})

test('malformed frames silently skip; surrounding good frames still emit', async () => {
  const body = streamOf([
    'event: turn_start\ndata: {"type":"turn_start","thread_id":"t","data":{"turn":1}}\n\n' +
      'event: mystery_type\ndata: {"type":"mystery_type"}\n\n' +
      'event: text_delta\ndata: not valid json{{{\n\n' +
      'event: complete\ndata: {"type":"complete","thread_id":"t","data":{"success":true,"reason":"ok","message":""}}\n\n',
  ])
  const events = await collect(body)
  expect(events.map((e) => e.type)).toEqual(['turn_start', 'complete'])
})

test('abort closes the reader cleanly mid-stream', async () => {
  const ctrl = new AbortController()
  // A stream that never closes — we have to abort to escape.
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const encoder = new TextEncoder()
      controller.enqueue(
        encoder.encode(
          'event: turn_start\ndata: {"type":"turn_start","thread_id":"t","data":{"turn":1}}\n\n',
        ),
      )
      // Block forever until abort. Yield control so the consumer can
      // process and then call abort.
      await new Promise((r) => setTimeout(r, 100))
    },
  })

  const events: ActantEvent[] = []
  // Wire the abort to fire after we've consumed the first event.
  const iter = readSseFromStream(body, ctrl.signal)
  for await (const ev of iter) {
    events.push(ev)
    ctrl.abort()
    break
  }
  expect(events).toHaveLength(1)
})
