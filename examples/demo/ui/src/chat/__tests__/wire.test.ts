/**
 * Unit tests for wire-level SSE frame parsing + envelope decoding.
 *
 * These tests use the EXACT byte shapes the server emits via
 * sse-starlette + PublishingThreadHooks. If the server's output
 * format changes, this is the test file that breaks first.
 */

import { expect, test } from 'bun:test'
import { decodeActantFrame, parseSseFrame, type ActantEvent } from '../wire'

test('parseSseFrame: simple event + data', () => {
  const frame = parseSseFrame('event: text_delta\ndata: {"x":1}')
  expect(frame.event).toBe('text_delta')
  expect(frame.data).toBe('{"x":1}')
})

test('parseSseFrame: tolerates "data:" with or without leading space', () => {
  const a = parseSseFrame('event: turn_start\ndata:{"x":1}')
  const b = parseSseFrame('event: turn_start\ndata: {"x":1}')
  expect(a.data).toBe('{"x":1}')
  expect(b.data).toBe('{"x":1}')
})

test('parseSseFrame: comment-only frame yields null event + null data', () => {
  const frame = parseSseFrame(': connected')
  expect(frame.event).toBeNull()
  expect(frame.data).toBeNull()
})

test('parseSseFrame: multi-line data is joined with \\n', () => {
  const frame = parseSseFrame('event: text_delta\ndata: line one\ndata: line two')
  expect(frame.data).toBe('line one\nline two')
})

test('parseSseFrame: ignores unknown fields (id:, retry:)', () => {
  const frame = parseSseFrame(
    'id: 42\nevent: turn_start\nretry: 500\ndata: {"turn":1}',
  )
  expect(frame.event).toBe('turn_start')
  expect(frame.data).toBe('{"turn":1}')
})

test('decodeActantFrame: round-trip on a real text_delta event', () => {
  const raw =
    'event: text_delta\n' +
    'data: {"type":"text_delta","thread_id":"t_1","data":{"delta":"hello"}}'
  const decoded = decodeActantFrame(parseSseFrame(raw))
  expect(decoded).not.toBeNull()
  const d = decoded as Extract<ActantEvent, { type: 'text_delta' }>
  expect(d.type).toBe('text_delta')
  expect(d.thread_id).toBe('t_1')
  expect(d.data.delta).toBe('hello')
})

test('decodeActantFrame: assistant_message with tool_calls + thought_summary', () => {
  const envelope = {
    type: 'assistant_message',
    thread_id: 't_1',
    data: {
      content: 'final',
      thought_summary: 'thinking',
      tool_calls: [
        { id: 'tc_1', function: { name: 'get_current_time', arguments: '{}' } },
      ],
    },
  }
  const raw = `event: assistant_message\ndata: ${JSON.stringify(envelope)}`
  const decoded = decodeActantFrame(parseSseFrame(raw))
  expect(decoded).not.toBeNull()
  const d = decoded as Extract<ActantEvent, { type: 'assistant_message' }>
  expect(d.data.content).toBe('final')
  expect(d.data.tool_calls).toHaveLength(1)
  expect(d.data.tool_calls[0].function.name).toBe('get_current_time')
})

test('decodeActantFrame: sub-thread events carry parent_thread_id + subagent', () => {
  const envelope = {
    type: 'text_delta',
    thread_id: 'sub_1',
    data: { delta: 'sub work' },
    parent_thread_id: 'thread_parent',
    parent_tool_call_id: 'tc_task_1',
    subagent: 'researcher',
  }
  const raw = `event: text_delta\ndata: ${JSON.stringify(envelope)}`
  const decoded = decodeActantFrame(parseSseFrame(raw))
  expect(decoded).not.toBeNull()
  expect(decoded?.parent_thread_id).toBe('thread_parent')
  expect(decoded?.parent_tool_call_id).toBe('tc_task_1')
  expect(decoded?.subagent).toBe('researcher')
})

test('decodeActantFrame: unknown event type yields null', () => {
  const raw =
    'event: mystery\ndata: {"type":"mystery","thread_id":"t_1","data":{}}'
  expect(decodeActantFrame(parseSseFrame(raw))).toBeNull()
})

test('decodeActantFrame: malformed JSON yields null', () => {
  const raw = 'event: text_delta\ndata: not valid json{{{'
  expect(decodeActantFrame(parseSseFrame(raw))).toBeNull()
})

test('decodeActantFrame: SSE event name must match envelope type field', () => {
  // If the server sends `event: text_delta` but the envelope's type
  // is `assistant_message`, that's a server bug and we should drop
  // the frame rather than trusting it.
  const raw =
    'event: text_delta\n' +
    'data: {"type":"assistant_message","thread_id":"t_1","data":{}}'
  expect(decodeActantFrame(parseSseFrame(raw))).toBeNull()
})

test('decodeActantFrame: keepalive (no event) yields null', () => {
  expect(decodeActantFrame(parseSseFrame(': connected'))).toBeNull()
})
