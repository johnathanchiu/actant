/**
 * history.ts: PersistedMessage[] → ConsoleEntry[].
 */

import { expect, test } from 'bun:test'
import type { PersistedMessage } from '../api'
import { historyToEntries } from '../history'
import { reduce } from '../reducer'
import type { ConsoleEntry, ToolCallEntry, TurnEntry, UserEntry } from '../state'
import type { ActantEvent } from '../wire'

const TID = 't_1'

function applyLive(events: ActantEvent[]): ConsoleEntry[] {
  let entries: ConsoleEntry[] = []
  for (const event of events) entries = reduce(entries, event)
  return entries
}

function semanticEntries(entries: ConsoleEntry[]) {
  return entries.map((entry) => {
    if (entry.kind === 'user') {
      return { kind: entry.kind, threadId: entry.threadId, text: entry.text }
    }
    if (entry.kind === 'event') {
      return {
        kind: entry.kind,
        threadId: entry.threadId,
        level: entry.level,
        text: entry.text,
      }
    }
    return {
      kind: entry.kind,
      threadId: entry.threadId,
      text: entry.text,
      thinking: entry.thinking,
      isStreaming: entry.isStreaming,
      toolCalls: entry.toolCalls.map(semanticToolCall),
    }
  })
}

function semanticToolCall(call: ToolCallEntry) {
  return {
    id: call.id,
    name: call.name,
    argsText: call.argsText,
    args: call.args,
    state: call.state,
    result: call.result,
    error: call.error,
    waitPrompt: call.waitPrompt,
    waitKind: call.waitKind,
    waitOptions: call.waitOptions,
    subThreadId: call.subThreadId,
    subagent: call.subagent,
  }
}

test('user → assistant: one user + one finalized turn', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'hello!', thought_summary: 'greeted' },
  ]
  const entries = historyToEntries(persisted, TID)
  expect(entries).toHaveLength(2)
  expect(entries[0].kind).toBe('user')
  expect((entries[0] as UserEntry).text).toBe('hi')
  const turn = entries[1] as TurnEntry
  expect(turn.text).toBe('hello!')
  expect(turn.thinking).toBe('greeted')
  expect(turn.isStreaming).toBe(false)
  expect(turn.toolCalls).toHaveLength(0)
})

test('user → assistant(tool_use) → tool result → assistant(text): TWO turns in order', () => {
  // Two assistant messages = two LLM rounds = two TurnEntries, just
  // like the live SSE stream produces. The first turn carries the
  // tool call (with its result attached); the second carries the
  // answer text. Collapsing them used to scramble the render order
  // — see the get_current_time refresh bug.
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'time?' },
    {
      role: 'assistant',
      content: '',
      thought_summary: 'need time',
      tool_calls: [{ id: 'tc_1', function: { name: 'get_time', arguments: '{}' } }],
    },
    {
      role: 'tool',
      content: '{"result": "2026-01-01T00:00:00", "tool_call_id": "tc_1"}',
      tool_call_id: 'tc_1',
      name: 'get_time',
    },
    { role: 'assistant', content: 'The time is 2026-01-01.' },
  ]
  const entries = historyToEntries(persisted, TID)
  // user + turn(tool_use) + turn(text)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn', 'turn'])

  const toolTurn = entries[1] as TurnEntry
  expect(toolTurn.text).toBe('')
  expect(toolTurn.thinking).toBe('need time')
  expect(toolTurn.toolCalls).toHaveLength(1)
  expect(toolTurn.toolCalls[0].name).toBe('get_time')
  expect(toolTurn.toolCalls[0].state).toBe('ok')
  expect(toolTurn.toolCalls[0].result).toContain('2026-01-01')

  const textTurn = entries[2] as TurnEntry
  expect(textTurn.text).toBe('The time is 2026-01-01.')
  expect(textTurn.toolCalls).toHaveLength(0)
})

test('failed tool result: state="error", error message extracted on tool turn', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'fetch please' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [{ id: 'tc_x', function: { name: 'fetch_url', arguments: '{"url":"x"}' } }],
    },
    {
      role: 'tool',
      content: '{"error": "connection refused", "tool_call_id": "tc_x"}',
      tool_call_id: 'tc_x',
      name: 'fetch_url',
    },
    { role: 'assistant', content: "couldn't fetch." },
  ]
  const entries = historyToEntries(persisted, TID)
  // user + turn(tool_use w/ error) + turn(apology text)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn', 'turn'])
  const toolTurn = entries[1] as TurnEntry
  expect(toolTurn.toolCalls[0].state).toBe('error')
  expect(toolTurn.toolCalls[0].error).toBe('connection refused')
  expect((entries[2] as TurnEntry).text).toBe("couldn't fetch.")
})

test('two exchanges: user → assistant, user → assistant produces 4 entries', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'q1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'q2' },
    { role: 'assistant', content: 'a2' },
  ]
  const entries = historyToEntries(persisted, TID)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn', 'user', 'turn'])
  expect((entries[1] as TurnEntry).text).toBe('a1')
  expect((entries[3] as TurnEntry).text).toBe('a2')
})

test('orphan tool message (no prior assistant) is silently dropped', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'hi' },
    {
      role: 'tool',
      content: '{"result": "stranded"}',
      tool_call_id: 'tc_unknown',
      name: 'mystery',
    },
    { role: 'assistant', content: 'hi back' },
  ]
  const entries = historyToEntries(persisted, TID)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn'])
  expect((entries[1] as TurnEntry).text).toBe('hi back')
})

test('multi-tool-call assistant: both tool messages attach to the tool turn; text is a SEPARATE turn', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'do two things' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'tc_a', function: { name: 'foo', arguments: '{}' } },
        { id: 'tc_b', function: { name: 'bar', arguments: '{}' } },
      ],
    },
    {
      role: 'tool',
      content: '{"result": "foo result", "tool_call_id": "tc_a"}',
      tool_call_id: 'tc_a',
      name: 'foo',
    },
    {
      role: 'tool',
      content: '{"result": "bar result", "tool_call_id": "tc_b"}',
      tool_call_id: 'tc_b',
      name: 'bar',
    },
    { role: 'assistant', content: 'done both' },
  ]
  const entries = historyToEntries(persisted, TID)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn', 'turn'])
  const toolTurn = entries[1] as TurnEntry
  expect(toolTurn.toolCalls).toHaveLength(2)
  expect(toolTurn.toolCalls[0].result).toContain('foo result')
  expect(toolTurn.toolCalls[1].result).toContain('bar result')
  expect(toolTurn.text).toBe('')

  const textTurn = entries[2] as TurnEntry
  expect(textTurn.text).toBe('done both')
  expect(textTurn.toolCalls).toHaveLength(0)
})

test('all rehydrated turns are non-streaming + carry threadId', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'q' },
    { role: 'assistant', content: 'a' },
  ]
  const entries = historyToEntries(persisted, TID)
  for (const e of entries) {
    if (e.kind === 'turn') {
      expect(e.isStreaming).toBe(false)
      expect(e.threadId).toBe(TID)
    }
    if (e.kind === 'user') expect(e.threadId).toBe(TID)
  }
})

test('empty messages array → empty entries array', () => {
  expect(historyToEntries([], TID)).toEqual([])
})

test('get_current_time regression: tool turn rendered BEFORE answer turn, never collapsed', () => {
  // Direct regression for thread_v3sk8go3-style bug. Before the fix,
  // the answer text and the tool call merged into one card with the
  // text appearing ABOVE the tool call — visually scrambling the
  // actual cause-then-effect order. Now they're two cards in order.
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'sorry can you check again' },
    {
      role: 'assistant',
      content: '',
      thought_summary: 'I should call get_current_time again to get the updated current time.',
      tool_calls: [
        { id: 'tc_time', function: { name: 'get_current_time', arguments: '{}' } },
      ],
    },
    {
      role: 'tool',
      content: '{"result": "2026-05-20T03:37:48+00:00", "tool_call_id": "tc_time"}',
      tool_call_id: 'tc_time',
      name: 'get_current_time',
    },
    {
      role: 'assistant',
      content: 'The current UTC time is 3:37 AM on Tuesday, May 20th, 2026.',
    },
  ]
  const entries = historyToEntries(persisted, TID)
  expect(entries.map((e) => e.kind)).toEqual(['user', 'turn', 'turn'])

  // Tool turn comes FIRST, has the tool call + thinking but no answer text.
  const toolTurn = entries[1] as TurnEntry
  expect(toolTurn.thinking).toContain('get_current_time')
  expect(toolTurn.toolCalls).toHaveLength(1)
  expect(toolTurn.toolCalls[0].state).toBe('ok')
  expect(toolTurn.text).toBe('')

  // Answer turn comes SECOND, has the text but no tool calls.
  const answerTurn = entries[2] as TurnEntry
  expect(answerTurn.text).toContain('3:37 AM')
  expect(answerTurn.toolCalls).toHaveLength(0)
  expect(answerTurn.thinking).toBe('')
})

test('history rehydration matches live reducer shape for tool turn plus answer turn', () => {
  const persisted: PersistedMessage[] = [
    { role: 'user', content: 'time?' },
    {
      role: 'assistant',
      content: '',
      thought_summary: 'need time',
      tool_calls: [{ id: 'tc_1', function: { name: 'get_time', arguments: '{}' } }],
    },
    {
      role: 'tool',
      content: '{"result": "2026-01-01T00:00:00", "tool_call_id": "tc_1"}',
      tool_call_id: 'tc_1',
      name: 'get_time',
    },
    { role: 'assistant', content: 'The time is 2026-01-01.' },
  ]
  const live = applyLive([
    {
      type: 'turn_start',
      thread_id: TID,
      data: { turn: 1, turn_uid: 'live_1' },
    },
    {
      type: 'assistant_message',
      thread_id: TID,
      data: {
        content: '',
        thought_summary: 'need time',
        tool_calls: [{ id: 'tc_1', function: { name: 'get_time', arguments: '{}' } }],
      },
    },
    {
      type: 'tool_result',
      thread_id: TID,
      data: {
        tool_call_id: 'tc_1',
        output: '{"result": "2026-01-01T00:00:00", "tool_call_id": "tc_1"}',
        error: null,
      },
    },
    {
      type: 'turn_start',
      thread_id: TID,
      data: { turn: 2, turn_uid: 'live_2' },
    },
    {
      type: 'assistant_message',
      thread_id: TID,
      data: {
        content: 'The time is 2026-01-01.',
        thought_summary: null,
        tool_calls: [],
      },
    },
  ])
  const history = historyToEntries(persisted, TID).filter((e) => e.kind !== 'user')

  expect(semanticEntries(history)).toEqual(semanticEntries(live))
})
