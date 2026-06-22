/**
 * Reducer tests. Top-level events only; sub-thread routing is the
 * subThreads.ts layer's responsibility.
 */

import { expect, test } from 'bun:test'
import { reduce } from '../reducer'
import type { ConsoleEntry, TurnEntry } from '../state'
import type { ActantEvent } from '../wire'

const TID = 't_1'

function applyAll(events: ActantEvent[]): ConsoleEntry[] {
  let s: ConsoleEntry[] = []
  for (const e of events) s = reduce(s, e)
  return s
}

// Event factories (concise).
const ts = (turn_uid = 'tu_1'): ActantEvent => ({
  type: 'turn_start',
  thread_id: TID,
  data: { turn: 1, turn_uid },
})
const td = (delta: string): ActantEvent => ({
  type: 'text_delta',
  thread_id: TID,
  data: { delta },
})
const thd = (delta: string): ActantEvent => ({
  type: 'thinking_delta',
  thread_id: TID,
  data: { delta },
})
const tcs = (id: string, name: string): ActantEvent => ({
  type: 'tool_call_start',
  thread_id: TID,
  data: { tool_call_id: id, name },
})
const tcArgsComplete = (id: string): ActantEvent => ({
  type: 'tool_call_args_complete',
  thread_id: TID,
  data: { tool_call_id: id },
})
const tcCall = (id: string, name: string, args: Record<string, unknown>): ActantEvent => ({
  type: 'tool_call',
  thread_id: TID,
  data: { tool_call_id: id, name, args },
})
const tcResult = (id: string, output: string | null, error: string | null = null): ActantEvent => ({
  type: 'tool_result',
  thread_id: TID,
  data: { tool_call_id: id, output, error },
})
const tcWait = (id: string, prompt: string, kind = 'question'): ActantEvent => ({
  type: 'tool_waiting',
  thread_id: TID,
  data: { tool_call_id: id, prompt, wait_kind: kind },
})
const tcResolved = (id: string, output: string): ActantEvent => ({
  type: 'tool_resolved',
  thread_id: TID,
  data: { tool_call_id: id, output },
})
const am = (
  content: string,
  thought_summary: string | null = null,
  tool_calls: Array<{ id: string; function: { name: string; arguments: string } }> = [],
): ActantEvent => ({
  type: 'assistant_message',
  thread_id: TID,
  data: { content, thought_summary, tool_calls },
})
const cmplt = (success = true): ActantEvent => ({
  type: 'complete',
  thread_id: TID,
  data: { success, reason: success ? 'completed' : 'failed', message: '' },
})

// ─── basic flows ────────────────────────────────────────────────────

test('one turn, text only: turn_start + deltas + assistant_message + complete → one finalized turn', () => {
  const entries = applyAll([
    ts(),
    td('hello '),
    td('world'),
    am('hello world'),
    cmplt(),
  ])
  const turns = entries.filter((e): e is TurnEntry => e.kind === 'turn')
  expect(turns).toHaveLength(1)
  expect(turns[0].text).toBe('hello world')
  expect(turns[0].isStreaming).toBe(false)
})

test('thinking_delta concats into the turn', () => {
  const entries = applyAll([
    ts(),
    thd('the user wants '),
    thd('a greeting'),
    am('hi', 'the user wants a greeting'),
    cmplt(),
  ])
  const turn = entries[0] as TurnEntry
  // After assistant_message, thinking is replaced by thought_summary.
  expect(turn.thinking).toBe('the user wants a greeting')
})

test('tool flow: turn_start → tool_call_start → args_complete → assistant_message → tool_call → tool_result', () => {
  const entries = applyAll([
    ts(),
    tcs('tc_1', 'get_time'),
    tcArgsComplete('tc_1'),
    am('', null, [{ id: 'tc_1', function: { name: 'get_time', arguments: '{}' } }]),
    tcCall('tc_1', 'get_time', {}),
    tcResult('tc_1', '2026-01-01T00:00:00'),
  ])
  const turn = entries[0] as TurnEntry
  expect(turn.toolCalls).toHaveLength(1)
  expect(turn.toolCalls[0].state).toBe('ok')
  expect(turn.toolCalls[0].result).toBe('2026-01-01T00:00:00')
})

test('deferred flow: tool_waiting → tool_resolved', () => {
  const entries = applyAll([
    ts(),
    tcs('tc_ask', 'ask_user'),
    tcArgsComplete('tc_ask'),
    am('', null, [{ id: 'tc_ask', function: { name: 'ask_user', arguments: '{"question":"color?"}' } }]),
    tcCall('tc_ask', 'ask_user', { question: 'color?' }),
    tcWait('tc_ask', 'color?', 'question'),
  ])
  const turn = entries[0] as TurnEntry
  const call = turn.toolCalls[0]
  expect(call.state).toBe('waiting')
  expect(call.waitPrompt).toBe('color?')
  expect(call.waitKind).toBe('question')
  // Then user replies and the runtime resolves:
  const after = reduce(entries, tcResolved('tc_ask', 'blue'))
  const resolvedCall = (after[0] as TurnEntry).toolCalls[0]
  expect(resolvedCall.state).toBe('resolved')
  expect(resolvedCall.result).toBe('blue')
})

test('multi-turn run: two turn_starts → two TurnEntries', () => {
  const entries = applyAll([
    ts('tu_a'),
    td('first turn'),
    am('first turn'),
    ts('tu_b'),
    td('second turn'),
    am('second turn'),
    cmplt(),
  ])
  const turns = entries.filter((e): e is TurnEntry => e.kind === 'turn')
  expect(turns).toHaveLength(2)
  expect(turns[0].text).toBe('first turn')
  expect(turns[1].text).toBe('second turn')
  // Both finalized after complete.
  expect(turns[0].isStreaming).toBe(false)
  expect(turns[1].isStreaming).toBe(false)
})

// ─── purity + idempotency ───────────────────────────────────────────

test('same turn_uid event applied twice is a no-op', () => {
  const seq: ActantEvent[] = [ts('tu_dup'), td('hello')]
  const once = applyAll(seq)
  // Re-applying turn_start with the same uid shouldn't create a second
  // turn or wipe the existing one.
  const twice = reduce(once, ts('tu_dup'))
  expect(twice).toEqual(once)
})

test('tool_call_start is idempotent for the same tool_call_id', () => {
  const entries = applyAll([
    ts(),
    tcs('tc_x', 'foo'),
    tcs('tc_x', 'foo'),  // duplicate
    tcs('tc_y', 'bar'),
  ])
  const turn = entries[0] as TurnEntry
  expect(turn.toolCalls.map((c) => c.id)).toEqual(['tc_x', 'tc_y'])
})

// ─── edge cases ─────────────────────────────────────────────────────

test('text_delta with no prior turn_start auto-creates a turn', () => {
  // Some servers may drop turn_start on flaky connections; the
  // reducer shouldn't crash, just synthesize a turn.
  const entries = applyAll([td('orphaned text')])
  const turns = entries.filter((e): e is TurnEntry => e.kind === 'turn')
  expect(turns).toHaveLength(1)
  expect(turns[0].text).toBe('orphaned text')
})

test('complete with success=false appends an error entry', () => {
  const entries = applyAll([
    ts(),
    td('partial'),
    cmplt(false),
  ])
  const errs = entries.filter((e) => e.kind === 'event')
  expect(errs).toHaveLength(1)
  expect((errs[0] as { level: string; text: string }).level).toBe('error')
})

test('sub-thread events (parent_thread_id set) are IGNORED at top level', () => {
  // The reducer only handles top-level events. Sub-thread routing is
  // the subThreads.ts layer's job. The reducer must not pollute the
  // top-level entries when a sub-thread event arrives.
  const subEvent: ActantEvent = {
    type: 'text_delta',
    thread_id: 'sub_1',
    parent_thread_id: 't_1',
    parent_tool_call_id: 'tc_task',
    data: { delta: 'sub work' },
  }
  const entries = applyAll([ts(), td('parent work'), subEvent])
  const turns = entries.filter((e): e is TurnEntry => e.kind === 'turn')
  expect(turns).toHaveLength(1)
  expect(turns[0].text).toBe('parent work')  // unchanged
})

test('back-to-back turn_starts: prior streaming turn auto-closes (no double pulse)', () => {
  // Regression for the "two gray pulse dots in back-to-back rows"
  // bug. If the server emits turn_start for T2 before T1's
  // assistant_message arrives (mid-turn error, dropped frame, batched
  // SSE chunks), without this guard T1 stays isStreaming=true
  // alongside T2, and both empty turns render their pulse dot
  // simultaneously. After the fix: T1 must be closed at the moment
  // T2 begins so at most one streaming turn per thread exists.
  const entries = applyAll([
    ts('tu_first'),
    ts('tu_second'),
  ])
  const turns = entries.filter((e): e is TurnEntry => e.kind === 'turn')
  expect(turns).toHaveLength(2)
  const streaming = turns.filter((t) => t.isStreaming)
  expect(streaming).toHaveLength(1)
  expect(streaming[0].turnUid).toBe('tu_second')
})

test('back-to-back turn_starts on DIFFERENT threads do NOT cross-close each other', () => {
  // The auto-close is scoped to the same threadId. A new turn on
  // thread A must not close a streaming turn on thread B.
  let entries: ConsoleEntry[] = []
  entries = reduce(entries, ts('tu_a'))
  entries = reduce(entries, { type: 'turn_start', thread_id: 'other', data: { turn: 1, turn_uid: 'tu_b' } })
  entries = reduce(entries, ts('tu_a2'))  // new turn on the original thread
  const byUid = (uid: string) =>
    entries.find((e) => e.kind === 'turn' && (e as TurnEntry).turnUid === uid) as TurnEntry
  // tu_a closed (replaced by tu_a2 on same thread), tu_b still streaming.
  expect(byUid('tu_a').isStreaming).toBe(false)
  expect(byUid('tu_a2').isStreaming).toBe(true)
  expect(byUid('tu_b').isStreaming).toBe(true)
})

test('complete only finalizes turns on the matching thread', () => {
  // Defensive: even though our SSE is per-thread today, the reducer
  // should still scope `complete` to the right thread_id.
  let entries: ConsoleEntry[] = []
  entries = reduce(entries, ts('tu_a'))
  entries = reduce(entries, { type: 'turn_start', thread_id: 'other', data: { turn: 1, turn_uid: 'tu_b' } })
  // Complete only `t_1`. The other thread's turn should stay streaming.
  entries = reduce(entries, cmplt())
  const turnA = entries.find((e) => e.kind === 'turn' && (e as TurnEntry).turnUid === 'tu_a') as TurnEntry
  const turnB = entries.find((e) => e.kind === 'turn' && (e as TurnEntry).turnUid === 'tu_b') as TurnEntry
  expect(turnA.isStreaming).toBe(false)
  expect(turnB.isStreaming).toBe(true)
})
