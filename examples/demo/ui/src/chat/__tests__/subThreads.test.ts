/**
 * Sub-thread reducer + backfill tests.
 */

import { expect, test } from 'bun:test'
import { backfillSubThread, reduceSubThread, type SubThreadMap } from '../subThreads'
import type { TurnEntry } from '../state'
import type { ActantEvent } from '../wire'

const PARENT = 't_parent'
const SUB = 'sub_1'
const PARENT_TC = 'tc_task_1'

const subTs = (turn_uid = 'sub_tu_1'): ActantEvent => ({
  type: 'turn_start',
  thread_id: SUB,
  parent_thread_id: PARENT,
  parent_tool_call_id: PARENT_TC,
  subagent: 'researcher',
  data: { turn: 1, turn_uid },
})
const subTd = (delta: string): ActantEvent => ({
  type: 'text_delta',
  thread_id: SUB,
  parent_thread_id: PARENT,
  parent_tool_call_id: PARENT_TC,
  subagent: 'researcher',
  data: { delta },
})
const subAm = (content: string): ActantEvent => ({
  type: 'assistant_message',
  thread_id: SUB,
  parent_thread_id: PARENT,
  parent_tool_call_id: PARENT_TC,
  subagent: 'researcher',
  data: { content, thought_summary: null, tool_calls: [] },
})
const subComplete = (success = true): ActantEvent => ({
  type: 'complete',
  thread_id: SUB,
  parent_thread_id: PARENT,
  parent_tool_call_id: PARENT_TC,
  subagent: 'researcher',
  data: { success, reason: success ? 'completed' : 'failed', message: '' },
})

function applyAll(events: ActantEvent[]): SubThreadMap {
  let map: SubThreadMap = {}
  for (const ev of events) map = reduceSubThread(map, ev, PARENT)
  return map
}

test('turn_start creates an activity with one streaming turn', () => {
  const map = applyAll([subTs()])
  expect(map[SUB]).toBeDefined()
  expect(map[SUB].parentThreadId).toBe(PARENT)
  expect(map[SUB].parentToolCallId).toBe(PARENT_TC)
  expect(map[SUB].subagent).toBe('researcher')
  expect(map[SUB].turns).toHaveLength(1)
  expect(map[SUB].turns[0].isStreaming).toBe(true)
})

test('text_delta accumulates into the current sub-thread turn', () => {
  const map = applyAll([subTs(), subTd('hello '), subTd('world')])
  expect(map[SUB].turns[0].text).toBe('hello world')
})

test('complete marks the activity AND its current turn not streaming', () => {
  const map = applyAll([subTs(), subTd('done'), subAm('done'), subComplete()])
  expect(map[SUB].isStreaming).toBe(false)
  expect(map[SUB].turns[0].isStreaming).toBe(false)
  expect(map[SUB].turns[0].text).toBe('done')
})

test('events with no parent_thread_id are IGNORED', () => {
  const wrongEvent: ActantEvent = {
    type: 'text_delta',
    thread_id: 't_random',
    data: { delta: 'noop' },
  }
  const map = applyAll([subTs(), wrongEvent])
  expect(map[SUB].turns[0].text).toBe('')  // wrongEvent didn't apply
})

test('events with non-matching parent_thread_id are IGNORED (defensive)', () => {
  const otherParentEvent: ActantEvent = {
    type: 'text_delta',
    thread_id: 'sub_x',
    parent_thread_id: 't_someone_else',
    parent_tool_call_id: 'tc_x',
    subagent: 'other',
    data: { delta: 'wrong parent' },
  }
  const map = applyAll([subTs(), subTd('mine'), otherParentEvent])
  expect(map[SUB].turns[0].text).toBe('mine')
  expect(map['sub_x']).toBeUndefined()
})

test('delta before turn_start auto-creates a turn (defensive)', () => {
  // Some servers may not deliver turn_start reliably; the reducer
  // shouldn't crash, just synthesize.
  const map = applyAll([subTd('orphaned')])
  expect(map[SUB].turns).toHaveLength(1)
  expect(map[SUB].turns[0].text).toBe('orphaned')
})

test('multiple sub-threads with different sub_thread_ids coexist', () => {
  const ev = (sub: string): ActantEvent => ({
    type: 'turn_start',
    thread_id: sub,
    parent_thread_id: PARENT,
    parent_tool_call_id: `tc_${sub}`,
    subagent: 'researcher',
    data: { turn: 1, turn_uid: `tu_${sub}` },
  })
  const map = applyAll([ev('sub_a'), ev('sub_b')])
  expect(map['sub_a']).toBeDefined()
  expect(map['sub_b']).toBeDefined()
  expect(map['sub_a'].parentToolCallId).toBe('tc_sub_a')
  expect(map['sub_b'].parentToolCallId).toBe('tc_sub_b')
})

// ─── backfillSubThread ──────────────────────────────────────────────

test('backfillSubThread populates an activity from history turns', () => {
  const turns: TurnEntry[] = [
    {
      kind: 'turn',
      id: 't_hist_1',
      turnUid: 'tu_hist_1',
      threadId: 'wrong',  // will be overwritten by backfill
      text: 'historical text',
      thinking: '',
      toolCalls: [],
      isStreaming: false,
      timestamp: 0,
    },
  ]
  const map = backfillSubThread(
    {},
    {
      sub_thread_id: SUB,
      parent_thread_id: PARENT,
      parent_tool_call_id: PARENT_TC,
    },
    turns,
    'researcher',
  )
  expect(map[SUB].turns).toHaveLength(1)
  expect(map[SUB].turns[0].threadId).toBe(SUB)  // re-tagged
  expect(map[SUB].turns[0].text).toBe('historical text')
  expect(map[SUB].isStreaming).toBe(false)
})

test('backfillSubThread merges over existing entry (last write wins)', () => {
  let map: SubThreadMap = {
    [SUB]: {
      subThreadId: SUB,
      parentThreadId: PARENT,
      parentToolCallId: PARENT_TC,
      subagent: null,
      turns: [],
      isStreaming: true,
    },
  }
  const turns: TurnEntry[] = [
    {
      kind: 'turn',
      id: 't_hist_2',
      turnUid: 'tu_hist_2',
      threadId: SUB,
      text: 'fresh history',
      thinking: '',
      toolCalls: [],
      isStreaming: false,
      timestamp: 0,
    },
  ]
  map = backfillSubThread(
    map,
    {
      sub_thread_id: SUB,
      parent_thread_id: PARENT,
      parent_tool_call_id: PARENT_TC,
    },
    turns,
    'researcher',
  )
  expect(map[SUB].turns).toHaveLength(1)
  expect(map[SUB].turns[0].text).toBe('fresh history')
  expect(map[SUB].subagent).toBe('researcher')
})
