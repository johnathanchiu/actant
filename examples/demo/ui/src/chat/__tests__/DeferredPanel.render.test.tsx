/**
 * DeferredPanel rendering tests.
 *
 * Targets two bug classes the user hit:
 *   1. DeferredPanel falsely appearing when a `task()` call was in
 *      `waiting` state (sub-agent hadn't returned yet). DeferredPanel
 *      must ONLY show for user-input tools.
 *   2. DeferredPanel NOT appearing when a sub-agent's `ask_user`
 *      went to waiting (prompt #8). The panel was walking only top-
 *      level entries, missing waits inside `subThreads[].turns[]`.
 *
 * Invariants tested:
 *   - Renders nothing for empty entries.
 *   - Renders nothing when only a `task()` call is in `waiting` state.
 *   - Renders nothing when only non-waiting calls exist.
 *   - Renders for `ask_user` in waiting state; shows option buttons.
 *   - Renders for `request_approval`; shows Approve / Deny.
 *   - Clicking option / Approve / Deny calls onResolve with
 *     (threadId, callId, body).
 *   - Renders for a sub-thread's ask_user; uses the sub-thread id
 *     when resolving (regression for prompt #8).
 *   - Top-level waits beat sub-thread waits if both exist.
 *   - Shows the no-options fallback for ask_user without waitOptions.
 */

import { expect, mock, test } from 'bun:test'
import { act, fireEvent, render } from '@testing-library/react'
import { DeferredPanel } from '../DeferredPanel'
import type { ConsoleEntry, ToolCallEntry, TurnEntry } from '../state'
import type { SubThreadActivity, SubThreadMap } from '../subThreads'

const MAIN_TID = 't_main'
const SUB_TID = 'sub_x'

function call(overrides: Partial<ToolCallEntry> = {}): ToolCallEntry {
  return {
    id: 'tc_1',
    name: 'get_current_time',
    argsText: '{}',
    args: {},
    state: 'ok',
    result: 'ok',
    error: null,
    waitPrompt: null,
    waitKind: null,
    waitOptions: null,
    subThreadId: null,
    subagent: null,
    startedAt: 0,
    ...overrides,
  }
}

function turn(toolCalls: ToolCallEntry[], threadId: string = MAIN_TID): TurnEntry {
  return {
    kind: 'turn',
    id: 'turn_1',
    turnUid: 'tu_1',
    threadId,
    text: '',
    thinking: '',
    toolCalls,
    isStreaming: false,
    timestamp: 0,
  }
}

function subActivity(
  toolCalls: ToolCallEntry[],
  subThreadId: string = SUB_TID,
): SubThreadActivity {
  return {
    subThreadId,
    parentThreadId: MAIN_TID,
    parentToolCallId: 'tc_task',
    subagent: 'researcher',
    turns: [turn(toolCalls, subThreadId)],
    isStreaming: false,
  }
}

const noOp = async () => {}

// --- 1. no-render cases --------------------------------------------

test('renders nothing for empty entries + empty subThreads', () => {
  const { container } = render(
    <DeferredPanel entries={[]} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toBe('')
})

test('renders nothing when there are no waiting calls', () => {
  const entries: ConsoleEntry[] = [
    turn([call({ name: 'fetch_url', state: 'ok' })]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toBe('')
})

test('renders nothing when only a task() call is waiting', () => {
  const entries: ConsoleEntry[] = [
    turn([
      call({
        name: 'task',
        state: 'waiting',
        subagent: 'researcher',
        subThreadId: 'sub_1',
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toBe('')
})

test('renders nothing for unknown-tool calls in waiting state', () => {
  const entries: ConsoleEntry[] = [
    turn([
      call({ name: 'fetch_url', state: 'waiting', waitPrompt: 'wat' }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toBe('')
})

// --- 2. ask_user rendering -----------------------------------------

test('renders multi-choice buttons for ask_user with options', () => {
  const entries: ConsoleEntry[] = [
    turn([
      call({
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'What is your favorite color?',
        waitKind: 'multiple_choice',
        waitOptions: ['Red', 'Green', 'Blue'],
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  const text = container.textContent ?? ''
  expect(text.toLowerCase()).toContain('agent is asking')
  expect(text).toContain('What is your favorite color?')

  const optionButtons = Array.from(container.querySelectorAll('button')).filter(
    (b) =>
      b.textContent === 'Red' ||
      b.textContent === 'Green' ||
      b.textContent === 'Blue',
  )
  expect(optionButtons.length).toBe(3)
})

test('shows fallback text when ask_user has no waitOptions', () => {
  const entries: ConsoleEntry[] = [
    turn([
      call({
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'old persisted call without options',
        waitKind: null,
        waitOptions: null,
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toContain('No options available')
})

test('clicking an option invokes onResolve with (mainThreadId, callId, { answer })', async () => {
  const onResolve = mock(
    async (
      _threadId: string,
      _callId: string,
      _body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
    ) => {},
  )
  const entries: ConsoleEntry[] = [
    turn([
      call({
        id: 'tc_q',
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'pick',
        waitKind: 'multiple_choice',
        waitOptions: ['A', 'B'],
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={onResolve} />,
  )
  const optionA = Array.from(container.querySelectorAll('button')).find(
    (b) => b.textContent === 'A',
  )
  await act(async () => {
    fireEvent.click(optionA!)
  })

  expect(onResolve.mock.calls.length).toBe(1)
  const [threadId, callId, body] = onResolve.mock.calls[0]
  expect(threadId).toBe(MAIN_TID)
  expect(callId).toBe('tc_q')
  expect(body.answer).toBe('A')
})

// --- 3. request_approval rendering ---------------------------------

test('renders Approve / Deny for request_approval', () => {
  const entries: ConsoleEntry[] = [
    turn([
      call({
        name: 'request_approval',
        state: 'waiting',
        waitPrompt: 'OK to delete the production DB?',
        waitKind: 'approval',
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  const text = container.textContent ?? ''
  expect(text.toLowerCase()).toContain('approval needed')
  expect(text).toContain('OK to delete the production DB?')
  const buttons = Array.from(container.querySelectorAll('button'))
  expect(buttons.find((b) => b.textContent?.toLowerCase().includes('approve'))).toBeDefined()
  expect(buttons.find((b) => b.textContent?.toLowerCase().includes('deny'))).toBeDefined()
})

test('clicking Approve invokes onResolve with (mainThreadId, callId, { approved: true })', async () => {
  const onResolve = mock(
    async (
      _threadId: string,
      _callId: string,
      _body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
    ) => {},
  )
  const entries: ConsoleEntry[] = [
    turn([
      call({
        id: 'tc_a',
        name: 'request_approval',
        state: 'waiting',
        waitPrompt: 'do the thing?',
        waitKind: 'approval',
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={onResolve} />,
  )
  const approve = Array.from(container.querySelectorAll('button')).find((b) =>
    b.textContent?.toLowerCase().includes('approve'),
  )
  await act(async () => {
    fireEvent.click(approve!)
  })

  expect(onResolve.mock.calls.length).toBe(1)
  const [threadId, callId, body] = onResolve.mock.calls[0]
  expect(threadId).toBe(MAIN_TID)
  expect(callId).toBe('tc_a')
  expect(body.approved).toBe(true)
})

test('clicking Deny invokes onResolve with (mainThreadId, callId, { approved: false })', async () => {
  const onResolve = mock(
    async (
      _threadId: string,
      _callId: string,
      _body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
    ) => {},
  )
  const entries: ConsoleEntry[] = [
    turn([
      call({
        id: 'tc_a',
        name: 'request_approval',
        state: 'waiting',
        waitPrompt: 'do the thing?',
        waitKind: 'approval',
      }),
    ]),
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={onResolve} />,
  )
  const deny = Array.from(container.querySelectorAll('button')).find((b) =>
    b.textContent?.toLowerCase().includes('deny'),
  )
  await act(async () => {
    fireEvent.click(deny!)
  })

  expect(onResolve.mock.calls.length).toBe(1)
  const [threadId, callId, body] = onResolve.mock.calls[0]
  expect(threadId).toBe(MAIN_TID)
  expect(callId).toBe('tc_a')
  expect(body.approved).toBe(false)
})

// --- 4. ordering across multiple turns -----------------------------

test('picks the first waiting user-input call across multiple turns', () => {
  const entries: ConsoleEntry[] = [
    turn([call({ name: 'fetch_url', state: 'ok' })]),
    {
      ...turn([
        call({ id: 'tc_task', name: 'task', state: 'waiting' }),
        call({
          id: 'tc_ask',
          name: 'ask_user',
          state: 'waiting',
          waitPrompt: 'pick',
          waitOptions: ['x', 'y'],
        }),
      ]),
      id: 'turn_2',
      turnUid: 'tu_2',
    },
  ]
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={{}} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toContain('pick')
})

// --- 5. sub-thread waits (prompt #8 regression) --------------------

test('surfaces a SUB-thread ask_user (researcher delegated ask_user)', () => {
  // The exact prompt #8 scenario: researcher pauses for user input.
  // The wait lives inside a sub-thread's tool call, NOT in top-level
  // entries. Before the fix, DeferredPanel silently ignored it.
  const subThreads: SubThreadMap = {
    [SUB_TID]: subActivity([
      call({
        id: 'tc_sub_ask',
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'Which site do you want?',
        waitKind: 'multiple_choice',
        waitOptions: ['example.com', 'example.org'],
      }),
    ]),
  }
  const { container } = render(
    <DeferredPanel entries={[]} subThreads={subThreads} onResolve={noOp} />,
  )
  const text = container.textContent ?? ''
  expect(text).toContain('Which site do you want?')
  expect(text).toContain('example.com')
  expect(text).toContain('example.org')
})

test('resolving a sub-thread wait POSTs against the SUB thread id, not main', async () => {
  // The resolve route's thread_id is what the coordinator uses to
  // derive which agent owns the parked activity. For sub-thread
  // waits this MUST be the sub id; otherwise the main agent's
  // runtime is asked to resolve a tool call it doesn't own.
  const onResolve = mock(
    async (
      _threadId: string,
      _callId: string,
      _body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
    ) => {},
  )
  const subThreads: SubThreadMap = {
    [SUB_TID]: subActivity([
      call({
        id: 'tc_sub_ask',
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'pick a site',
        waitKind: 'multiple_choice',
        waitOptions: ['example.com', 'example.org'],
      }),
    ]),
  }
  const { container } = render(
    <DeferredPanel entries={[]} subThreads={subThreads} onResolve={onResolve} />,
  )
  const button = Array.from(container.querySelectorAll('button')).find(
    (b) => b.textContent === 'example.com',
  )
  await act(async () => {
    fireEvent.click(button!)
  })

  expect(onResolve.mock.calls.length).toBe(1)
  const [threadId, callId, body] = onResolve.mock.calls[0]
  expect(threadId).toBe(SUB_TID)
  expect(callId).toBe('tc_sub_ask')
  expect(body.answer).toBe('example.com')
})

test('top-level wait beats sub-thread wait when both are present', () => {
  // findFirstWaiting walks entries first, then subThreads. So a
  // main-thread wait takes priority. (Either order would be fine
  // semantically; we just pin the order so the test catches drift.)
  const entries: ConsoleEntry[] = [
    turn([
      call({
        id: 'tc_main_ask',
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'MAIN question',
        waitKind: 'multiple_choice',
        waitOptions: ['m1', 'm2'],
      }),
    ]),
  ]
  const subThreads: SubThreadMap = {
    [SUB_TID]: subActivity([
      call({
        id: 'tc_sub_ask',
        name: 'ask_user',
        state: 'waiting',
        waitPrompt: 'SUB question',
        waitKind: 'multiple_choice',
        waitOptions: ['s1', 's2'],
      }),
    ]),
  }
  const { container } = render(
    <DeferredPanel entries={entries} subThreads={subThreads} onResolve={noOp} />,
  )
  expect(container.textContent ?? '').toContain('MAIN question')
  expect(container.textContent ?? '').not.toContain('SUB question')
})
