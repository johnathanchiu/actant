/**
 * TurnRow rendering tests.
 *
 * Mounts TurnRow with various TurnEntry shapes and asserts on the
 * DOM structure. Specifically targets the "stray dots" bug the
 * smoke tests + reducer tests don't catch — those verify state, not
 * DOM.
 *
 * Invariants tested:
 *   - A turn with text + tool calls renders NO streaming pulse dot.
 *   - A turn with empty state + isStreaming renders EXACTLY ONE
 *     pulse dot (not multiple).
 *   - A finalized turn with content has the footer line.
 *   - A compact turn (used in nested transcripts) has no footer +
 *     uses naked markdown (no double-bubble).
 *   - Nested transcripts render their own turns properly.
 */

import { expect, test } from 'bun:test'
import { render } from '@testing-library/react'
import { TurnRow } from '../TurnRow'
import type { ToolCallEntry, TurnEntry } from '../state'
import type { SubThreadActivity, SubThreadMap } from '../subThreads'

const TID = 't_1'

function turn(overrides: Partial<TurnEntry> = {}): TurnEntry {
  return {
    kind: 'turn',
    id: 'turn_1',
    turnUid: 'tu_1',
    threadId: TID,
    text: '',
    thinking: '',
    toolCalls: [],
    isStreaming: false,
    timestamp: 0,
    ...overrides,
  }
}

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

/** Count nodes matching the streaming-pulse dot's distinctive
 * animation class. If this finds anything in a "has content" state,
 * we have a stray-dot bug. */
function countPulseDots(container: HTMLElement): number {
  return container.querySelectorAll(
    '[class*="turn-streaming-pulse"]',
  ).length
}

// ─── 1. content-bearing turns must not have stray dots ───────────────

test('turn with text + tool calls renders NO pulse dot', () => {
  const t = turn({
    text: 'Let me try fetching this URL',
    toolCalls: [call({ name: 'fetch_url' })],
    isStreaming: false,
  })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect(countPulseDots(container)).toBe(0)
})

test('turn with text + tool calls + isStreaming=true STILL renders no pulse dot', () => {
  // The dot is for "agent is starting a turn but has nothing yet" —
  // once there's content, it must go away even if the turn is
  // technically still streaming.
  const t = turn({
    text: 'Let me try fetching this URL',
    toolCalls: [call({ name: 'fetch_url' })],
    isStreaming: true,
  })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect(countPulseDots(container)).toBe(0)
})

test('turn with only thinking + isStreaming=true renders NO pulse dot (thinking IS content)', () => {
  const t = turn({
    thinking: 'thinking about the request',
    isStreaming: true,
  })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect(countPulseDots(container)).toBe(0)
})

test('turn with NO content + isStreaming renders EXACTLY ONE pulse dot', () => {
  const t = turn({ isStreaming: true })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect(countPulseDots(container)).toBe(1)
})

test('turn with NO content + isStreaming=false renders NO pulse dot', () => {
  const t = turn({ isStreaming: false })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect(countPulseDots(container)).toBe(0)
})

// ─── 2. footer presence ─────────────────────────────────────────────

test('non-compact turn with content has an "assistant" footer', () => {
  const t = turn({ text: 'hello' })
  const { container } = render(<TurnRow model="claude-x" subThreads={{}} turn={t} />)
  expect(container.textContent ?? '').toContain('assistant')
  expect(container.textContent ?? '').toContain('claude-x')
})

test('compact turn (nested transcript) has NO footer', () => {
  const t = turn({ text: 'hello' })
  const { container } = render(
    <TurnRow compact model="claude-x" subThreads={{}} turn={t} />,
  )
  // The footer span starts with "assistant"; compact strips it.
  const lower = (container.textContent ?? '').toLowerCase()
  expect(lower).not.toContain('assistant ·')
})

// ─── 3. naked markdown inside nested transcripts ────────────────────

test('compact turn renders AssistantMarkdown without bubble border', () => {
  const t = turn({ text: 'nested content' })
  const { container } = render(
    <TurnRow compact model={null} subThreads={{}} turn={t} />,
  )
  // The default bubble class includes `bg-white/85`. In naked mode it
  // shouldn't be there.
  const markdownDivs = container.querySelectorAll('div')
  let bubbleCount = 0
  markdownDivs.forEach((d) => {
    if (d.className.includes('bg-white/85')) bubbleCount++
  })
  expect(bubbleCount).toBe(0)
})

test('non-compact turn renders AssistantMarkdown WITH bubble border', () => {
  const t = turn({ text: 'top-level content' })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  const markdownDivs = container.querySelectorAll('div')
  let bubbleCount = 0
  markdownDivs.forEach((d) => {
    if (d.className.includes('bg-white/85')) bubbleCount++
  })
  expect(bubbleCount).toBeGreaterThanOrEqual(1)
})

// ─── 4. nested transcripts ──────────────────────────────────────────

test('task() tool call with a sub-thread renders the nested transcript inline', () => {
  const subTurn: TurnEntry = {
    kind: 'turn',
    id: 'subturn_1',
    turnUid: 'sub_tu_1',
    threadId: 'sub_1',
    text: 'sub agent reply',
    thinking: '',
    toolCalls: [],
    isStreaming: false,
    timestamp: 0,
  }
  const activity: SubThreadActivity = {
    subThreadId: 'sub_1',
    parentThreadId: TID,
    parentToolCallId: 'tc_task',
    subagent: 'researcher',
    turns: [subTurn],
    isStreaming: false,
  }
  const subThreads: SubThreadMap = { sub_1: activity }
  const t = turn({
    toolCalls: [
      call({
        id: 'tc_task',
        name: 'task',
        subThreadId: 'sub_1',
        subagent: 'researcher',
        state: 'resolved',
        result: '{"text": "sub agent reply"}',
      }),
    ],
  })
  const { container } = render(
    <TurnRow model={null} subThreads={subThreads} turn={t} />,
  )
  expect(container.textContent ?? '').toContain('task → researcher')
  // Sub-thread's text should be rendered inline.
  expect(container.textContent ?? '').toContain('sub agent reply')
  // Sub-agent transcript should be marked as "sub-agent transcript".
  expect((container.textContent ?? '').toLowerCase()).toContain('sub-agent transcript')
  // No stray dot in the parent turn (it has tool calls = content).
  expect(countPulseDots(container)).toBe(0)
})

test('task() tool call WITHOUT sub-thread activity yet shows the placeholder', () => {
  // Sub-thread has just been spawned; events haven't arrived yet.
  // The tool call has `subThreadId` set but no entry in `subThreads`.
  const t = turn({
    toolCalls: [
      call({
        id: 'tc_task',
        name: 'task',
        subThreadId: 'sub_pending',
        subagent: 'researcher',
        state: 'waiting',
      }),
    ],
  })
  const { container } = render(<TurnRow model={null} subThreads={{}} turn={t} />)
  expect((container.textContent ?? '').toLowerCase()).toContain('sub-agent is starting')
})

// ─── 5. multi-turn rendering order ──────────────────────────────────

test('rendering multiple turns preserves their array order', () => {
  // This catches the "task came after the text response" complaint —
  // if the renderer (or its consumers) reverse turn order somewhere,
  // this fails.
  const turn1 = turn({
    id: 'turn_a',
    turnUid: 'tu_a',
    toolCalls: [call({ id: 'tc_x', name: 'fetch_url' })],
  })
  const turn2 = turn({
    id: 'turn_b',
    turnUid: 'tu_b',
    text: 'final response',
  })
  const { container } = render(
    <div>
      <TurnRow model={null} subThreads={{}} turn={turn1} />
      <TurnRow model={null} subThreads={{}} turn={turn2} />
    </div>,
  )
  const text = container.textContent ?? ''
  const fetchPos = text.indexOf('fetch_url')
  const finalPos = text.indexOf('final response')
  expect(fetchPos).toBeGreaterThanOrEqual(0)
  expect(finalPos).toBeGreaterThanOrEqual(0)
  expect(fetchPos).toBeLessThan(finalPos)
})
