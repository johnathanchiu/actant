/**
 * ToolCallRow rendering tests.
 *
 * Targets the specific bug class the user reported: the "double-chevron
 * / open shows nothing" state and the `task()` label shape. Mounts the
 * component with various ToolCallEntry shapes + nestedTranscript props
 * and asserts DOM structure.
 *
 * Invariants tested:
 *   - Exactly one chevron toggle (no nested second toggle inside the body).
 *   - Default-open when error / waiting / nestedTranscript present.
 *   - Default-closed otherwise.
 *   - Click toggles open <-> closed; chevron gets `rotate-90` when open.
 *   - Body, when open, contains the args section + result + nested
 *     transcript (when provided).
 *   - labelFor: `task -> researcher`, `task -> researcher failed`,
 *     `task -> researcher…` while streaming.
 */

import { expect, test } from 'bun:test'
import { fireEvent, render } from '@testing-library/react'
import { ToolCallRow } from '../ToolCallRow'
import type { ToolCallEntry } from '../state'

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

function chevronButtons(container: HTMLElement): HTMLButtonElement[] {
  return Array.from(container.querySelectorAll('button')).filter((b) =>
    b.querySelector('svg.lucide-chevron-right'),
  ) as HTMLButtonElement[]
}

// --- 1. exactly one chevron ----------------------------------------

test('renders exactly ONE chevron toggle (no nested second toggle)', () => {
  const { container } = render(
    <ToolCallRow call={call({ state: 'ok', result: 'done' })} />,
  )
  expect(chevronButtons(container).length).toBe(1)
})

test('renders exactly ONE chevron toggle even with a nestedTranscript', () => {
  const { container } = render(
    <ToolCallRow
      call={call({ name: 'task', state: 'resolved', subagent: 'researcher' })}
      nestedTranscript={<div data-testid="nested">nested content</div>}
    />,
  )
  expect(chevronButtons(container).length).toBe(1)
})

// --- 2. default-open behavior --------------------------------------

test('defaults closed for a plain successful call', () => {
  const { container } = render(<ToolCallRow call={call({ state: 'ok' })} />)
  // Body contains the "arguments" section label only when open.
  expect((container.textContent ?? '').toLowerCase()).not.toContain('arguments')
})

test('defaults open when state=error', () => {
  const { container } = render(
    <ToolCallRow call={call({ state: 'error', error: 'kaboom' })} />,
  )
  expect((container.textContent ?? '').toLowerCase()).toContain('arguments')
  expect(container.textContent ?? '').toContain('kaboom')
})

test('defaults open when state=waiting', () => {
  const { container } = render(
    <ToolCallRow
      call={call({ state: 'waiting', waitPrompt: 'pick one' })}
    />,
  )
  expect((container.textContent ?? '').toLowerCase()).toContain('arguments')
  expect(container.textContent ?? '').toContain('pick one')
})

test('defaults open when nestedTranscript is provided', () => {
  const { container, queryByTestId } = render(
    <ToolCallRow
      call={call({ name: 'task', state: 'resolved', subagent: 'researcher' })}
      nestedTranscript={<div data-testid="nested">nested content</div>}
    />,
  )
  expect(queryByTestId('nested')).not.toBeNull()
  expect((container.textContent ?? '').toLowerCase()).toContain('sub-agent transcript')
})

// --- 3. click toggles ----------------------------------------------

test('clicking the chevron toggles open <-> closed and rotates the icon', () => {
  const { container } = render(<ToolCallRow call={call({ state: 'ok' })} />)
  const button = chevronButtons(container)[0]
  expect(button).toBeDefined()

  // Initially closed: no rotate-90 on the chevron.
  let chevron = container.querySelector('svg.lucide-chevron-right')
  expect(chevron?.getAttribute('class') ?? '').not.toContain('rotate-90')
  expect((container.textContent ?? '').toLowerCase()).not.toContain('arguments')

  // Click -> open.
  fireEvent.click(button)
  chevron = container.querySelector('svg.lucide-chevron-right')
  expect(chevron?.getAttribute('class') ?? '').toContain('rotate-90')
  expect((container.textContent ?? '').toLowerCase()).toContain('arguments')

  // Click -> closed again.
  fireEvent.click(button)
  chevron = container.querySelector('svg.lucide-chevron-right')
  expect(chevron?.getAttribute('class') ?? '').not.toContain('rotate-90')
  expect((container.textContent ?? '').toLowerCase()).not.toContain('arguments')
})

// --- 4. body contents when open ------------------------------------

test('open body shows args, result, and nested transcript in order', () => {
  const { container } = render(
    <ToolCallRow
      call={call({
        name: 'task',
        state: 'resolved',
        result: '{"text":"summary"}',
        args: { subagent: 'researcher', prompt: 'fetch X' },
        subagent: 'researcher',
      })}
      nestedTranscript={<div data-testid="nested">nested content</div>}
    />,
  )
  const text = container.textContent ?? ''
  const subPos = text.toLowerCase().indexOf('sub-agent transcript')
  const argsPos = text.toLowerCase().indexOf('arguments')
  const resultPos = text.toLowerCase().indexOf('result')
  expect(subPos).toBeGreaterThanOrEqual(0)
  expect(argsPos).toBeGreaterThan(subPos)
  expect(resultPos).toBeGreaterThan(argsPos)
})

test('open body shows error section when state=error', () => {
  const { container } = render(
    <ToolCallRow
      call={call({ state: 'error', error: 'connection refused' })}
    />,
  )
  const text = container.textContent ?? ''
  expect(text.toLowerCase()).toContain('error')
  expect(text).toContain('connection refused')
})

// --- 5. labelFor for task() ----------------------------------------

test('task() label uses arrow + subagent name when resolved', () => {
  const { container } = render(
    <ToolCallRow
      call={call({ name: 'task', state: 'resolved', subagent: 'researcher' })}
    />,
  )
  expect(container.textContent ?? '').toContain('task → researcher')
})

test('task() label has ellipsis suffix while streaming', () => {
  const { container } = render(
    <ToolCallRow
      call={call({
        name: 'task',
        state: 'streaming',
        argsText: '{"subagent":"researcher"',
        args: { subagent: 'researcher' },
        subagent: 'researcher',
      })}
    />,
  )
  expect(container.textContent ?? '').toContain('task → researcher…')
})

test('task() label shows "failed" when state=error', () => {
  const { container } = render(
    <ToolCallRow
      call={call({
        name: 'task',
        state: 'error',
        error: 'sub-agent crashed',
        subagent: 'researcher',
      })}
    />,
  )
  expect(container.textContent ?? '').toContain('task → researcher failed')
})

test('falls back to bare "task" when no subagent is known', () => {
  const { container } = render(
    <ToolCallRow call={call({ name: 'task', state: 'streaming' })} />,
  )
  // No arrow, just the name + ellipsis suffix for streaming.
  expect(container.textContent ?? '').toContain('task…')
  expect(container.textContent ?? '').not.toContain('→')
})

// --- 6. args preview in collapsed header ---------------------------

test('non-task call shows args preview in the collapsed header', () => {
  const { container } = render(
    <ToolCallRow
      call={call({
        name: 'fetch_url',
        argsText: '{"url":"https://example.com"}',
        args: { url: 'https://example.com' },
        state: 'ok',
      })}
    />,
  )
  expect(container.textContent ?? '').toContain('https://example.com')
})

test('task call does NOT show args preview in header (uses label arrow instead)', () => {
  const { container } = render(
    <ToolCallRow
      call={call({
        name: 'task',
        argsText: '{"subagent":"researcher","prompt":"fetch X"}',
        args: { subagent: 'researcher', prompt: 'fetch X' },
        state: 'resolved',
        subagent: 'researcher',
      })}
    />,
  )
  // The header shows "task -> researcher"; the prompt should not bleed
  // into the header even though args contains it.
  const header = container.querySelector('button')
  expect(header?.textContent ?? '').toContain('task → researcher')
  expect(header?.textContent ?? '').not.toContain('fetch X')
})
