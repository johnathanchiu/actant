#!/usr/bin/env bun
/**
 * Wire-level smoke test for the demo server's SSE stream.
 *
 * Runs against a live `just demo`. Sends a fresh thread one prompt,
 * records every SSE event, and prints:
 *
 *   1. The full event sequence in arrival order.
 *   2. Per-event timing relative to send.
 *   3. Invariant checks per mode.
 *
 * Modes:
 *   --mode task        (default) — exercises the task → researcher
 *                       sub-agent dispatch path.
 *   --mode ask-user    — sends a prompt that should trigger the
 *                       ask_user tool. When `tool_waiting` arrives,
 *                       auto-resolves with the first option, then
 *                       verifies tool_resolved + tool_result land.
 *   --mode approval    — sends a prompt that should trigger
 *                       request_approval. Auto-resolves with
 *                       approved=true. Verifies tool_resolved.
 *
 * Usage:
 *   bun run scripts/sse_smoke.ts
 *   bun run scripts/sse_smoke.ts --mode ask-user
 *   bun run scripts/sse_smoke.ts --mode approval --raw
 *   bun run scripts/sse_smoke.ts "custom prompt"
 */

import { decodeActantFrame, parseSseFrame, type ActantEvent } from '../src/chat/wire'

const BASE = process.env.ACTANT_DEMO_BASE ?? 'http://localhost:8181'
const args = process.argv.slice(2)
const rawMode = args.includes('--raw')
const modeIdx = args.indexOf('--mode')
const mode: Mode =
  modeIdx >= 0 && args[modeIdx + 1]
    ? ((args[modeIdx + 1] as Mode) ?? 'task')
    : 'task'

type Mode = 'task' | 'ask-user' | 'approval' | 'parallel' | 'nested' | 'nested-deferred'

const DEFAULT_PROMPT: Record<Mode, string> = {
  task: 'use the task tool: delegate to researcher: fetch https://example.com and summarize in one sentence',
  'ask-user':
    'I want to plan a one-day trip. Use ask_user to confirm which city I am visiting — pick one of: Tokyo, Paris, New York.',
  approval:
    'Use the request_approval tool to confirm before deleting the file /tmp/important.txt.',
  parallel:
    'Fetch these three URLs IN PARALLEL (issue all three fetch_url calls in a single response, not sequentially), then tell me the byte length of each: https://example.com, https://example.org, https://httpbin.org/get',
  nested:
    'Delegate to the researcher with this exact instruction: "Fetch https://example.com, then delegate the summary writing to the summarizer subagent via the task tool, then return that summary as your final answer."',
  'nested-deferred':
    'Delegate to the researcher with this exact instruction: "Use ask_user with options [example.com, example.org] to confirm which site I want, then fetch it and summarize."',
}

const customPrompt = args.find(
  (a, i) => !a.startsWith('--') && args[i - 1] !== '--mode',
)
const prompt = customPrompt ?? DEFAULT_PROMPT[mode]

type Recorded = { t: number; event: ActantEvent }

async function main() {
  const threadId = `smoke_${Date.now()}`
  console.log(`mode:   ${mode}`)
  console.log(`thread: ${threadId}`)
  console.log(`prompt: ${prompt}`)
  console.log('---')

  const ctrl = new AbortController()
  const events: Recorded[] = []
  const t0 = Date.now()
  const resolved = new Set<string>()

  // Open SSE first.
  const resp = await fetch(
    `${BASE}/api/threads/${encodeURIComponent(threadId)}/events`,
    { headers: { Accept: 'text/event-stream' }, signal: ctrl.signal },
  )
  if (!resp.ok || !resp.body) {
    console.error(`SSE failed: ${resp.status}`)
    process.exit(1)
  }

  const readerDone = (async () => {
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
      while (true) {
        const sep = buf.indexOf('\n\n')
        if (sep < 0) break
        const raw = buf.slice(0, sep)
        buf = buf.slice(sep + 2)
        const ev = decodeActantFrame(parseSseFrame(raw))
        if (!ev) continue
        events.push({ t: Date.now() - t0, event: ev })

        // Interactive resolution. ask-user / approval resolve waits
        // on the MAIN thread; nested-deferred resolves waits emitted
        // by a SUB thread (whose events also flow to the root channel
        // via the demo coordinator's publish-to-root chain).
        const resolveOnMain =
          mode === 'ask-user' || mode === 'approval'
        const resolveOnSub = mode === 'nested-deferred'
        const isMainWait = ev.type === 'tool_waiting' && !ev.parent_thread_id
        const isSubWait = ev.type === 'tool_waiting' && ev.parent_thread_id
        if (
          ev.type === 'tool_waiting' &&
          ((resolveOnMain && isMainWait) || (resolveOnSub && isSubWait)) &&
          !resolved.has(ev.data.tool_call_id)
        ) {
          resolved.add(ev.data.tool_call_id)
          const body = resolveBodyFor(mode, ev.data.wait_kind, ev.data.wait_payload)
          // Sub-thread waits resolve via the SUB thread's URL (the
          // resolve route's `thread_id` segment is what tells the
          // coordinator which agent owns the waiting tool call).
          const resolveThreadId = ev.thread_id
          console.log(
            `+${pad(Date.now() - t0)}ms (auto-resolve thread=${resolveThreadId.slice(0, 16)}) ${JSON.stringify(body)}`,
          )
          // Fire-and-forget — actual outcome surfaces as tool_resolved / tool_result.
          void fetch(
            `${BASE}/api/threads/${encodeURIComponent(resolveThreadId)}/tool_calls/${encodeURIComponent(
              ev.data.tool_call_id,
            )}/resolve`,
            {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify(body),
            },
          ).catch((e) => console.error(`resolve failed: ${e}`))
        }

        if (ev.type === 'complete' && !ev.parent_thread_id) {
          ctrl.abort()
          return
        }
      }
    }
  })()

  await new Promise((r) => setTimeout(r, 300))
  const sendT = Date.now() - t0
  console.log(`+${pad(sendT)}ms POST send_message`)
  const send = await fetch(
    `${BASE}/api/threads/${encodeURIComponent(threadId)}/messages`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ content: prompt }),
    },
  )
  if (send.status !== 202) {
    console.error(`send failed: ${send.status}`)
    process.exit(1)
  }

  await Promise.race([
    readerDone,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 90s')), 90_000)),
  ]).catch((e) => {
    console.error(`reader error: ${e}`)
    ctrl.abort()
  })

  printSequence(events)
  console.log('---')
  printInvariants(events, mode)
}

function resolveBodyFor(
  mode: Mode,
  waitKind: string | undefined,
  payload: Record<string, unknown> | undefined,
): { approved?: boolean; answer?: string } {
  if (mode === 'approval' || waitKind === 'approval') {
    return { approved: true }
  }
  // ask-user / multiple_choice: pick the first option.
  const opts = Array.isArray(payload?.options) ? (payload!.options as unknown[]) : []
  const first = opts.find((o): o is string => typeof o === 'string') ?? ''
  return { answer: first }
}

function printSequence(events: Recorded[]) {
  for (const { t, event } of events) {
    const scope = event.parent_thread_id ? `SUB(${event.thread_id.slice(0, 12)})` : 'main'
    const detail = describe(event)
    console.log(`+${pad(t)}ms ${scope.padEnd(28)} ${event.type.padEnd(26)} ${detail}`)
  }
  if (rawMode) {
    console.log('\n--- raw envelopes ---')
    for (const { t, event } of events) {
      console.log(`+${pad(t)}ms ${JSON.stringify(event)}`)
    }
  }
}

function pad(n: number) {
  return String(n).padStart(5, ' ')
}

function describe(event: ActantEvent): string {
  switch (event.type) {
    case 'turn_start':
      return `turn=${event.data.turn} uid=${event.data.turn_uid ?? event.data.turn_id ?? '?'}`
    case 'text_delta':
      return `+${JSON.stringify(event.data.delta).slice(0, 40)}`
    case 'thinking_delta':
      return `+${JSON.stringify(event.data.delta).slice(0, 40)}`
    case 'tool_call_start':
      return `name=${event.data.name} id=${event.data.tool_call_id.slice(0, 16)}`
    case 'tool_call_args_delta':
      return `+${JSON.stringify(event.data.delta).slice(0, 40)}`
    case 'tool_call_args_complete':
      return `id=${event.data.tool_call_id.slice(0, 16)}`
    case 'assistant_message':
      return `content=${JSON.stringify(event.data.content).slice(0, 40)} tool_calls=${event.data.tool_calls.length}`
    case 'tool_call':
      return `name=${event.data.name} id=${event.data.tool_call_id.slice(0, 16)}`
    case 'tool_result':
      return `id=${event.data.tool_call_id.slice(0, 16)} ok=${!event.data.error}`
    case 'tool_waiting':
      return `kind=${event.data.wait_kind ?? '?'} id=${event.data.tool_call_id.slice(0, 16)} options=${JSON.stringify((event.data.wait_payload?.options as unknown[]) ?? []).slice(0, 40)}`
    case 'tool_resolved':
      return `id=${event.data.tool_call_id.slice(0, 16)}`
    case 'complete':
      return `success=${event.data.success}`
    case 'error':
      return `msg=${event.data.message.slice(0, 60)}`
  }
}

function printInvariants(events: Recorded[], mode: Mode) {
  console.log('invariants:')
  // 1. Each turn_start (main thread, with explicit uid) has a matching
  //    assistant_message. Synthesized turns with client-side uids
  //    aren't expected on the server side.
  const mainTurnStarts = events
    .filter((e) => !e.event.parent_thread_id && e.event.type === 'turn_start')
    .map((e) => (e.event as Extract<ActantEvent, { type: 'turn_start' }>))
  const mainAssistantMessages = events.filter(
    (e) => !e.event.parent_thread_id && e.event.type === 'assistant_message',
  )
  check(
    `main: turn_start count (${mainTurnStarts.length}) >= assistant_message count (${mainAssistantMessages.length})`,
    mainTurnStarts.length >= mainAssistantMessages.length,
  )

  // 2. complete fires exactly once on the main thread.
  const mainComplete = events.filter(
    (e) => !e.event.parent_thread_id && e.event.type === 'complete',
  )
  check(
    `main: complete fires exactly once (got ${mainComplete.length})`,
    mainComplete.length === 1,
  )

  // 3. Every sub-thread event has both parent_thread_id AND parent_tool_call_id.
  const subEvents = events.filter((e) => e.event.parent_thread_id)
  const subMissingParent = subEvents.filter((e) => !e.event.parent_tool_call_id)
  check(
    `sub: every event carries parent_tool_call_id (missing on ${subMissingParent.length}/${subEvents.length})`,
    subMissingParent.length === 0,
  )

  // 4. Within a single thread, no assistant_message arrives before any turn_start.
  const lastTurnStartUidByThread: Record<string, string> = {}
  let assistantBeforeStart = 0
  for (const { event } of events) {
    if (event.type === 'turn_start') {
      lastTurnStartUidByThread[event.thread_id] =
        event.data.turn_uid ?? event.data.turn_id ?? ''
    }
    if (event.type === 'assistant_message') {
      if (!lastTurnStartUidByThread[event.thread_id]) assistantBeforeStart++
    }
  }
  check(
    `order: no assistant_message arrived before any turn_start on its thread (violations: ${assistantBeforeStart})`,
    assistantBeforeStart === 0,
  )

  // 5. Every tool_call has a prior tool_call_start.
  const seenStarts = new Set<string>()
  let toolCallBeforeStart = 0
  for (const { event } of events) {
    if (event.type === 'tool_call_start') seenStarts.add(event.data.tool_call_id)
    if (event.type === 'tool_call' && !seenStarts.has(event.data.tool_call_id)) {
      toolCallBeforeStart++
    }
  }
  check(
    `order: every tool_call has a prior tool_call_start (violations: ${toolCallBeforeStart})`,
    toolCallBeforeStart === 0,
  )

  // Mode-specific invariants.
  if (mode === 'ask-user') {
    const waiting = events.find(
      (e) =>
        e.event.type === 'tool_waiting' &&
        !e.event.parent_thread_id &&
        e.event.data.wait_kind === 'multiple_choice',
    )
    check(
      `ask-user: a multiple_choice tool_waiting event was emitted`,
      Boolean(waiting),
    )
    if (waiting && waiting.event.type === 'tool_waiting') {
      const opts = (waiting.event.data.wait_payload?.options as unknown[]) ?? []
      const allStrings = opts.length >= 2 && opts.every((o) => typeof o === 'string')
      check(
        `ask-user: payload.options is an array of >=2 strings (got ${JSON.stringify(opts).slice(0, 80)})`,
        allStrings,
      )
    }
    const resolvedEv = events.filter(
      (e) => e.event.type === 'tool_resolved' && !e.event.parent_thread_id,
    )
    check(
      `ask-user: tool_resolved fires after our auto-resolve (got ${resolvedEv.length})`,
      resolvedEv.length >= 1,
    )
  }

  if (mode === 'parallel') {
    // True parallelism: at least 2 tool_call_start events arrive
    // BEFORE the first tool_result. Sequential execution would
    // interleave start -> result -> start -> result; parallel
    // execution issues all starts up front.
    const fetchStarts = events.filter(
      (e) =>
        e.event.type === 'tool_call_start' &&
        !e.event.parent_thread_id &&
        e.event.data.name === 'fetch_url',
    )
    const firstResultT = events.find(
      (e) => e.event.type === 'tool_result' && !e.event.parent_thread_id,
    )?.t
    const startsBeforeFirstResult = fetchStarts.filter(
      (e) => firstResultT === undefined || e.t < firstResultT,
    )
    check(
      `parallel: agent issued >=2 fetch_url calls (got ${fetchStarts.length})`,
      fetchStarts.length >= 2,
    )
    check(
      `parallel: >=2 fetch_url tool_call_start events arrived before the first tool_result (got ${startsBeforeFirstResult.length})`,
      startsBeforeFirstResult.length >= 2,
    )
  }

  if (mode === 'nested') {
    // 2-level nesting: main spawns researcher (parent_thread_id = main),
    // researcher spawns summarizer (parent_thread_id = researcher's sub).
    // Both sets of events must reach the main thread's SSE channel via
    // the demo coordinator's publish-to-root chain.
    const subEvents2 = events.filter((e) => e.event.parent_thread_id)
    const parentThreads = new Set(
      subEvents2.map((e) => e.event.parent_thread_id!),
    )
    check(
      `nested: events arrived from >=2 distinct parent_thread_ids (got ${parentThreads.size})`,
      parentThreads.size >= 2,
    )
    const subagents = new Set(
      subEvents2.flatMap((e) => (e.event.subagent ? [e.event.subagent] : [])),
    )
    check(
      `nested: both 'researcher' and 'summarizer' subagent events surfaced (got ${JSON.stringify([...subagents])})`,
      subagents.has('researcher') && subagents.has('summarizer'),
    )
  }

  if (mode === 'nested-deferred') {
    // Sub-thread emitted a tool_waiting that surfaced on main's SSE
    // (carries parent_thread_id + parent_tool_call_id). Our auto-
    // resolver then unblocked it, and tool_resolved fired on that
    // sub-thread.
    const subWait = events.find(
      (e) =>
        e.event.type === 'tool_waiting' &&
        e.event.parent_thread_id &&
        (e.event.data.wait_kind === 'multiple_choice' ||
          e.event.data.wait_kind === 'approval'),
    )
    check(
      `nested-deferred: a sub-thread tool_waiting surfaced on main SSE (with parent_thread_id)`,
      Boolean(subWait),
    )
    if (subWait && subWait.event.type === 'tool_waiting') {
      check(
        `nested-deferred: that sub-thread wait carries parent_tool_call_id`,
        Boolean(subWait.event.parent_tool_call_id),
      )
    }
    const subResolved = events.filter(
      (e) => e.event.type === 'tool_resolved' && e.event.parent_thread_id,
    )
    check(
      `nested-deferred: sub-thread tool_resolved fired after auto-resolve (got ${subResolved.length})`,
      subResolved.length >= 1,
    )
  }

  if (mode === 'approval') {
    const waiting = events.find(
      (e) =>
        e.event.type === 'tool_waiting' &&
        !e.event.parent_thread_id &&
        e.event.data.wait_kind === 'approval',
    )
    check(
      `approval: an "approval" tool_waiting event was emitted`,
      Boolean(waiting),
    )
    const resolvedEv = events.filter(
      (e) => e.event.type === 'tool_resolved' && !e.event.parent_thread_id,
    )
    check(
      `approval: tool_resolved fires after our auto-resolve (got ${resolvedEv.length})`,
      resolvedEv.length >= 1,
    )
    // Deferred tools encode the outcome in tool_resolved.data.output
    // (no separate tool_result event). approved=true should produce a
    // JSON-ish output mentioning the approved action.
    const resolvedOk = resolvedEv
      .map((e) => e.event as Extract<ActantEvent, { type: 'tool_resolved' }>)
      .at(-1)
    check(
      `approval: tool_resolved carries an output (got ${JSON.stringify(resolvedOk?.data.output ?? null).slice(0, 80)})`,
      Boolean(resolvedOk?.data.output),
    )
  }
}

function check(label: string, ok: boolean) {
  console.log(`  ${ok ? '✓' : '✗'} ${label}`)
  if (!ok) process.exitCode = 2
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
