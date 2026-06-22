/**
 * Thin React wiring for the chat surface.
 *
 * This hook is intentionally small. ALL the logic lives in the
 * layered modules:
 *   - wire.ts        — server event types + SSE frame parser
 *   - sseClient.ts   — fetch-streams transport
 *   - reducer.ts     — pure event → entries reducer
 *   - subThreads.ts  — sub-thread state + reducer
 *   - history.ts     — persisted messages → entries
 *   - api.ts         — typed REST client
 *
 * The hook does:
 *   1. Loads history (messages + waiting tool calls + sub-thread links).
 *   2. Opens the SSE stream.
 *   3. Routes incoming events to the right reducer (top-level vs sub-thread).
 *   4. Exposes state + sendMessage + resolveTool.
 *
 * StrictMode-safe via a generation counter: each useEffect run
 * increments the counter; in-flight async work bails out if the
 * counter advanced past it (i.e. a remount happened).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Api } from './api'
import { historyToEntries } from './history'
import { reduce } from './reducer'
import type { ConsoleEntry, TurnEntry } from './state'
import { backfillSubThread, reduceSubThread, type SubThreadMap } from './subThreads'
import { openActantStream } from './sseClient'

export type StreamState = 'connecting' | 'open' | 'reconnecting' | 'error'

export type UseAgentConsoleOptions = {
  api: Api
  threadId: string
  /** Bumped by the parent after a user message is sent so the sidebar
   * can refresh thread previews/counts. */
  onMessageSent?: () => void
}

export type UseAgentConsoleReturn = {
  entries: ConsoleEntry[]
  subThreads: SubThreadMap
  streamState: StreamState
  sending: boolean
  isLoadingHistory: boolean
  historyError: string | null
  isEmpty: boolean
  sendMessage: (content: string) => Promise<void>
  resolveTool: (
    threadId: string,
    toolCallId: string,
    body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
  ) => Promise<void>
}

export function useAgentConsole({
  api,
  threadId,
  onMessageSent,
}: UseAgentConsoleOptions): UseAgentConsoleReturn {
  const [entries, setEntries] = useState<ConsoleEntry[]>([])
  const [subThreads, setSubThreads] = useState<SubThreadMap>({})
  const [streamState, setStreamState] = useState<StreamState>('connecting')
  const [sending, setSending] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(true)
  const [historyError, setHistoryError] = useState<string | null>(null)

  const onMessageSentRef = useRef(onMessageSent)
  useEffect(() => {
    onMessageSentRef.current = onMessageSent
  }, [onMessageSent])

  const generationRef = useRef(0)

  useEffect(() => {
    const myGen = ++generationRef.current
    const ctrl = new AbortController()

    // Reset state for the new thread.
    setEntries([])
    setSubThreads({})
    setIsLoadingHistory(true)
    setHistoryError(null)
    setStreamState('connecting')

    void bootstrap(myGen, ctrl.signal)

    return () => {
      ctrl.abort()
    }

    async function bootstrap(gen: number, signal: AbortSignal) {
      // 1. Fetch history in parallel: messages, waiting tool calls,
      //    sub-thread links. All independent.
      let entries: ConsoleEntry[]
      let subMap: SubThreadMap = {}
      try {
        const [messages, waiting, subLinks] = await Promise.all([
          api.fetchMessages(threadId),
          api.fetchWaitingToolCalls(threadId).catch(() => []),
          api.fetchSubThreads(threadId).catch(() => []),
        ])
        if (gen !== generationRef.current) return

        entries = patchWaitingState(
          patchSubThreadLinks(historyToEntries(messages, threadId), subLinks),
          waiting,
        )
        setEntries(entries)

        // 2. Backfill each sub-thread's transcript from its own
        //    persisted messages — that way NestedTranscript renders
        //    on refresh, not just live. ALSO fetch each sub-thread's
        //    own waiting tool calls and patch them onto its turns —
        //    otherwise a refresh while a sub-agent is paused in
        //    ask_user / request_approval loses the deferred panel
        //    (the wait lives on the sub-thread, not main).
        await Promise.all(
          subLinks.map(async (link) => {
            try {
              const [subMessages, subWaiting] = await Promise.all([
                api.fetchMessages(link.sub_thread_id),
                api.fetchWaitingToolCalls(link.sub_thread_id).catch(() => []),
              ])
              if (gen !== generationRef.current) return
              const subEntries = patchWaitingState(
                historyToEntries(subMessages, link.sub_thread_id),
                subWaiting,
              )
              const subTurns = subEntries.filter(
                (e): e is TurnEntry => e.kind === 'turn',
              )
              subMap = backfillSubThread(subMap, link, subTurns, null)
              setSubThreads(subMap)
            } catch {
              // Sub-thread backfill is best-effort — a single failure
              // shouldn't fail the whole load.
            }
          }),
        )
      } catch (err) {
        if (gen !== generationRef.current) return
        if ((err as { name?: string }).name === 'AbortError') return
        setHistoryError(err instanceof Error ? err.message : String(err))
      } finally {
        if (gen === generationRef.current) setIsLoadingHistory(false)
      }

      if (gen !== generationRef.current) return

      // 3. Open the SSE stream and route events.
      try {
        setStreamState('open')
        const url = `${api.baseUrl}/api/threads/${encodeURIComponent(threadId)}/events`
        for await (const event of openActantStream(url, signal)) {
          if (gen !== generationRef.current) return
          if (event.parent_thread_id === threadId) {
            // Sub-thread event.
            setSubThreads((prev) => reduceSubThread(prev, event, threadId))
            // Also annotate the parent's tool call with the sub_thread_id
            // (so the renderer knows where to find the nested transcript).
            setEntries((prev) =>
              annotateSubThreadOnToolCall(
                prev,
                event.parent_tool_call_id ?? null,
                event.thread_id,
                event.subagent ?? null,
              ),
            )
          } else if (!event.parent_thread_id) {
            // Top-level event.
            setEntries((prev) => reduce(prev, event))
          }
          // Sub-thread events not for our parent are ignored.
        }
        if (gen === generationRef.current) setStreamState('reconnecting')
      } catch (err) {
        if (gen !== generationRef.current) return
        if ((err as { name?: string }).name === 'AbortError') return
        setStreamState('error')
      }
    }
  }, [api, threadId])

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim()
      if (!trimmed || sending) return
      setSending(true)
      // Optimistically append the user message; SSE doesn't echo it.
      setEntries((prev) => [
        ...prev,
        {
          kind: 'user',
          id: `user_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
          threadId,
          text: trimmed,
          timestamp: Date.now(),
        },
      ])
      try {
        await api.sendMessage(threadId, trimmed)
        onMessageSentRef.current?.()
      } catch (err) {
        setEntries((prev) => [
          ...prev,
          {
            kind: 'event',
            id: `event_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
            threadId,
            level: 'error',
            text: err instanceof Error ? err.message : String(err),
            timestamp: Date.now(),
          },
        ])
      } finally {
        setSending(false)
      }
    },
    [api, sending, threadId],
  )

  const resolveTool = useCallback(
    async (
      // `threadId` here can be the main thread OR a sub-thread id —
      // sub-agent ask_user / request_approval waits live on the sub-
      // thread's tool call, so the resolve must POST to that thread.
      targetThreadId: string,
      toolCallId: string,
      body: { approved?: boolean; answer?: string; payload?: Record<string, unknown> },
    ) => {
      await api.resolveTool(targetThreadId, toolCallId, body)
    },
    [api],
  )

  const isEmpty = useMemo(() => entries.length === 0, [entries.length])

  return {
    entries,
    subThreads,
    streamState,
    sending,
    isLoadingHistory,
    historyError,
    isEmpty,
    sendMessage,
    resolveTool,
  }
}

// ─── Helpers (pure) ─────────────────────────────────────────────────

/** Patch tool calls that are currently parked in WAITING state on the
 * server (via /waiting_tool_calls) — `historyToEntries` defaults them
 * to `state='pending'`. Without this step, a refresh mid-`ask_user`
 * loses the deferred panel. */
function patchWaitingState(
  entries: ConsoleEntry[],
  waiting: Array<{
    tool_call_id: string
    prompt: string | null
    wait_request: {
      kind?: string
      payload?: { options?: unknown } | Record<string, unknown>
    } | null
  }>,
): ConsoleEntry[] {
  if (waiting.length === 0) return entries
  const byId = new Map(waiting.map((w) => [w.tool_call_id, w]))
  return entries.map((e) => {
    if (e.kind !== 'turn') return e
    let mutated = false
    const toolCalls = e.toolCalls.map((c) => {
      const w = byId.get(c.id)
      if (!w) return c
      mutated = true
      const rawOptions = (w.wait_request?.payload as { options?: unknown } | undefined)
        ?.options
      const options = Array.isArray(rawOptions)
        ? rawOptions.filter((o): o is string => typeof o === 'string')
        : null
      return {
        ...c,
        state: 'waiting' as const,
        waitPrompt: w.prompt,
        waitKind: w.wait_request?.kind ?? null,
        waitOptions: options && options.length > 0 ? options : null,
      }
    })
    return mutated ? { ...e, toolCalls } : e
  })
}

/** Attach `subThreadId` to each `task()` tool call so NestedTranscript
 * can find the matching sub-thread activity. */
function patchSubThreadLinks(
  entries: ConsoleEntry[],
  links: Array<{ sub_thread_id: string; parent_tool_call_id: string }>,
): ConsoleEntry[] {
  if (links.length === 0) return entries
  const byToolCall = new Map(links.map((l) => [l.parent_tool_call_id, l]))
  return entries.map((e) => {
    if (e.kind !== 'turn') return e
    let mutated = false
    const toolCalls = e.toolCalls.map((c) => {
      const link = byToolCall.get(c.id)
      if (!link) return c
      mutated = true
      return { ...c, subThreadId: link.sub_thread_id }
    })
    return mutated ? { ...e, toolCalls } : e
  })
}

/** Live SSE counterpart of `patchSubThreadLinks`. When the first
 * sub-thread event arrives on the parent's channel, we know the
 * parent_tool_call_id → sub_thread_id mapping. Reflect it onto the
 * tool call so the renderer can show the nested transcript live
 * (not just after refresh). */
function annotateSubThreadOnToolCall(
  entries: ConsoleEntry[],
  parentToolCallId: string | null,
  subThreadId: string,
  subagent: string | null,
): ConsoleEntry[] {
  if (!parentToolCallId) return entries
  let mutated = false
  const next = entries.map((e) => {
    if (e.kind !== 'turn') return e
    if (!e.toolCalls.some((c) => c.id === parentToolCallId)) return e
    mutated = true
    return {
      ...e,
      toolCalls: e.toolCalls.map((c) =>
        c.id === parentToolCallId
          ? { ...c, subThreadId, subagent: subagent ?? c.subagent }
          : c,
      ),
    }
  })
  return mutated ? next : entries
}
