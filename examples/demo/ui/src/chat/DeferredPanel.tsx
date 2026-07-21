import { Check, Loader2, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import type { ConsoleEntry, ToolCallEntry } from './state'
import type { SubThreadMap } from './subThreads'

type Props = {
  entries: ConsoleEntry[]
  /** Sub-thread state. Required so the panel can surface ask_user /
   * request_approval calls emitted by sub-agents (e.g. the researcher
   * pausing to ask the user a clarifying question). Without this, only
   * the main thread's waits would pop the panel. */
  subThreads: SubThreadMap
  onResolve: (
    threadId: string,
    toolCallId: string,
    body: {
      approved?: boolean
      answer?: string
      payload?: Record<string, unknown>
    },
  ) => Promise<void>
}

type Pending = {
  /** Which thread owns the waiting tool call. For sub-agent waits this
   * is the sub-thread id, not the main thread. The resolve POST must
   * target this thread so the runtime signals the owning workflow. */
  threadId: string
  call: ToolCallEntry
  kind: 'approval' | 'question' | 'generic'
}

export function DeferredPanel({ entries, subThreads, onResolve }: Props) {
  const pending = findFirstWaiting(entries, subThreads)
  if (!pending) return null
  return (
    <DeferredCard
      key={pending.call.id}
      onResolve={onResolve}
      pending={pending}
    />
  )
}

function DeferredCard({
  pending,
  onResolve,
}: {
  pending: Pending
  onResolve: Props['onResolve']
}) {
  const { call, kind, threadId } = pending
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
  }, [call.id])

  async function resolve(body: {
    approved?: boolean
    answer?: string
    payload?: Record<string, unknown>
  }) {
    setSubmitting(true)
    setError(null)
    try {
      await onResolve(threadId, call.id, body)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  const options = call.waitOptions ?? []

  return (
    <div className="mx-auto w-full max-w-[44rem] animate-[fade-rise_240ms_ease-out_both]">
      <div className="rounded-[1rem] border border-accent-blue/30 bg-accent-blue/[0.04] p-3.5">
        <div className="mb-2 flex items-center gap-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-accent-blue">
          {submitting ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <span className="inline-block h-1.5 w-1.5 animate-[turn-streaming-pulse_1.2s_ease-in-out_infinite] rounded-full bg-accent-blue" />
          )}
          <span>{labelFor(kind)}</span>
          <span>·</span>
          <span className="text-ink-fade">{call.name}</span>
        </div>
        <div className="mb-3 text-[0.92rem] leading-snug text-ink">
          {call.waitPrompt}
        </div>

        {kind === 'approval' ? (
          <div className="flex items-center gap-2">
            <button
              className={cn(
                'inline-flex items-center gap-1.5 rounded-md border border-ink bg-ink px-3 py-1.5 text-[0.82rem] text-paper transition-opacity hover:opacity-90',
                'disabled:cursor-not-allowed disabled:opacity-50',
              )}
              disabled={submitting}
              onClick={() => resolve({ approved: true })}
              type="button"
            >
              <Check className="h-3.5 w-3.5" />
              Approve
            </button>
            <button
              className={cn(
                'inline-flex items-center gap-1.5 rounded-md border border-line-strong bg-white px-3 py-1.5 text-[0.82rem] text-ink-soft transition-colors hover:bg-ink/[0.04]',
                'disabled:cursor-not-allowed disabled:opacity-50',
              )}
              disabled={submitting}
              onClick={() => resolve({ approved: false })}
              type="button"
            >
              <X className="h-3.5 w-3.5" />
              Deny
            </button>
          </div>
        ) : options.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {options.map((option) => (
              <button
                className={cn(
                  'inline-flex items-center rounded-md border border-line-strong bg-white px-3 py-1.5 text-[0.86rem] text-ink transition-colors hover:border-ink/40 hover:bg-ink/[0.04]',
                  'disabled:cursor-not-allowed disabled:opacity-50',
                )}
                disabled={submitting}
                key={option}
                onClick={() => resolve({ answer: option })}
                type="button"
              >
                {option}
              </button>
            ))}
          </div>
        ) : (
          // Fallback: ask_user without options. Shouldn't happen now
          // the tool requires them at the schema level, but defensive
          // for old persisted calls.
          <div className="text-[0.78rem] italic text-ink-fade">
            (No options available — the agent should reissue this call with
            multiple-choice options.)
          </div>
        )}

        {error ? (
          <div className="mt-2 text-[0.78rem] text-accent-warm">{error}</div>
        ) : null}
      </div>
    </div>
  )
}

function labelFor(kind: Pending['kind']): string {
  if (kind === 'approval') return 'approval needed'
  if (kind === 'question') return 'agent is asking'
  return 'awaiting input'
}

// Older persisted calls may not have a wait kind, so retain the
// well-known tool names as a compatibility fallback. New calls are
// classified by their declared wait request instead of their tool name.
const USER_INPUT_TOOLS = new Set(['ask_user', 'request_approval'])

function pendingFor(call: ToolCallEntry, threadId: string): Pending | null {
  if (call.state !== 'waiting') return null
  const kind: Pending['kind'] =
    call.waitKind === 'approval'
      ? 'approval'
      : call.waitKind === 'multiple_choice'
        ? 'question'
        : call.name === 'request_approval'
          ? 'approval'
          : call.name === 'ask_user'
            ? 'question'
            : 'generic'
  if (kind === 'generic' && !USER_INPUT_TOOLS.has(call.name)) return null
  return { call, threadId, kind }
}

function findFirstWaiting(
  entries: ConsoleEntry[],
  subThreads: SubThreadMap,
): Pending | null {
  // Top-level entries first (main thread's own waits).
  for (const entry of entries) {
    if (entry.kind !== 'turn') continue
    for (const call of entry.toolCalls) {
      const p = pendingFor(call, entry.threadId)
      if (p) return p
    }
  }
  // Then walk every sub-thread's turns — sub-agent ask_user /
  // request_approval calls land here, NOT in top-level entries.
  // Without this loop, prompt #8 (sub-agent deferred) silently
  // fails: the wait event arrives on main's SSE channel and routes
  // to the sub-thread reducer, the tool's state becomes 'waiting',
  // but the panel never sees it.
  for (const activity of Object.values(subThreads)) {
    for (const turn of activity.turns) {
      for (const call of turn.toolCalls) {
        const p = pendingFor(call, activity.subThreadId)
        if (p) return p
      }
    }
  }
  return null
}
