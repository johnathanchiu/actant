import { ChevronRight, Loader2, AlertCircle, Check, Hourglass, Workflow } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import type { ToolCallEntry, ToolCallState } from './state'

const STATE_LABEL: Record<ToolCallState, string> = {
  streaming: 'building call',
  pending: 'queued',
  running: 'running',
  waiting: 'waiting on input',
  resolved: 'resolved',
  ok: 'ok',
  error: 'error',
}

export function ToolCallRow({
  call,
  nestedTranscript,
}: {
  call: ToolCallEntry
  /** Sub-agent's nested transcript for `task()` calls. Rendered
   * inside the unified body section, between the args and result. */
  nestedTranscript?: ReactNode
}) {
  const isTask = call.name === 'task'
  // Default open when there's something the user should pay attention
  // to: an error, an explicit wait state, OR a sub-agent transcript
  // (the latter IS the interesting bit of a task() call).
  const [open, setOpen] = useState(
    call.state === 'error' || call.state === 'waiting' || Boolean(nestedTranscript),
  )
  const Icon = isTask ? Workflow : iconForState(call.state)
  const argsPreview = formatArgsPreview(call)

  return (
    <div className="max-w-[44rem] overflow-hidden rounded-[0.85rem] border border-line bg-white/70 text-[0.82rem] text-ink-soft backdrop-blur-md">
      <button
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-ink/[0.025]"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <ChevronRight
          className={cn(
            'h-3.5 w-3.5 shrink-0 text-ink-fade transition-transform',
            open && 'rotate-90',
          )}
        />
        <Icon
          className={cn(
            'h-3.5 w-3.5 shrink-0',
            isTask ? 'text-accent-blue' : iconColorForState(call.state),
          )}
        />
        <span className="font-mono text-[0.78rem] text-ink">{labelFor(call)}</span>
        {!isTask && argsPreview ? (
          <span className="truncate font-mono text-[0.74rem] text-ink-fade">
            {argsPreview}
          </span>
        ) : null}
        <span className="ml-auto font-mono text-[0.62rem] uppercase tracking-[0.16em] text-ink-fade">
          {STATE_LABEL[call.state]}
        </span>
      </button>

      {open ? (
        <div className="border-t border-line bg-paper-warm/40 px-3 py-2.5">
          {nestedTranscript ? (
            <Section label="sub-agent transcript">{nestedTranscript}</Section>
          ) : null}
          <Section label="arguments">
            <pre className="m-0 overflow-x-auto whitespace-pre-wrap break-words font-mono text-[0.75rem] text-ink-soft">
              {prettyArgs(call)}
            </pre>
          </Section>
          {call.waitPrompt ? (
            <Section label="waiting on">
              <div className="text-[0.8rem] text-ink-soft">{call.waitPrompt}</div>
            </Section>
          ) : null}
          {call.error ? (
            <Section label="error">
              <pre className="m-0 overflow-x-auto whitespace-pre-wrap break-words font-mono text-[0.75rem] text-accent-warm">
                {call.error}
              </pre>
            </Section>
          ) : null}
          {call.result ? (
            <Section label="result">
              <pre className="m-0 max-h-[14rem] overflow-auto whitespace-pre-wrap break-words font-mono text-[0.75rem] text-ink-soft">
                {prettyResult(call.result)}
              </pre>
            </Section>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mt-2 first:mt-0">
      <div className="mb-1 font-mono text-[0.62rem] uppercase tracking-[0.16em] text-ink-fade">
        {label}
      </div>
      {children}
    </div>
  )
}

function labelFor(call: ToolCallEntry): string {
  if (call.name === 'task') {
    const sub =
      call.subagent ??
      (call.args && typeof call.args.subagent === 'string'
        ? (call.args.subagent as string)
        : null)
    const suffix = call.state === 'streaming' || call.state === 'pending' ? '…' : ''
    if (call.state === 'error') return sub ? `task → ${sub} failed` : 'task failed'
    return sub ? `task → ${sub}${suffix}` : `task${suffix}`
  }
  return call.name
}

function iconForState(state: ToolCallState) {
  switch (state) {
    case 'ok':
    case 'resolved':
      return Check
    case 'error':
      return AlertCircle
    case 'waiting':
      return Hourglass
    default:
      return Loader2
  }
}

function iconColorForState(state: ToolCallState) {
  switch (state) {
    case 'ok':
    case 'resolved':
      return 'text-accent-leaf'
    case 'error':
      return 'text-accent-warm'
    case 'waiting':
      return 'text-accent-blue'
    default:
      return 'animate-spin text-ink-mute'
  }
}

function prettyArgs(call: ToolCallEntry): string {
  if (call.args) {
    try {
      return JSON.stringify(call.args, null, 2)
    } catch {
      /* fall through */
    }
  }
  return call.argsText || '{}'
}

function prettyResult(text: string): string {
  const trimmed = text.trim()
  const looksObject = trimmed.startsWith('{') && trimmed.endsWith('}')
  const looksArray = trimmed.startsWith('[') && trimmed.endsWith(']')
  if (!(looksObject || looksArray)) return text
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2)
  } catch {
    return text
  }
}

function formatArgsPreview(call: ToolCallEntry): string {
  const text = call.argsText || (call.args ? JSON.stringify(call.args) : '')
  if (!text) return ''
  const one = text.replace(/\s+/g, ' ').trim()
  return one.length <= 60 ? one : one.slice(0, 57) + '…'
}
