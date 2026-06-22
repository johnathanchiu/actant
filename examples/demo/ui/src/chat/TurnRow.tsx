import { ChevronRight, Sparkles } from 'lucide-react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'
import { AssistantMarkdown } from './AssistantMarkdown'
import { NestedTranscript } from './NestedTranscript'
import { ToolCallRow } from './ToolCallRow'
import { formatTimestamp } from './formatTimestamp'
import type { TurnEntry } from './state'
import type { SubThreadMap } from './subThreads'

type Props = {
  turn: TurnEntry
  model: string | null
  subThreads: SubThreadMap
  /** Compact variant for nested transcripts — no footer, tighter spacing. */
  compact?: boolean
}

export function TurnRow({ turn, model, subThreads, compact = false }: Props) {
  const hasText = turn.text.length > 0
  const hasToolCalls = turn.toolCalls.length > 0
  const hasThinking = turn.thinking.length > 0
  const showStreamingDot =
    turn.isStreaming && !hasText && !hasToolCalls && !hasThinking

  return (
    <div
      className={cn(
        'flex animate-[fade-rise_240ms_ease-out_both] items-start gap-3.5',
        compact && 'gap-2',
      )}
    >
      <div
        className={cn('flex min-w-0 flex-1 flex-col', compact ? 'gap-2' : 'gap-3.5')}
      >
        {hasThinking ? (
          <ThinkingBlock streaming={turn.isStreaming && !hasText} thinking={turn.thinking} />
        ) : null}
        {hasText ? <AssistantMarkdown naked={compact} text={turn.text} /> : null}
        {hasToolCalls ? (
          <div className="flex flex-col gap-1 pl-0.5">
            {turn.toolCalls.map((call) => {
              const sub = call.subThreadId ? subThreads[call.subThreadId] : undefined
              return (
                <ToolCallRow
                  call={call}
                  key={call.id}
                  nestedTranscript={
                    sub ? (
                      <NestedTranscript activity={sub} model={model} subThreads={subThreads} />
                    ) : call.subThreadId ? (
                      <div className="text-[0.78rem] italic text-ink-fade">
                        sub-agent is starting…
                      </div>
                    ) : undefined
                  }
                />
              )
            })}
          </div>
        ) : null}
        {showStreamingDot ? (
          <span
            aria-hidden="true"
            className="inline-block h-1.5 w-1.5 animate-[turn-streaming-pulse_1.2s_ease-in-out_infinite] rounded-full bg-ink-mute"
          />
        ) : null}
        {!compact && (hasText || hasToolCalls || hasThinking) ? (
          <div className="mt-1 flex items-center gap-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-ink-fade">
            <span>assistant</span>
            {model ? (
              <>
                <span>·</span>
                <span>{model}</span>
              </>
            ) : null}
            <span>·</span>
            <span>{formatTimestamp(turn.timestamp)}</span>
          </div>
        ) : null}
      </div>
    </div>
  )
}

function ThinkingBlock({ streaming, thinking }: { streaming: boolean; thinking: string }) {
  return (
    <Collapsible defaultOpen={streaming}>
      <CollapsibleTrigger className="group/thinking inline-flex items-center gap-1.5 rounded-md border border-line bg-ink/[0.03] px-2 py-1 text-left font-mono text-[0.7rem] uppercase tracking-[0.18em] text-ink-soft transition-colors hover:border-line-strong hover:bg-white">
        <Sparkles className="text-accent-leaf" size={12} />
        <span>Thinking{streaming ? '…' : ''}</span>
        <ChevronRight
          className="transition-transform group-data-[state=open]/thinking:rotate-90"
          size={12}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2">
        <div className="max-h-64 overflow-y-auto border-l-2 border-line pl-3 text-[0.86rem] italic leading-relaxed text-ink-mute">
          <AssistantMarkdown naked text={thinking} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
