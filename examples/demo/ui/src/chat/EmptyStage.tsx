import { Sparkles } from 'lucide-react'

export function EmptyStage({ model, tools }: { model: string | null; tools: string[] }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-line bg-white/80 text-ink-mute">
        <Sparkles className="h-5 w-5" />
      </div>
      <div className="max-w-[28rem] space-y-1.5">
        <div className="text-[1.05rem] font-medium text-ink">Actant demo</div>
        <div className="text-[0.88rem] leading-relaxed text-ink-soft">
          A durable, Temporal-backed agent runtime. Send a message to start; the
          server runs one workflow per thread, streams text and tool calls back
          over SSE, and parks deferred tool calls without burning compute.
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 font-mono text-[0.7rem] uppercase tracking-[0.15em] text-ink-fade">
        {model ? <span>{model}</span> : null}
        {tools.length ? (
          <>
            {model ? <span>·</span> : null}
            <span>tools: {tools.join(', ')}</span>
          </>
        ) : null}
      </div>
    </div>
  )
}
