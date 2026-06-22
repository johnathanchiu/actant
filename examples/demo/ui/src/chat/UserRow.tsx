import type { UserEntry } from './state'
import { formatTimestamp } from './formatTimestamp'

export function UserRow({ entry }: { entry: UserEntry }) {
  return (
    <div className="flex animate-[fade-rise_240ms_ease-out_both] items-start justify-end gap-3.5">
      <div className="min-w-0">
        <div className="max-w-[38rem] whitespace-pre-wrap break-words rounded-[1.1rem] rounded-br-[0.4rem] border border-ink bg-ink px-5 py-4 text-[0.94rem] leading-[1.65] text-paper">
          {entry.text}
        </div>
        <div className="mt-1.5 flex items-center justify-end gap-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-ink-fade">
          <span>you</span>
          <span>·</span>
          <span>{formatTimestamp(entry.timestamp)}</span>
        </div>
      </div>
    </div>
  )
}
