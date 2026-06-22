import { AlertCircle, Info } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { EventEntry } from './state'

export function EventRow({ entry }: { entry: EventEntry }) {
  const Icon = entry.level === 'error' ? AlertCircle : Info
  return (
    <div
      className={cn(
        'flex max-w-[44rem] items-start gap-2 rounded-[0.85rem] border px-3 py-2 text-[0.82rem]',
        entry.level === 'error'
          ? 'border-accent-warm/40 bg-accent-warm/5 text-accent-warm'
          : 'border-line bg-white/70 text-ink-soft',
      )}
    >
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div className="whitespace-pre-wrap break-words">{entry.text}</div>
    </div>
  )
}
