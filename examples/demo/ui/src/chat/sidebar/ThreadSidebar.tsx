import { MessageSquare, PanelLeft, SquarePen } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ThreadSummary } from '../api'
import { ThreadNavItem } from './ThreadNavItem'

type Props = {
  threads: ThreadSummary[]
  activeThreadId: string | null
  open: boolean
  onToggle: () => void
  onSelectThread: (threadId: string) => void
  onNewThread: () => void
  newDisabled: boolean
}

export function ThreadSidebar({
  threads,
  activeThreadId,
  open,
  onToggle,
  onSelectThread,
  onNewThread,
  newDisabled,
}: Props) {
  return (
    <aside
      aria-label="Threads"
      className={cn(
        'flex shrink-0 flex-col overflow-hidden border-r border-line bg-paper-warm/45 transition-[width] duration-200',
        open ? 'w-64' : 'w-14',
      )}
    >
      <div className={cn('flex shrink-0 flex-col gap-1 px-1.5 py-2', !open && 'items-center px-0')}>
        <div className={cn('flex items-center', open ? 'justify-end' : 'justify-center')}>
          <button
            aria-label={open ? 'Hide threads' : 'Show threads'}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border-0 bg-transparent text-ink-mute transition-colors hover:bg-ink/5 hover:text-ink"
            onClick={onToggle}
            type="button"
          >
            <PanelLeft size={16} />
          </button>
        </div>
        <div className={cn('flex items-center', open ? 'justify-stretch' : 'justify-center')}>
          <ThreadNavItem
            disabled={newDisabled}
            icon={SquarePen}
            label="New thread"
            onClick={onNewThread}
            open={open}
          />
        </div>
      </div>

      {open ? (
        <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-1.5 pb-2.5">
          {threads.length === 0 ? (
            <div className="px-2 py-3 text-[0.78rem] text-ink-fade">No threads yet</div>
          ) : (
            threads.map((thread) => {
              const active = thread.id === activeThreadId
              const label = thread.preview || `Thread ${thread.id.slice(0, 8)}`
              const meta =
                thread.message_count > 0
                  ? `${thread.message_count} msg${thread.message_count === 1 ? '' : 's'}`
                  : 'empty'
              return (
                <ThreadNavItem
                  active={active}
                  icon={MessageSquare}
                  key={thread.id}
                  label={label}
                  meta={meta}
                  onClick={() => onSelectThread(thread.id)}
                  open
                />
              )
            })
          )}
        </div>
      ) : null}
    </aside>
  )
}
