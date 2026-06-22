import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

type Props = {
  icon: LucideIcon
  label: string
  meta?: string
  active?: boolean
  open: boolean
  disabled?: boolean
  onClick: () => void
}

/** Sidebar nav row: icon + label, with a rounded background on hover/active.
 * When `open=false` the row collapses to an icon-only square. */
export function ThreadNavItem({ icon: Icon, label, meta, active, open, disabled, onClick }: Props) {
  return (
    <button
      aria-current={active ? 'page' : undefined}
      aria-label={open ? undefined : label}
      className={cn(
        'group inline-flex w-full items-center gap-2.5 rounded-md border-0 bg-transparent px-2 py-1.5 text-left text-ink transition-colors duration-100',
        'hover:bg-ink/5',
        active && 'bg-ink/[0.07] hover:bg-ink/10',
        disabled && 'cursor-not-allowed opacity-55 hover:bg-transparent',
        !open && 'h-9 w-9 justify-center gap-0 px-0',
      )}
      disabled={disabled}
      onClick={onClick}
      title={open ? undefined : label}
      type="button"
    >
      <Icon
        className={cn(
          'shrink-0 text-ink-mute transition-colors',
          !disabled && 'group-hover:text-ink',
          active && 'text-ink',
        )}
        size={16}
        strokeWidth={1.7}
      />
      {open ? (
        <span className="flex min-w-0 flex-1 flex-col gap-px">
          <span className="overflow-hidden whitespace-nowrap text-ellipsis text-[0.85rem] leading-tight">
            {label}
          </span>
          {meta ? <span className="font-mono text-[0.62rem] text-ink-fade">{meta}</span> : null}
        </span>
      ) : null}
    </button>
  )
}
