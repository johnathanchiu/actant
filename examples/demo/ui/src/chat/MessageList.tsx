import { type RefObject } from 'react'
import { EventRow } from './EventRow'
import { TurnRow } from './TurnRow'
import { UserRow } from './UserRow'
import { useStickyScroll } from './useStickyScroll'
import type { ConsoleEntry } from './state'
import type { SubThreadMap } from './subThreads'

type Props = {
  entries: ConsoleEntry[]
  subThreads: SubThreadMap
  scrollRef: RefObject<HTMLDivElement | null>
  model: string | null
}

export function MessageList({ entries, subThreads, scrollRef, model }: Props) {
  useStickyScroll(scrollRef, entries.length)

  return (
    <div className="flex h-full flex-col gap-5 px-6 pb-10 pt-6">
      {entries.map((entry) => {
        if (entry.kind === 'user') return <UserRow entry={entry} key={entry.id} />
        if (entry.kind === 'event') return <EventRow entry={entry} key={entry.id} />
        return (
          <TurnRow
            key={entry.id}
            model={model}
            subThreads={subThreads}
            turn={entry}
          />
        )
      })}
    </div>
  )
}
