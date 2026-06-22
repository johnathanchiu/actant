import { TurnRow } from './TurnRow'
import type { SubThreadActivity, SubThreadMap } from './subThreads'

/** Renders the turns of a sub-thread inside the parent's tool-call
 * card. Visually distinguished by a thin left border. */
export function NestedTranscript({
  activity,
  model,
  subThreads,
}: {
  activity: SubThreadActivity
  model: string | null
  subThreads: SubThreadMap
}) {
  if (activity.turns.length === 0) {
    return (
      <div className="border-l-2 border-accent-blue/30 pl-3 text-[0.78rem] italic text-ink-fade">
        {activity.isStreaming ? 'sub-agent starting…' : 'no turns'}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3 border-l-2 border-accent-blue/30 pl-3">
      {activity.turns.map((turn) => (
        <TurnRow
          compact
          key={turn.id}
          model={model}
          subThreads={subThreads}
          turn={turn}
        />
      ))}
    </div>
  )
}
