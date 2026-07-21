import { Sparkles } from 'lucide-react'

export function EmptyStage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-line bg-white/80 text-ink-mute">
        <Sparkles className="h-5 w-5" />
      </div>
      <div className="text-[1.05rem] font-medium text-ink">Actant demo</div>
    </div>
  )
}
