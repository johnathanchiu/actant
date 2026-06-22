import { Streamdown } from 'streamdown'
import { cn } from '@/lib/utils'

/** Renders assistant markdown. Streamdown handles partial markdown gracefully
 * during token streaming (e.g. an unclosed ``` fence won't break layout).
 *
 * Two variants:
 * - default: bubble chrome (rounded white border, used for top-level
 *   assistant text in the main message list).
 * - naked: prose only (no border, no bg, no rounding). Used inside
 *   nested transcripts and thinking blocks where the surrounding
 *   container already provides the visual frame and another bubble
 *   would read as a double-box.
 */
export function AssistantMarkdown({
  text,
  naked = false,
}: {
  text: string
  naked?: boolean
}) {
  return (
    <div
      className={cn(
        'max-w-[44rem] break-words text-[0.94rem] leading-[1.6] text-ink',
        '[&_pre]:my-2 [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-ink/[0.04] [&_pre]:p-2.5 [&_pre]:text-[0.82rem]',
        '[&_code]:font-mono [&_:not(pre)>code]:rounded [&_:not(pre)>code]:bg-ink/[0.06] [&_:not(pre)>code]:px-1.5 [&_:not(pre)>code]:py-0.5 [&_:not(pre)>code]:text-[0.85em]',
        '[&_p]:my-0 [&_p+p]:mt-2.5 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5',
        '[&_a]:text-accent-blue [&_a]:underline [&_a]:underline-offset-2',
        !naked &&
          'rounded-[1.1rem] rounded-bl-[0.4rem] border border-line bg-white/85 px-4 py-3 backdrop-blur-md',
      )}
    >
      <Streamdown controls={{ table: false, mermaid: false, code: false }} lineNumbers={false}>
        {text}
      </Streamdown>
    </div>
  )
}
