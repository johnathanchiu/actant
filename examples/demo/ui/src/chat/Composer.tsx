import { ArrowUp } from 'lucide-react'
import { type FormEvent, type KeyboardEvent, useState } from 'react'
import TextareaAutosize from 'react-textarea-autosize'
import { cn } from '@/lib/utils'

type Props = {
  hero: boolean
  disabled: boolean
  onSubmit: (text: string) => void
}

export function Composer({ hero, disabled, onSubmit }: Props) {
  const [value, setValue] = useState('')

  function submit() {
    const text = value.trim()
    if (!text || disabled) return
    onSubmit(text)
    setValue('')
  }

  function handleFormSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  const placeholder = hero ? 'ask anything' : 'reply'
  const sendDisabled = disabled || value.trim().length === 0

  return (
    <form className="flex flex-col gap-2 py-2 pb-3.5" onSubmit={handleFormSubmit}>
      <div
        className={cn(
          'relative flex min-h-[5.5rem] flex-col rounded-[1.1rem] border border-line-strong bg-white/95 px-2 py-2.5 transition-[border-color,box-shadow] duration-200 focus-within:border-ink/30 focus-within:shadow-[0_0_0_3px_rgba(17,17,17,0.04)]',
          hero && 'min-h-[9rem] rounded-[1.5rem] px-4 pt-[1.15rem] pb-3.5',
        )}
      >
        <TextareaAutosize
          className={cn(
            'w-full resize-none overflow-y-auto border-0 bg-transparent px-2 py-1.5 text-[0.96rem] leading-snug text-ink outline-none placeholder:text-ink/35',
            hero && 'text-[1.02rem] leading-relaxed',
          )}
          disabled={disabled}
          maxRows={hero ? 8 : 6}
          minRows={1}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          spellCheck={false}
          value={value}
        />
        <div className="mt-auto flex items-center justify-end gap-2 pt-2">
          <button
            aria-label="Send"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-ink bg-ink text-paper transition-opacity duration-150 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
            disabled={sendDisabled}
            title="Send (Enter)"
            type="submit"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
      </div>
    </form>
  )
}
