import { type RefObject, useEffect, useRef } from 'react'

export function useStickyScroll<S>(
  scrollRef: RefObject<HTMLDivElement | null>,
  signal: S | undefined,
) {
  const stickToBottomRef = useRef(true)

  useEffect(() => {
    const node = scrollRef.current
    if (!node) return
    function onScroll() {
      if (!node) return
      const distance = node.scrollHeight - node.scrollTop - node.clientHeight
      stickToBottomRef.current = distance < 80
    }
    node.addEventListener('scroll', onScroll, { passive: true })
    return () => node.removeEventListener('scroll', onScroll)
  }, [scrollRef])

  useEffect(() => {
    if (signal === undefined) return
    const node = scrollRef.current
    if (!node || !stickToBottomRef.current) return
    node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' })
  }, [scrollRef, signal])
}
