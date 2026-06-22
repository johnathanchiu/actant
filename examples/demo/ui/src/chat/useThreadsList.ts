import { useCallback, useEffect, useState } from 'react'
import type { Api, ThreadSummary } from './api'

/** Lightweight polling-free thread list: fetches on mount and whenever
 * `refreshSignal` changes. Callers bump the signal after sending a message
 * or creating a thread so the sidebar picks up new state. */
export function useThreadsList(api: Api, refreshSignal: number) {
  const [threads, setThreads] = useState<ThreadSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await api.fetchThreads()
      setThreads(list)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [api])

  useEffect(() => {
    void refresh()
  }, [refresh, refreshSignal])

  return { threads, loading, error, refresh }
}
