import { useEffect, useMemo, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { ChatPage } from '@/chat/ChatPage'
import { createApi, type AgentInfo } from '@/chat/api'

const API_BASE = import.meta.env.VITE_ACTANT_API_BASE ?? 'http://localhost:8181'

function newThreadId(): string {
  return `thread_${Math.random().toString(36).slice(2, 10)}`
}

export function App() {
  const api = useMemo(() => createApi(API_BASE), [])
  const [agent, setAgent] = useState<AgentInfo | null>(null)
  const [agentError, setAgentError] = useState<string | null>(null)
  const [initialThreadId] = useState(newThreadId)

  useEffect(() => {
    let cancelled = false
    api
      .fetchAgent()
      .then((info) => {
        if (!cancelled) setAgent(info)
      })
      .catch((err: unknown) => {
        if (!cancelled) setAgentError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [api])

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Navigate replace to={`/t/${initialThreadId}`} />} path="/" />
        <Route element={<ChatPage agent={agent} agentError={agentError} api={api} />} path="/t/:threadId" />
        <Route element={<Navigate replace to="/" />} path="*" />
      </Routes>
    </BrowserRouter>
  )
}
