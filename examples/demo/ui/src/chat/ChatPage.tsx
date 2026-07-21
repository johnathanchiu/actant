import { useCallback, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import type { AgentInfo, Api } from './api'
import { Composer } from './Composer'
import { DeferredPanel } from './DeferredPanel'
import { EmptyStage } from './EmptyStage'
import { MessageList } from './MessageList'
import { ThreadSidebar } from './sidebar/ThreadSidebar'
import { useAgentConsole } from './useAgentConsole'
import { useThreadsList } from './useThreadsList'

function newThreadId(): string {
  return `thread_${Math.random().toString(36).slice(2, 10)}`
}

type Props = {
  api: Api
  agent: AgentInfo | null
  agentError: string | null
}

export function ChatPage({ api, agent, agentError }: Props) {
  const { threadId } = useParams<{ threadId: string }>()
  const navigate = useNavigate()
  const scrollRef = useRef<HTMLDivElement>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [refreshSignal, setRefreshSignal] = useState(0)

  const activeThreadId = threadId ?? ''
  const { threads } = useThreadsList(api, refreshSignal)

  const onSelectThread = useCallback(
    (id: string) => navigate(`/t/${encodeURIComponent(id)}`),
    [navigate],
  )
  const onNewThread = useCallback(
    () => navigate(`/t/${encodeURIComponent(newThreadId())}`),
    [navigate],
  )
  const bumpThreads = useCallback(() => setRefreshSignal((n) => n + 1), [])

  if (!activeThreadId) {
    return (
      <div className="flex h-screen items-center justify-center text-[0.85rem] text-ink-fade">
        Loading…
      </div>
    )
  }

  return (
    <div className="flex h-screen">
      <ThreadSidebar
        activeThreadId={activeThreadId}
        newDisabled={false}
        onNewThread={onNewThread}
        onSelectThread={onSelectThread}
        onToggle={() => setSidebarOpen((v) => !v)}
        open={sidebarOpen}
        threads={threads}
      />
      <ChatPane
        agent={agent}
        agentError={agentError}
        api={api}
        onMessageSent={bumpThreads}
        scrollRef={scrollRef}
        threadId={activeThreadId}
      />
    </div>
  )
}

type ChatPaneProps = {
  api: Api
  agent: AgentInfo | null
  agentError: string | null
  threadId: string
  onMessageSent: () => void
  scrollRef: React.RefObject<HTMLDivElement | null>
}

function ChatPane({
  api,
  agent,
  agentError,
  threadId,
  onMessageSent,
  scrollRef,
}: ChatPaneProps) {
  const visibleModel =
    agent?.model === 'demo/deterministic' ? null : (agent?.model ?? null)
  const {
    entries,
    subThreads,
    streamState,
    sending,
    isLoadingHistory,
    historyError,
    isEmpty,
    sendMessage,
    resolveTool,
  } = useAgentConsole({ api, threadId, onMessageSent })

  const statusLabel = useMemo(() => {
    if (streamState === 'open') return 'connected'
    if (streamState === 'connecting') return 'connecting…'
    if (streamState === 'reconnecting') return 'reconnecting…'
    return 'stream error'
  }, [streamState])

  return (
    <div className="mx-auto flex h-full max-w-[52rem] flex-1 flex-col">
      <header className="flex items-center justify-between border-b border-line px-6 py-3.5">
        <div className="flex items-center gap-2 font-mono text-[0.72rem] uppercase tracking-[0.2em] text-ink">
          actant <span className="text-ink-fade">demo</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-ink-fade">
          <span>thread {threadId.slice(7)}</span>
          <span>·</span>
          <span>{statusLabel}</span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto" ref={scrollRef}>
          {isLoadingHistory ? (
            <div className="flex h-full items-center justify-center text-[0.85rem] text-ink-fade">
              Loading thread…
            </div>
          ) : isEmpty ? (
            agentError ? (
              <ErrorStage title="Could not reach the demo server." detail={agentError} />
            ) : historyError ? (
              <ErrorStage title="Could not load thread history." detail={historyError} />
            ) : (
              <EmptyStage />
            )
          ) : (
            <MessageList
              entries={entries}
              model={visibleModel}
              scrollRef={scrollRef}
              subThreads={subThreads}
            />
          )}
        </div>
        <div className="bg-paper px-6 pb-2 pt-4">
          <DeferredPanel
            entries={entries}
            onResolve={resolveTool}
            subThreads={subThreads}
          />
          <Composer
            disabled={sending || streamState !== 'open' || isLoadingHistory}
            hero={isEmpty && !isLoadingHistory}
            onSubmit={sendMessage}
          />
        </div>
      </div>
    </div>
  )
}

function ErrorStage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-[0.85rem] text-accent-warm">
      <div>{title}</div>
      <div className="font-mono text-[0.75rem] text-ink-fade">{detail}</div>
    </div>
  )
}
