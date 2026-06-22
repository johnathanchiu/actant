/**
 * Typed HTTP client for the demo server.
 *
 * Wraps the routes exposed by ``examples/demo/server/app/routes.py``. No
 * SSE here — that lives in sseClient.ts. This module is REST only.
 */

export type AgentInfo = {
  id: string
  name: string
  model: string
  tools: string[]
}

export type ThreadSummary = {
  id: string
  agent_id: string
  status: string
  turn_count: number
  message_count: number
  preview: string
}

export type PersistedToolCall = {
  id: string
  function: { name: string; arguments: string }
  type?: string
}

export type PersistedMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | Array<Record<string, unknown>> | null
  tool_calls?: PersistedToolCall[]
  tool_call_id?: string
  name?: string
  thought_summary?: string | null
}

export type WaitingToolCall = {
  tool_call_id: string
  name: string
  args: Record<string, unknown>
  prompt: string | null
  wait_request: {
    kind?: string
    prompt?: string
    payload?: Record<string, unknown>
  } | null
}

export type SubThreadLink = {
  sub_thread_id: string
  parent_thread_id: string
  parent_tool_call_id: string
}

/** Resolve-tool payload that the UI sends when the user replies to a
 * deferred tool call. Mirrors ``actant.tools.admission.ToolResolution``. */
export type ResolveBody = {
  approved?: boolean
  answer?: string
  payload?: Record<string, unknown>
}

export type Api = {
  baseUrl: string
  fetchAgent: () => Promise<AgentInfo>
  fetchThreads: () => Promise<ThreadSummary[]>
  fetchMessages: (threadId: string) => Promise<PersistedMessage[]>
  fetchWaitingToolCalls: (threadId: string) => Promise<WaitingToolCall[]>
  fetchSubThreads: (threadId: string) => Promise<SubThreadLink[]>
  sendMessage: (threadId: string, content: string) => Promise<void>
  resolveTool: (
    threadId: string,
    toolCallId: string,
    body: ResolveBody,
  ) => Promise<void>
}

export function createApi(baseUrl: string): Api {
  async function getJson<T>(path: string): Promise<T> {
    const r = await fetch(`${baseUrl}${path}`)
    if (!r.ok) throw new Error(`${path} → ${r.status}: ${await r.text()}`)
    return (await r.json()) as T
  }
  async function postJson(path: string, body: unknown, expectedStatus: number): Promise<void> {
    const r = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (r.status !== expectedStatus) {
      throw new Error(`${path} → ${r.status}: ${await r.text()}`)
    }
  }
  return {
    baseUrl,
    fetchAgent: () => getJson<AgentInfo>('/api/agent'),
    fetchThreads: () => getJson<ThreadSummary[]>('/api/threads'),
    fetchMessages: (id) =>
      getJson<PersistedMessage[]>(`/api/threads/${encodeURIComponent(id)}/messages`),
    fetchWaitingToolCalls: (id) =>
      getJson<WaitingToolCall[]>(
        `/api/threads/${encodeURIComponent(id)}/waiting_tool_calls`,
      ),
    fetchSubThreads: (id) =>
      getJson<SubThreadLink[]>(`/api/threads/${encodeURIComponent(id)}/sub_threads`),
    sendMessage: (id, content) =>
      postJson(`/api/threads/${encodeURIComponent(id)}/messages`, { content }, 202),
    resolveTool: (threadId, toolCallId, body) =>
      postJson(
        `/api/threads/${encodeURIComponent(threadId)}/tool_calls/${encodeURIComponent(toolCallId)}/resolve`,
        body,
        204,
      ),
  }
}
