import { expect, test, type Page } from '@playwright/test'

async function send(page: Page, text: string) {
  const composer = page.getByPlaceholder(/ask anything|reply/)
  await composer.fill(text)
  await page.getByRole('button', { name: 'Send' }).click()
}

async function recordingPause(page: Page, milliseconds = 1_000) {
  if (process.env.ACTANT_RECORD_VIDEO === '1') {
    await page.waitForTimeout(milliseconds)
  }
}

test('exercises streaming, durable approval, questions, and nested subagents', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await send(page, 'Show me an approval')
  await expect(page.getByText('approval needed', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toBeVisible()

  await send(page, 'Ask me a question')
  await expect(page.getByText('agent is asking', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Streaming response' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toHaveCount(2)

  await send(page, 'Delegate this to a subagent')
  await expect(
    page.getByRole('button', { name: 'task → researcher resolved' }),
  ).toBeVisible()
  // Reload to verify that the viewer reconstructs both levels from
  // persisted projections after all live events have completed.
  await page.reload()
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'task → summarizer ok' }),
  ).toBeVisible()
  await expect(
    page.getByText('Durable delegation verified', { exact: true }),
  ).toBeVisible()
})

test('holds mixed tool groups and surfaces nested subagent approval', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()

  await send(page, 'Run a mixed parallel tool group')
  await expect(
    page.getByText('get_current_time', { exact: true }),
  ).toBeVisible()
  await expect(page.getByText('approval needed', { exact: true })).toBeVisible()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toBeVisible()

  await send(page, 'Delegate an approval task to a subagent')
  await expect(page.getByText('approval needed', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByRole('button', { name: 'task → researcher resolved' }),
  ).toBeVisible()
})

test('has a natural deferred question and parallel weather tools', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await recordingPause(page)

  await send(page, 'Help me choose a pizza for tonight')
  await expect(page.getByText('agent is asking', { exact: true })).toBeVisible()
  await expect(
    page.getByText('What kind of pizza sounds good right now?', {
      exact: true,
    }),
  ).toBeVisible()
  await recordingPause(page, 1_500)
  await page.getByRole('button', { name: 'Surprise me' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toBeVisible()
  await recordingPause(page)

  await send(page, 'What is the weather in New York, London, and Tokyo?')
  await expect(page.getByRole('button', { name: /^get_weather/ })).toHaveCount(
    3,
  )
  await expect(page.getByText(/Share New York.*weather service/)).toBeVisible()
  await recordingPause(page, 1_500)
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(page.getByText(/Share London.*weather service/)).toBeVisible({
    timeout: 15_000,
  })
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toHaveCount(1)
  await recordingPause(page, 1_200)
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(page.getByText(/Share Tokyo.*weather service/)).toBeVisible({
    timeout: 15_000,
  })
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toHaveCount(1)
  await recordingPause(page, 1_200)
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toHaveCount(2, { timeout: 15_000 })
  await recordingPause(page, 2_000)
})

test('continues the same thread after cancelling a partially resolved group', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()

  await send(page, 'Check the weather in New York, London, and Tokyo')
  await expect(page.getByRole('button', { name: /^get_weather/ })).toHaveCount(
    3,
  )
  await expect(page.getByText(/Share New York.*weather service/)).toBeVisible()
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(page.getByText(/Share London.*weather service/)).toBeVisible({
    timeout: 15_000,
  })

  const threadId = new URL(page.url()).pathname.split('/').at(-1)
  const cancelled = await page.request.delete(
    `http://localhost:8181/api/threads/${encodeURIComponent(threadId ?? '')}`,
  )
  expect(cancelled.status()).toBe(204)

  await expect
    .poll(async () => {
      const response = await page.request.get(
        'http://localhost:8181/api/threads',
      )
      const threads = (await response.json()) as Array<{
        id: string
        status: string
      }>
      return threads.find((thread) => thread.id === threadId)?.status
    })
    .toBe('cancelled')
  await page.reload()
  await expect(page.getByText('connected', { exact: true })).toBeVisible()

  await send(page, 'Sorry, can you continue what you were doing?')
  await expect(page.getByRole('button', { name: /^get_weather/ })).toHaveCount(
    4,
    {
      timeout: 15_000,
    },
  )
  await expect(page.getByText('approval needed', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByText('Done — the result came back', { exact: false }),
  ).toBeVisible()
})
