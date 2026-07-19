import { expect, test, type Page } from '@playwright/test'

async function send(page: Page, text: string) {
  const composer = page.getByPlaceholder(/ask anything|reply/)
  await composer.fill(text)
  await page.getByRole('button', { name: 'Send' }).click()
}

test('exercises streaming, durable approval, questions, and nested subagents', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await expect(page.getByText('demo/deterministic', { exact: true })).toBeVisible()

  await send(page, 'Show me an approval')
  await expect(page.getByText('approval needed', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(
    page.getByText('The durable tool call completed successfully.', { exact: false }),
  ).toBeVisible()

  await send(page, 'Ask me a question')
  await expect(page.getByText('agent is asking', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Streaming response' }).click()
  await expect(
    page.getByText('The durable tool call completed successfully.', { exact: false }),
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
  await expect(page.getByText('Durable delegation verified', { exact: true })).toBeVisible()
})
