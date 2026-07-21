import { expect, test } from '@playwright/test'

test.skip(
  process.env.ACTANT_LIVE_DEMO !== '1',
  'Live demo requires an explicitly configured provider and API key',
)

test('records a real model producing parallel deferred weather calls', async ({
  page,
}) => {
  test.setTimeout(120_000)

  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await expect(page.getByText('gpt-5.4-nano', { exact: true })).toHaveCount(0)
  await page.waitForTimeout(2_000)

  const prompt =
    'What is the weather in New York, London, and Tokyo? Compare all three for me.'
  const composer = page.getByPlaceholder(/ask anything|reply/)
  await composer.pressSequentially(prompt, { delay: 55 })
  await page.waitForTimeout(800)
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByRole('button', { name: /^get_weather/ })).toHaveCount(
    3,
    { timeout: 30_000 },
  )

  for (let approval = 0; approval < 3; approval += 1) {
    await expect(page.getByText('approval needed', { exact: true })).toBeVisible({
      timeout: 20_000,
    })
    await page.waitForTimeout(3_500)
    await page.getByRole('button', { name: 'Approve' }).click()
  }

  await expect(page.getByText('approval needed', { exact: true })).toHaveCount(0, {
    timeout: 20_000,
  })
  await expect(page.getByText('assistant', { exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await page.waitForTimeout(8_000)
})
