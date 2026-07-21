import { expect, test } from '@playwright/test'

test.skip(
  process.env.ACTANT_LIVE_DEMO !== '1',
  'Live demo requires an explicitly configured provider and API key',
)

test('records a real model using QA and parallel deferred weather calls', async ({
  page,
}) => {
  test.setTimeout(120_000)

  await page.goto('/')
  await expect(page.getByText('connected', { exact: true })).toBeVisible()
  await expect(page.getByText('gpt-5.4-nano', { exact: true })).toHaveCount(0)
  await page.waitForTimeout(2_000)

  const composer = page.getByPlaceholder(/ask anything|reply/)
  const pizzaPrompt =
    'Help me choose a pizza. First ask me one multiple-choice question with these options: Pepperoni, Margherita, Mushroom, or Surprise me.'
  await composer.pressSequentially(pizzaPrompt, { delay: 45 })
  await page.waitForTimeout(800)
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByText('agent is asking', { exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await page.waitForTimeout(3_500)
  await page
    .getByRole('button', {
      name: /^(Pepperoni|Margherita|Mushroom|Surprise me)$/,
    })
    .first()
    .click()
  await expect(page.getByText('agent is asking', { exact: true })).toHaveCount(0, {
    timeout: 20_000,
  })
  await page.waitForTimeout(4_000)

  const weatherPrompt =
    'What is the weather in New York, London, and Tokyo? Compare all three for me.'
  await composer.pressSequentially(weatherPrompt, { delay: 55 })
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
  await expect(page.getByText('assistant', { exact: true })).toHaveCount(4, {
    timeout: 30_000,
  })
  await page.getByText('assistant', { exact: true }).last().scrollIntoViewIfNeeded()
  await page.waitForTimeout(8_000)
})
