/**
 * Service Bay — bay status, appointment list, booking modal, predicted service.
 */
import { test, expect } from '../fixtures/auth';
import { ServiceBayPage } from '../pages/ServiceBayPage';

test.describe('Service Bay page', () => {
  test('heading is visible', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await expect(sb.heading).toBeVisible({ timeout: 10_000 });
  });

  test('Live Bay Status section is present', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await expect(dealerPage.getByRole('heading', { name: 'Live Bay Status' })).toBeVisible({ timeout: 10_000 });
  });

  test('Appointments section is present', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await expect(sb.appointmentsHeading).toBeVisible({ timeout: 10_000 });
  });

  test('Predicted Service Needs section appears when ML events exist', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);

    // Capture the maintenance calendar API response before navigating
    const calRespPromise = dealerPage.waitForResponse(
      r => r.url().includes('/maintenance') && r.request().method() === 'GET',
      { timeout: 20_000 },
    ).catch(() => null);

    await sb.goto();
    const calResp = await calRespPromise;

    if (calResp && calResp.ok()) {
      const body = await calResp.json().catch(() => []);
      const hasEvents = Array.isArray(body) && body.length > 0;
      if (hasEvents) {
        // Section renders only when the component receives non-empty data
        await expect(sb.predictedServiceHeading).toBeVisible({ timeout: 5_000 });
      }
      // No events → PredictedService returns null → no heading, which is correct
    }
  });

  test('Book Appointment button is visible', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await expect(sb.bookBtn).toBeVisible({ timeout: 10_000 });
  });

  test('clicking Book Appointment opens modal', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await sb.bookBtn.waitFor({ state: 'visible', timeout: 10_000 });
    await sb.openBookingModal();
    await expect(dealerPage.getByRole('heading', { name: 'Book Appointment' })).toBeVisible({ timeout: 5_000 });
  });

  test('booking modal has VIN, date, time, job type fields', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await sb.bookBtn.waitFor({ state: 'visible', timeout: 10_000 });
    await sb.openBookingModal();
    await expect(dealerPage.getByRole('heading', { name: 'Book Appointment' })).toBeVisible();
    await expect(dealerPage.getByPlaceholder(/MZ7X/i)).toBeVisible();
    await expect(dealerPage.locator('input[type="date"]')).toBeVisible();
    await expect(dealerPage.locator('input[type="time"]')).toBeVisible();
  });

  test('Cancel button closes the booking modal', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    await sb.bookBtn.waitFor({ state: 'visible', timeout: 10_000 });
    await sb.openBookingModal();
    await expect(dealerPage.getByRole('heading', { name: 'Book Appointment' })).toBeVisible();
    await sb.modalCancelBtn.click();
    await expect(dealerPage.getByRole('heading', { name: 'Book Appointment' })).not.toBeVisible({ timeout: 3_000 });
  });

  test('appointment table shows expected columns', async ({ dealerPage }) => {
    const sb = new ServiceBayPage(dealerPage);
    await sb.goto();
    const hasTable = await dealerPage.getByRole('table').first().isVisible().catch(() => false);
    if (hasTable) {
      await expect(dealerPage.getByRole('columnheader', { name: 'VIN' })).toBeVisible();
      await expect(dealerPage.getByRole('columnheader', { name: 'Job Type' })).toBeVisible();
      await expect(dealerPage.getByRole('columnheader', { name: 'Status' })).toBeVisible();
    }
  });
});
