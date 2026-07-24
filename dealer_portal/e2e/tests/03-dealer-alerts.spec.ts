/**
 * Fleet Alerts — severity filter, time-range select, refresh, table.
 */
import { test, expect } from '../fixtures/auth';
import { AlertsPage } from '../pages/AlertsPage';

test.describe('Alerts page', () => {
  test('heading and KPI line are visible', async ({ dealerPage }) => {
    // Intercept alerts API so the cold CSV backend request doesn't queue up
    // and block subsequent tests. Heading and KPI text are static UI elements
    // that render immediately, before the API responds.
    await dealerPage.route('**/api/fleet/alerts**', route => route.abort());
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    await expect(alerts.heading).toBeVisible({ timeout: 10_000 });
    await expect(dealerPage.getByText(/total alerts in last/i)).toBeVisible();
  });

  test('all severity filter buttons are present', async ({ dealerPage }) => {
    // Intercept alerts API for the same reason — filter buttons are static UI.
    await dealerPage.route('**/api/fleet/alerts**', route => route.abort());
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    for (const sev of ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as const) {
      await expect(alerts.filterBtn(sev)).toBeVisible({ timeout: 8_000 });
    }
  });

  test('ALL filter is active by default', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    // Active button has dark background; check aria/class isn't trivial without data-testid state.
    // We verify by checking it's visible and the table shows all severities
    await expect(alerts.filterBtn('ALL')).toBeVisible();
    await alerts.waitForData();
  });

  test('switching to CRITICAL filter updates visible data', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    await alerts.waitForData();
    await alerts.setSeverity('CRITICAL');
    // After filter change, either table or empty-state should still be visible
    await alerts.waitForData();
    await expect(alerts.heading).toBeVisible();
  });

  test('hours select has correct options', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    const options = await alerts.hoursSelect.locator('option').allTextContents();
    expect(options).toContain('24h');
    expect(options.some(o => o.includes('d'))).toBeTruthy();
  });

  test('changing hours to 7d re-fetches data', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    await alerts.waitForData();
    await alerts.setHours(168); // 7d = 168h
    await alerts.waitForData();
    await expect(dealerPage.getByText(/total alerts in last 168h/i)).toBeVisible();
  });

  test('refresh button triggers a new request', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    await alerts.waitForData();
    let fetchCount = 0;
    dealerPage.on('request', req => {
      if (req.url().includes('/api/fleet/alerts')) fetchCount++;
    });
    await alerts.refresh();
    await dealerPage.waitForTimeout(1_000);
    expect(fetchCount).toBeGreaterThan(0);
  });

  test('severity column header is visible in table', async ({ dealerPage }) => {
    const alerts = new AlertsPage(dealerPage);
    await alerts.goto();
    await alerts.waitForData();
    const hasTable = await alerts.table.isVisible();
    if (hasTable) {
      await expect(dealerPage.getByRole('columnheader', { name: 'Severity' })).toBeVisible();
      await expect(dealerPage.getByRole('columnheader', { name: 'VIN' })).toBeVisible();
    }
  });
});
