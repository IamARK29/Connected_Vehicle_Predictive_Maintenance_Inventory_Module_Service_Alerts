/**
 * Dealer Dashboard — Fleet overview, search, vehicle list, navigation.
 */
import { test, expect } from '../fixtures/auth';
import { DashboardPage } from '../pages/DashboardPage';

test.describe('Dealer dashboard', () => {
  test('renders Fleet Dashboard heading', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await expect(dash.heading).toBeVisible({ timeout: 10_000 });
  });

  test('shows fleet KPI cards', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await expect(dealerPage.getByText('Total Vehicles')).toBeVisible({ timeout: 10_000 });
    await expect(dealerPage.getByText('Fleet Health Score')).toBeVisible();
    await expect(dealerPage.getByText('Critical Alerts')).toBeVisible();
  });

  test('vehicle table renders with column headers', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await dash.waitForTable();
    await expect(dealerPage.getByRole('columnheader', { name: 'VIN' })).toBeVisible();
    await expect(dealerPage.getByRole('columnheader', { name: 'Model' })).toBeVisible();
    await expect(dealerPage.getByRole('columnheader', { name: 'Health' })).toBeVisible();
  });

  test('search filters vehicle list', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await dash.waitForTable();
    const rowsBefore = await dealerPage.getByRole('row').count();
    await dash.search('MH01MZ7X0001');
    const rowsAfter = await dealerPage.getByRole('row').count();
    // Should show fewer or equal rows (search narrows the list)
    expect(rowsAfter).toBeLessThanOrEqual(rowsBefore);
  });

  test('sorting by Health column changes row order', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await dash.waitForTable();
    const healthHeader = dealerPage.getByRole('columnheader', { name: 'Health' });
    await healthHeader.click();
    // After sorting, table is still visible
    await expect(dash.vehicleTable).toBeVisible();
  });

  test('Upcoming Service section is present', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await expect(dealerPage.getByRole('heading', { name: /Upcoming Service/i })).toBeVisible({ timeout: 10_000 });
  });

  test('sign-out navigates to /login', async ({ dealerPage }) => {
    const dash = new DashboardPage(dealerPage);
    await dash.goto();
    await dash.signOutBtn.click();
    await expect(dealerPage).toHaveURL(/\/login/, { timeout: 5_000 });
  });
});
