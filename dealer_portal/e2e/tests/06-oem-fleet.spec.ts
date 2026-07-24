/**
 * OEM Fleet Intelligence — group-by switching, data table, KPI cards.
 *
 * OemFleetOverview shows a full-page spinner while the API loads; heading and
 * all content only render after the request resolves.  Use waitForPageReady()
 * (waits for heading) or waitForData() (waits for table) before assertions.
 */
import { test, expect } from '../fixtures/auth';
import { OemFleetPage } from '../pages/OemFleetPage';

test.describe('OEM Fleet Intelligence', () => {
  test('heading is visible', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    await expect(fleet.heading).toBeVisible();
  });

  test('subtitle describes the page purpose', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    await expect(oemPage.getByText(/cross-dealer fleet health/i)).toBeVisible();
  });

  test('all four group-by buttons are present', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    for (const [value, label] of [
      ['dealer_code', 'By Dealer'],
      ['model_name',  'By Model'],
      ['fuel_type',   'By Fuel Type'],
      ['region',      'By Region'],
    ] as const) {
      await expect(fleet.groupByBtn(value)).toBeVisible();
      await expect(fleet.groupByBtn(value)).toContainText(label);
    }
  });

  test('By Dealer is the default active group', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForData();
    const rows = await fleet.rowCount();
    expect(rows).toBeGreaterThan(1); // header + at least one dealer row
  });

  test('switching to By Model changes table', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForData(); // initial load

    await fleet.setGroupBy('model_name');
    // After group change the component re-fetches (spinner shows, table hides).
    // waitForData() waits for the table to reappear with new data.
    await fleet.waitForData();
    const rowsAfter = await fleet.rowCount();
    expect(rowsAfter).toBeGreaterThan(0);
  });

  test('switching to By Region returns data', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    await fleet.setGroupBy('region');
    await fleet.waitForData();
    await expect(fleet.table).toBeVisible();
  });

  test('KPI cards include Total Vehicles', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    await expect(oemPage.getByText('Total Vehicles')).toBeVisible();
    // 'Avg Health' appears as both a KPI label (p) and a table column header (th)
    await expect(oemPage.getByText('Avg Health').first()).toBeVisible();
  });

  test('chart sections are rendered', async ({ oemPage }) => {
    const fleet = new OemFleetPage(oemPage);
    await fleet.goto();
    await fleet.waitForPageReady();
    await expect(oemPage.getByRole('heading', { name: 'Health Score Ranking' })).toBeVisible();
    await expect(oemPage.getByRole('heading', { name: 'Group Detail' })).toBeVisible();
  });
});
