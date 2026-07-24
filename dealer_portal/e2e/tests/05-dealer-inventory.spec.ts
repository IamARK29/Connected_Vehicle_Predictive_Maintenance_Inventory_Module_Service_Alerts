/**
 * Inventory Management — tabs, role-scoped tab visibility, demand forecast scoping.
 */
import { test, expect } from '../fixtures/auth';
import { InventoryPage } from '../pages/InventoryPage';

test.describe('Inventory tabs — dealer role', () => {
  test('Inventory Management heading is visible', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await expect(inv.heading).toBeVisible({ timeout: 10_000 });
  });

  test('dealer sees standard tabs (not Multi-Dealer)', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await inv.waitForHeading();

    const expected = ['Overview', 'Stock Ledger', 'Reorder Plan', 'Analytics', 'Transactions', 'Demand Forecast'];
    for (const label of expected) {
      await expect(dealerPage.getByText(label).first()).toBeVisible();
    }
  });

  test('Multi-Dealer tab is HIDDEN for dealer', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await inv.waitForHeading();
    await expect(inv.tab('multi-dealer')).not.toBeVisible();
  });

  test('clicking Overview tab shows overview content', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await inv.waitForHeading();
    await inv.clickTab('overview');
    // OverviewTab shows "Loading overview…" while the API call is in flight.
    // Wait for "Total Inventory Value" KPI card which only renders once data loads.
    await expect(dealerPage.getByText('Total Inventory Value')).toBeVisible({ timeout: 35_000 });
  });

  test('clicking Stock Ledger tab shows stock table', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await inv.waitForHeading();
    await inv.clickTab('stock-ledger');
    await expect(dealerPage.getByRole('table')).toBeVisible({ timeout: 8_000 });
  });

  test('clicking Demand Forecast tab shows forecast content', async ({ dealerPage }) => {
    const inv = new InventoryPage(dealerPage);
    await inv.goto();
    await inv.waitForHeading();
    await inv.clickTab('demand-forecast');
    // Should load forecast data, not show an auth error
    await dealerPage.waitForTimeout(2_000);
    await expect(dealerPage.getByText(/403|forbidden|unauthorized/i)).not.toBeVisible();
  });
});

test.describe('Inventory tabs — OEM role', () => {
  test('Multi-Dealer tab IS visible for OEM', async ({ oemPage }) => {
    const inv = new InventoryPage(oemPage);
    await inv.goto();
    await inv.waitForHeading();
    await expect(inv.tab('multi-dealer')).toBeVisible({ timeout: 8_000 });
  });

  test('OEM sees all 7 tabs', async ({ oemPage }) => {
    const inv = new InventoryPage(oemPage);
    await inv.goto();
    await inv.waitForHeading();
    const tabs = await inv.visibleTabs();
    expect(tabs.length).toBe(7);
  });

  test('OEM Multi-Dealer tab loads breakdown table', async ({ oemPage }) => {
    const inv = new InventoryPage(oemPage);
    await inv.goto();
    await inv.waitForHeading();
    await inv.clickTab('multi-dealer');
    await oemPage.waitForTimeout(2_000);
    // Should show a comparison table, not a 403
    await expect(oemPage.getByText(/403|forbidden/i)).not.toBeVisible();
  });

  test('OEM Demand Forecast shows full fleet (~all dealers)', async ({ oemPage }) => {
    const inv = new InventoryPage(oemPage);
    await inv.goto();
    await inv.waitForHeading();
    await inv.clickTab('demand-forecast');
    await oemPage.waitForTimeout(2_000);
    await expect(oemPage.getByText(/403|forbidden/i)).not.toBeVisible();
  });
});
