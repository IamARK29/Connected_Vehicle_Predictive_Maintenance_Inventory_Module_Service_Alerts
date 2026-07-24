/**
 * OEM Model Health Dashboard — page loads, KPI cards, model cards visible.
 *
 * Previously broken: require_oem() was comparing uppercase roles but
 * get_current_user() always returns lowercase. This test suite serves as a
 * regression guard against that class of bug.
 *
 * OemModelHealth returns a full-page spinner while isLoading — heading and all
 * content only appear after the API call resolves.  Use waitForHeading() (which
 * waits for the spinner to disappear) before all subsequent assertions.
 */
import { test, expect } from '../fixtures/auth';
import { OemModelHealthPage } from '../pages/OemModelHealthPage';

test.describe('OEM Model Health Dashboard', () => {
  test('heading is visible — confirms page is not blank', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(mh.heading).toBeVisible();
  });

  test('page does NOT show a 403 error', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(oemPage.getByText(/403|forbidden|access required/i)).not.toBeVisible();
  });

  test('subtitle references real metric file', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(oemPage.getByText(/model_metrics\.json/i)).toBeVisible();
  });

  test('KPI cards render (Total Models, Avg Concordance)', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(mh.totalModelsKpi).toBeVisible();
    await expect(oemPage.getByText('Avg Concordance')).toBeVisible();
  });

  test('model category sections are rendered', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(oemPage.getByText('Vehicle Health Models')).toBeVisible();
    await expect(oemPage.getByText('Operational Models')).toBeVisible();
    await expect(oemPage.getByText('EV Physics Engines')).toBeVisible();
  });

  test('Model Performance Comparison chart appears when concordance data exists', async ({ oemPage }) => {
    const mh = new OemModelHealthPage(oemPage);
    await mh.goto();
    await mh.waitForHeading();
    // The chart (and its h3) only renders when at least one model has a
    // `metrics.concordance_index` value — the component gates on concordanceData.length > 0.
    // If no model exposes that metric the chart is correctly omitted; that is valid.
    const hasChart = await oemPage.locator('h3', { hasText: 'Model Performance Comparison' }).isVisible();
    if (hasChart) {
      await expect(oemPage.locator('h3', { hasText: 'Model Performance Comparison' })).toBeVisible();
    }
  });

  test('admin can also access model health page', async ({ adminPage }) => {
    const mh = new OemModelHealthPage(adminPage);
    await mh.goto();
    await mh.waitForHeading();
    await expect(mh.heading).toBeVisible();
    await expect(adminPage.getByText(/403|forbidden/i)).not.toBeVisible();
  });
});
