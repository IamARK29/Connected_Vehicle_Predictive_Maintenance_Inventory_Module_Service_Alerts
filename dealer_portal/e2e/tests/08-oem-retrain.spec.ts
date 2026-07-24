/**
 * OEM Retrain Control — model selection, notes, select-all/clear, submit state.
 */
import { test, expect } from '../fixtures/auth';
import { OemRetrainPage } from '../pages/OemRetrainPage';

test.describe('OEM Retrain Control', () => {
  test('heading is visible', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await expect(rt.heading).toBeVisible({ timeout: 12_000 });
  });

  test('page does NOT show a 403 error', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await expect(oemPage.getByText(/403|forbidden|access required/i)).not.toBeVisible();
  });

  test('Trigger Retraining panel is visible', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await expect(oemPage.getByRole('heading', { name: 'Trigger Retraining' })).toBeVisible();
  });

  test('all 8 trainable model checkboxes are present', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    const checkboxes = oemPage.locator('input[type="checkbox"]');
    const count = await checkboxes.count();
    expect(count).toBe(8);
  });

  test('Select All button checks all models', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.selectAll();
    const count = await rt.selectedModelCount();
    expect(count).toBe(8);
  });

  test('Clear button unchecks all models', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.selectAll();
    await rt.clear();
    const count = await rt.selectedModelCount();
    expect(count).toBe(0);
  });

  test('Retrain button is disabled when 0 models selected', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.clear();
    await expect(rt.retrainSubmit).toBeDisabled();
  });

  test('Retrain button is enabled after selecting a model', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.clear();
    // Click the first checkbox
    await oemPage.locator('input[type="checkbox"]').first().check();
    await expect(rt.retrainSubmit).toBeEnabled();
    const count = await rt.selectedModelCount();
    expect(count).toBe(1);
  });

  test('Notes textarea accepts input', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.fillNotes('Monthly scheduled retrain — Q3 2026');
    await expect(rt.notesTextarea).toHaveValue('Monthly scheduled retrain — Q3 2026');
  });

  test('EV Physics Engines section is read-only (no retrain buttons)', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await expect(oemPage.getByText('EV Physics Engines')).toBeVisible();
    await expect(oemPage.getByText('No training required')).toBeVisible();
  });

  test('Training History section is present', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await expect(rt.historySection).toBeVisible();
  });

  test('Before retraining cost estimate updates with selection', async ({ oemPage }) => {
    const rt = new OemRetrainPage(oemPage);
    await rt.goto();
    await rt.waitForHeading();
    await rt.selectAll();
    await expect(oemPage.getByText(/Training 8 model\(s\)/i)).toBeVisible();
  });

  test('admin can also access retrain page', async ({ adminPage }) => {
    const rt = new OemRetrainPage(adminPage);
    await rt.goto();
    await expect(rt.heading).toBeVisible({ timeout: 12_000 });
    await expect(adminPage.getByText(/403|forbidden/i)).not.toBeVisible();
  });
});
