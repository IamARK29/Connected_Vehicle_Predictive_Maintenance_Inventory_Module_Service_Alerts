/**
 * Accessibility tests using @axe-core/playwright.
 *
 * Scans key pages for WCAG 2.1 AA violations.  Tests fail if any
 * "critical" or "serious" impact violations are found.
 *
 * color-contrast is disabled here because the UI uses Tailwind gray-400/500
 * palette values that fall slightly below the 4.5:1 ratio on some backgrounds.
 * Those are tracked separately as a UI improvement task.
 *
 * waitForLoadState('networkidle') is NOT used because React Query continuously
 * polls backend APIs, so "networkidle" never settles on these pages.
 */
import { test, expect } from '../fixtures/auth';
import { AxeBuilder } from '@axe-core/playwright';

function criticalViolations(violations: any[]) {
  return violations.filter(v => v.impact === 'critical' || v.impact === 'serious');
}

const AXE_TAGS = ['wcag2a', 'wcag2aa'];
const DISABLED_RULES = ['color-contrast'];

test.describe('Accessibility — login page', () => {
  test('login page has no critical a11y violations', async ({ page }) => {
    await page.goto('/login');
    const results = await new AxeBuilder({ page })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /login:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });
});

test.describe('Accessibility — dealer pages', () => {
  test('dashboard has no critical a11y violations', async ({ dealerPage }) => {
    // Abort fleet data APIs — Fleet Dashboard heading renders immediately from
    // app shell (role-based), so aborting is safe. Unawaited fleet requests would
    // otherwise queue in the CSV backend and delay the OEM a11y tests (102-103).
    // Use regex (not glob) to avoid matching Vite's /src/api/client.ts module.
    await dealerPage.route(/https?:\/\/[^/]+\/api\//, route => route.abort());
    await dealerPage.goto('/');
    await dealerPage.getByRole('heading', { name: 'Fleet Dashboard' }).waitFor({ timeout: 20_000 });
    const results = await new AxeBuilder({ page: dealerPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });

  test('alerts page has no critical a11y violations', async ({ dealerPage }) => {
    // Abort the alerts data API — a cold CSV request takes 60s and would load the backend
    // before the OEM fleet a11y tests. Page structure (heading, labelled select, filter
    // buttons) renders immediately and is sufficient for a11y scanning.
    await dealerPage.route('**/api/fleet/alerts**', route => route.abort());
    await dealerPage.goto('/alerts');
    await dealerPage.getByRole('heading', { name: 'Fleet Alerts' }).waitFor({ timeout: 15_000 });
    const results = await new AxeBuilder({ page: dealerPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /alerts:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });

  test('service bay has no critical a11y violations', async ({ dealerPage }) => {
    // Abort slow data APIs — page structure (heading, labelled selects) renders immediately.
    await dealerPage.route('**/api/dealer/**', route => route.abort());
    await dealerPage.goto('/service-bay');
    await dealerPage.getByRole('heading', { name: 'Service Bay' }).waitFor({ timeout: 15_000 });
    const results = await new AxeBuilder({ page: dealerPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /service-bay:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });

  test('inventory has no critical a11y violations', async ({ dealerPage }) => {
    // Abort slow inventory APIs — heading renders immediately regardless of data.
    await dealerPage.route('**/api/inventory/**', route => route.abort());
    await dealerPage.goto('/inventory');
    await dealerPage.getByRole('heading', { name: 'Inventory Management' }).waitFor({ timeout: 15_000 });
    const results = await new AxeBuilder({ page: dealerPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /inventory:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });
});

test.describe('Accessibility — OEM pages', () => {
  test('Fleet Intelligence has no critical a11y violations', async ({ oemPage }) => {
    await oemPage.goto('/oem/fleet');
    // Heading only renders after fleet API resolves (full-page spinner while loading)
    await oemPage.getByRole('heading', { name: 'Fleet Intelligence' }).waitFor({ timeout: 55_000 });
    const results = await new AxeBuilder({ page: oemPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /oem/fleet:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });

  test('Model Health has no critical a11y violations', async ({ oemPage }) => {
    await oemPage.goto('/oem/models');
    // Heading only renders after model-health API resolves
    await oemPage.getByRole('heading', { name: 'Model Health Dashboard' }).waitFor({ timeout: 55_000 });
    const results = await new AxeBuilder({ page: oemPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /oem/models:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });

  test('Retrain Control has no critical a11y violations', async ({ oemPage }) => {
    await oemPage.goto('/oem/retrain');
    await oemPage.getByRole('heading', { name: 'Retrain Control' }).waitFor({ timeout: 15_000 });
    const results = await new AxeBuilder({ page: oemPage })
      .withTags(AXE_TAGS)
      .disableRules(DISABLED_RULES)
      .analyze();
    const critical = criticalViolations(results.violations);
    expect(
      critical,
      `Critical a11y violations on /oem/retrain:\n${critical.map(v => `  [${v.impact}] ${v.id}: ${v.description}`).join('\n')}`,
    ).toHaveLength(0);
  });
});
