/**
 * Role-based routing guards — React-level (OemRoute / AdminRoute) and
 * nav section visibility.
 *
 * Tests that navigate to data-heavy pages intercept API calls to prevent
 * slow CSV backend requests from queuing up before the OEM a11y tests.
 * Nav labels, role badges, and tab visibility are driven by the JWT role
 * (localStorage, available immediately), so aborting API calls is safe.
 *
 * IMPORTANT: use the regex /https?:\/\/[^/]+\/api\// rather than the glob
 * '*\/api\/**'.  The glob accidentally matches Vite's source file at
 * /src/api/client.ts, which prevents React from mounting — routing guards
 * never fire and redirect assertions fail.  The regex only matches URLs where
 * /api/ is the FIRST path segment (actual backend calls), not a nested dir.
 */
import { test, expect } from '../fixtures/auth';

// Matches /api/ as the first path segment only (not /src/api/client.ts).
const DATA_API = /https?:\/\/[^/]+\/api\//;

test.describe('OEM route guard (dealer blocked)', () => {
  const OEM_ROUTES = ['/oem/fleet', '/oem/models', '/oem/eda', '/oem/whatif', '/oem/retrain'];

  for (const route of OEM_ROUTES) {
    test(`dealer navigating to ${route} is redirected`, async ({ dealerPage }) => {
      // Abort data APIs — after redirect, Dashboard loads and would otherwise
      // fire slow fleet CSV requests that queue the backend.
      await dealerPage.route(DATA_API, r => r.abort());
      await dealerPage.goto(route);
      await dealerPage.waitForTimeout(1_000);
      await expect(dealerPage).not.toHaveURL(new RegExp(route.replace('/', '\\/')));
    });
  }
});

test.describe('OEM nav section visibility', () => {
  test('dealer does NOT see OEM Intelligence section', async ({ dealerPage }) => {
    await dealerPage.route(DATA_API, r => r.abort());
    await dealerPage.goto('/');
    await dealerPage.waitForTimeout(2_000);
    await expect(dealerPage.getByText('OEM Intelligence')).not.toBeVisible();
  });

  test('OEM user sees OEM Intelligence section', async ({ oemPage }) => {
    await oemPage.route(DATA_API, r => r.abort());
    await oemPage.goto('/');
    await expect(oemPage.getByText('OEM Intelligence')).toBeVisible({ timeout: 8_000 });
  });

  test('admin sees OEM Intelligence section', async ({ adminPage }) => {
    await adminPage.route(DATA_API, r => r.abort());
    await adminPage.goto('/');
    await expect(adminPage.getByText('OEM Intelligence')).toBeVisible({ timeout: 8_000 });
  });

  test('dealer does NOT see Admin nav section', async ({ dealerPage }) => {
    await dealerPage.route(DATA_API, r => r.abort());
    await dealerPage.goto('/');
    await dealerPage.waitForTimeout(1_500);
    await expect(dealerPage.getByText('Admin').first()).not.toBeVisible();
  });

  test('admin sees Admin nav section', async ({ adminPage }) => {
    await adminPage.route(DATA_API, r => r.abort());
    await adminPage.goto('/');
    await expect(adminPage.getByText('Admin').first()).toBeVisible({ timeout: 8_000 });
  });
});

test.describe('Inventory Multi-Dealer tab gating', () => {
  test('dealer does not see Multi-Dealer tab', async ({ dealerPage }) => {
    await dealerPage.route(DATA_API, r => r.abort());
    await dealerPage.goto('/inventory');
    await dealerPage.waitForTimeout(2_000);
    await expect(dealerPage.getByTestId('tab-multi-dealer')).not.toBeVisible();
  });

  test('OEM sees Multi-Dealer tab', async ({ oemPage }) => {
    await oemPage.route(DATA_API, r => r.abort());
    await oemPage.goto('/inventory');
    await oemPage.waitForTimeout(2_000);
    await expect(oemPage.getByTestId('tab-multi-dealer')).toBeVisible();
  });

  test('admin sees Multi-Dealer tab', async ({ adminPage }) => {
    await adminPage.route(DATA_API, r => r.abort());
    await adminPage.goto('/inventory');
    await adminPage.waitForTimeout(2_000);
    await expect(adminPage.getByTestId('tab-multi-dealer')).toBeVisible();
  });
});

test.describe('Demand forecast regional breakdown gating', () => {
  test('dealer sees no regional/dealer breakdown pivot', async ({ dealerPage }) => {
    await dealerPage.route(DATA_API, r => r.abort());
    await dealerPage.goto('/inventory');
    await dealerPage.waitForTimeout(1_000);
    await dealerPage.getByTestId('tab-demand-forecast').click();
    await dealerPage.waitForTimeout(2_000);
    // Breakdown pivot only shows for OEM role
    await expect(dealerPage.getByText(/breakdown by region|dealer breakdown/i)).not.toBeVisible();
  });
});

test.describe('Portal label in sidebar', () => {
  test('dealer sees Dealer · DL001 label', async ({ dealerPage }) => {
    await dealerPage.route(DATA_API, r => r.abort());
    await dealerPage.goto('/');
    await expect(dealerPage.getByText(/Dealer · DL001/i)).toBeVisible({ timeout: 8_000 });
  });

  test('OEM sees OEM Portal label', async ({ oemPage }) => {
    await oemPage.route(DATA_API, r => r.abort());
    await oemPage.goto('/');
    await expect(oemPage.getByText('OEM Portal')).toBeVisible({ timeout: 8_000 });
  });

  test('admin sees Admin Portal label', async ({ adminPage }) => {
    await adminPage.route(DATA_API, r => r.abort());
    await adminPage.goto('/');
    await expect(adminPage.getByText('Admin Portal')).toBeVisible({ timeout: 8_000 });
  });
});
