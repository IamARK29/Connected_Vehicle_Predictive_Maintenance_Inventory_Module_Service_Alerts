/**
 * Authentication tests — login form UX, redirects, token persistence, logout.
 * These tests intentionally use raw `page` (no pre-auth) so they exercise the
 * login form itself.
 */
import { test, expect } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';

test.describe('Login page', () => {
  test('renders all form elements', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.goto();
    await expect(lp.username).toBeVisible();
    await expect(lp.password).toBeVisible();
    await expect(lp.submitBtn).toBeVisible();
    await expect(page.getByText('Sign In →')).toBeVisible();
  });

  test('shows AutoPredict branding', async ({ page }) => {
    await page.goto('/login');
    // Multiple elements contain "AutoPredict" — assert the first one (h1 in sidebar)
    await expect(page.getByText('AutoPredict').first()).toBeVisible();
    await expect(page.getByText('Predict before it breaks.')).toBeVisible();
  });

  test('displays demo credentials', async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByText('dealer / dealer123')).toBeVisible();
    await expect(page.getByText('oem / oem123')).toBeVisible();
    await expect(page.getByText('admin / admin123')).toBeVisible();
  });

  test('wrong password shows error', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.login('dealer', 'wrongpassword');
    // Backend processes the 401 in <1s normally; allow up to 15s under load
    await expect(lp.errorMsg).toBeVisible({ timeout: 15_000 });
    await expect(lp.errorMsg).toContainText(/invalid/i);
    // Should stay on /login
    await expect(page).toHaveURL(/\/login/);
  });

  test('unknown user shows error', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.login('nobody', 'whatever');
    await expect(lp.errorMsg).toBeVisible({ timeout: 15_000 });
  });

  test('submit button shows Signing in… while loading', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.goto();
    await lp.username.fill('dealer');
    await lp.password.fill('dealer123');
    // Slow down the API to catch the loading state
    await page.route('**/api/auth/token', async route => {
      await new Promise(r => setTimeout(r, 800));
      await route.continue();
    });
    await lp.submitBtn.click();
    await expect(page.getByText('Signing in…')).toBeVisible();
  });
});

test.describe('Unauthenticated redirects', () => {
  test('/ redirects to /login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login/);
  });

  test('/inventory redirects to /login', async ({ page }) => {
    await page.goto('/inventory');
    await expect(page).toHaveURL(/\/login/);
  });

  test('/oem/fleet redirects to /login', async ({ page }) => {
    await page.goto('/oem/fleet');
    await expect(page).toHaveURL(/\/login/);
  });

  test('/alerts redirects to /login', async ({ page }) => {
    await page.goto('/alerts');
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('Successful login flows', () => {
  // Form-based login round-trips the API and waits for React to redirect.
  // Mark these slow so Playwright triples the timeout (135s total).
  test.slow();

  test('dealer login lands on dashboard', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.loginAndWait('dealer', 'dealer123');
    await expect(page.getByRole('heading', { name: 'Fleet Dashboard' })).toBeVisible({ timeout: 15_000 });
  });

  test('OEM login lands on dashboard', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.loginAndWait('oem', 'oem123');
    await expect(page).not.toHaveURL(/\/login/);
  });

  test('admin login lands on dashboard', async ({ page }) => {
    const lp = new LoginPage(page);
    await lp.loginAndWait('admin', 'admin123');
    await expect(page).not.toHaveURL(/\/login/);
  });
});
