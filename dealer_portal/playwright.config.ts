import { defineConfig, devices } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE_URL = process.env.BASE_URL ?? 'http://localhost:3000';

export const AUTH_DIR = path.join(__dirname, 'e2e', '.auth');

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 12_000 },

  retries: process.env.CI ? 2 : 0,
  // Single worker prevents parallel backend load — the Python FastAPI backend
  // processes large CSV files synchronously, so concurrent test requests cause
  // 25-45s timeouts.  Serial execution keeps each API call under 25s.
  workers: 1,

  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
  ],

  use: {
    baseURL: BASE_URL,
    headless: true,
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
    // Don't set actionTimeout — it propagates to APIRequestContext and causes
    // API contract tests to time out. Let each test control its own timing.
    navigationTimeout: 25_000,
  },

  projects: [
    {
      name: 'setup',
      testDir: './e2e',
      testMatch: /global-setup\.ts/,
      teardown: 'teardown',
    },
    {
      name: 'teardown',
      testDir: './e2e',
      testMatch: /global-teardown\.ts/,
    },
    {
      name: 'chromium',
      testDir: './e2e/tests',
      use: { ...devices['Desktop Chrome'] },
      dependencies: ['setup'],
    },
  ],
});

export { BASE_URL };
