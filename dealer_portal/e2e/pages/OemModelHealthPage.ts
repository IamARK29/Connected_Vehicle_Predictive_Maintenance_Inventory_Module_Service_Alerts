import { Page, Locator } from '@playwright/test';

export class OemModelHealthPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly totalModelsKpi: Locator;

  constructor(page: Page) {
    this.page = page;
    // heading renders only when isLoading is false (same full-page spinner pattern)
    this.heading        = page.getByRole('heading', { name: 'Model Health Dashboard' });
    this.totalModelsKpi = page.getByText('Total Models').first();
  }

  async goto() {
    await this.page.goto('/oem/models');
  }

  /**
   * Waits until the "Model Health Dashboard" heading is visible.
   * OemModelHealth returns a full-page spinner while isLoading=true, so the
   * heading only renders once the API call resolves.
   */
  async waitForHeading() {
    await this.heading.waitFor({ state: 'visible', timeout: 40_000 });
  }

  modelCard(name: string): Locator {
    return this.page.getByText(name, { exact: false }).first();
  }
}
