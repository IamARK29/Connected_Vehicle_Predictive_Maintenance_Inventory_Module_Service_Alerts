import { Page, Locator } from '@playwright/test';

export type GroupBy = 'dealer_code' | 'model_name' | 'fuel_type' | 'region';

export class OemFleetPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly table: Locator;

  constructor(page: Page) {
    this.page = page;
    // heading renders only when isLoading is false (spinner removed, data present)
    this.heading = page.getByRole('heading', { name: 'Fleet Intelligence' });
    this.table   = page.getByRole('table');
  }

  async goto() {
    await this.page.goto('/oem/fleet');
  }

  /**
   * Waits until the "Fleet Intelligence" heading is visible.
   * OemFleetOverview swaps the whole page for a spinner while isLoading=true,
   * so the heading only appears once the API call resolves.
   */
  async waitForPageReady() {
    await this.heading.waitFor({ state: 'visible', timeout: 40_000 });
  }

  /**
   * Waits until the data table is visible.
   * Safe to call after a group-by click — the table disappears while
   * re-fetching and reappears once the new data arrives.
   */
  async waitForData() {
    await this.table.waitFor({ state: 'visible', timeout: 40_000 });
  }

  groupByBtn(value: GroupBy): Locator {
    return this.page.getByTestId(`group-by-${value}`);
  }

  async setGroupBy(value: GroupBy) {
    await this.groupByBtn(value).click();
  }

  async rowCount(): Promise<number> {
    return this.page.getByRole('row').count();
  }
}
