import { Page, Locator } from '@playwright/test';

export type InventoryTab =
  | 'overview'
  | 'stock-ledger'
  | 'reorder-plan'
  | 'analytics'
  | 'multi-dealer'
  | 'transactions'
  | 'demand-forecast';

export class InventoryPage {
  readonly page: Page;
  readonly heading: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading = page.getByRole('heading', { name: 'Inventory Management' });
  }

  async goto() {
    await this.page.goto('/inventory');
  }

  tab(name: InventoryTab): Locator {
    return this.page.getByTestId(`tab-${name}`);
  }

  async clickTab(name: InventoryTab) {
    await this.tab(name).click();
  }

  async waitForHeading() {
    await this.heading.waitFor({ state: 'visible' });
  }

  async visibleTabs(): Promise<string[]> {
    const tabs = await this.page.locator('[data-testid^="tab-"]').allTextContents();
    return tabs.map(t => t.trim());
  }
}
