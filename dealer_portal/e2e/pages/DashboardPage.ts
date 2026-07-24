import { Page, Locator } from '@playwright/test';

export class DashboardPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly searchInput: Locator;
  readonly searchBtn: Locator;
  readonly vehicleTable: Locator;
  readonly signOutBtn: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading     = page.getByRole('heading', { name: 'Fleet Dashboard' });
    this.searchInput = page.getByPlaceholder('Search by VIN, model, or driver profile...');
    this.searchBtn   = page.getByRole('button', { name: 'Go' });
    this.vehicleTable = page.getByRole('table').first();
    this.signOutBtn  = page.getByTestId('sign-out-btn');
  }

  async goto() {
    await this.page.goto('/');
  }

  async search(query: string) {
    await this.searchInput.fill(query);
    await this.searchBtn.click();
  }

  async waitForTable() {
    await this.vehicleTable.waitFor({ state: 'visible' });
  }
}
