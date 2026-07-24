import { Page, Locator } from '@playwright/test';

export type Severity = 'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export class AlertsPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly refreshBtn: Locator;
  readonly hoursSelect: Locator;
  readonly table: Locator;
  readonly emptyState: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading     = page.getByRole('heading', { name: 'Fleet Alerts' });
    this.refreshBtn  = page.getByTestId('alerts-refresh');
    this.hoursSelect = page.getByTestId('alerts-hours');
    this.table       = page.getByRole('table');
    this.emptyState  = page.getByText(/No .* alerts in this period/);
  }

  async goto() {
    await this.page.goto('/alerts');
  }

  filterBtn(severity: Severity): Locator {
    return this.page.getByTestId(`filter-${severity.toLowerCase()}`);
  }

  async setSeverity(severity: Severity) {
    await this.filterBtn(severity).click();
  }

  async setHours(hours: number) {
    await this.hoursSelect.selectOption(String(hours));
  }

  async refresh() {
    await this.refreshBtn.click();
  }

  /**
   * Waits until the alerts data has loaded — either a table of results or
   * the empty-state message.  The Alerts component shows "Loading alerts…"
   * while the API is in-flight; once resolved, exactly one of table/empty-state
   * becomes visible.
   *
   * With a single Playwright worker the FastAPI backend (single-threaded,
   * CSV-based) returns the alerts data in ~10-20s.
   */
  async waitForData() {
    await Promise.any([
      this.table.waitFor({ state: 'visible', timeout: 80_000 }),
      this.emptyState.waitFor({ state: 'visible', timeout: 80_000 }),
    ]);
  }
}
