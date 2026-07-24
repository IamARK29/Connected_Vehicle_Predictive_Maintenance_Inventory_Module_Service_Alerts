import { Page, Locator } from '@playwright/test';

export class OemRetrainPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly selectAllBtn: Locator;
  readonly clearBtn: Locator;
  readonly retrainSubmit: Locator;
  readonly notesTextarea: Locator;
  readonly historySection: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading        = page.getByRole('heading', { name: 'Retrain Control' });
    this.selectAllBtn   = page.getByTestId('retrain-select-all');
    this.clearBtn       = page.getByTestId('retrain-clear');
    this.retrainSubmit  = page.getByTestId('retrain-submit');
    this.notesTextarea  = page.getByTestId('retrain-notes');
    this.historySection = page.getByRole('heading', { name: 'Training History' });
  }

  async goto() {
    await this.page.goto('/oem/retrain');
  }

  async waitForHeading() {
    await this.heading.waitFor({ state: 'visible', timeout: 12_000 });
  }

  async selectAll() {
    await this.selectAllBtn.click();
  }

  async clear() {
    await this.clearBtn.click();
  }

  async fillNotes(text: string) {
    await this.notesTextarea.fill(text);
  }

  /** Button text contains the count of models selected. */
  async selectedModelCount(): Promise<number> {
    const text = await this.retrainSubmit.textContent();
    const match = text?.match(/(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  }
}
