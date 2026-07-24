import { Page, Locator } from '@playwright/test';

export class ServiceBayPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly bookBtn: Locator;
  readonly appointmentsHeading: Locator;
  readonly predictedServiceHeading: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading                  = page.getByRole('heading', { name: 'Service Bay' });
    this.bookBtn                  = page.getByRole('button', { name: '+ Book Appointment' });
    this.appointmentsHeading      = page.getByRole('heading', { name: 'Appointments' });
    this.predictedServiceHeading  = page.getByRole('heading', { name: 'Predicted Service Needs' });
  }

  async goto() {
    await this.page.goto('/service-bay');
  }

  async openBookingModal() {
    await this.bookBtn.click();
  }

  /** Locator for the VIN input inside the booking modal */
  get modalVinInput(): Locator {
    return this.page.getByRole('textbox', { name: /VIN/i });
  }

  get modalCancelBtn(): Locator {
    return this.page.getByRole('button', { name: 'Cancel' });
  }

  get modalBookBtn(): Locator {
    return this.page.getByRole('button', { name: /^Book$|^Book$/i });
  }
}
