import { Page, Locator } from '@playwright/test';

export class LoginPage {
  readonly page: Page;
  readonly username: Locator;
  readonly password: Locator;
  readonly submitBtn: Locator;
  readonly errorMsg: Locator;

  constructor(page: Page) {
    this.page = page;
    this.username  = page.getByTestId('login-username');
    this.password  = page.getByTestId('login-password');
    this.submitBtn = page.getByTestId('login-submit');
    // Match the text set by setError('Invalid username or password')
    this.errorMsg  = page.getByText(/invalid username or password/i);
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(username: string, password: string) {
    await this.goto();
    await this.username.fill(username);
    await this.password.fill(password);
    await this.submitBtn.click();
  }

  async loginAndWait(username: string, password: string) {
    await this.login(username, password);
    await this.page.waitForURL(url => !url.pathname.includes('/login'), {
      timeout: 45_000,
    });
  }
}
