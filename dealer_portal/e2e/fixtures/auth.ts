import { test as base, expect, Page, BrowserContext } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const AUTH_DIR = path.join(__dirname, '..', '.auth');

type AuthFixtures = {
  dealerPage:   Page;
  dealer2Page:  Page;
  oemPage:      Page;
  adminPage:    Page;
  dealerContext: BrowserContext;
  oemContext:    BrowserContext;
};

export const test = base.extend<AuthFixtures>({
  dealerPage: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'dealer.json') });
    const page = await ctx.newPage();
    await use(page);
    await ctx.close();
  },

  dealer2Page: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'dealer2.json') });
    const page = await ctx.newPage();
    await use(page);
    await ctx.close();
  },

  oemPage: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'oem.json') });
    const page = await ctx.newPage();
    await use(page);
    await ctx.close();
  },

  adminPage: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'admin.json') });
    const page = await ctx.newPage();
    await use(page);
    await ctx.close();
  },

  dealerContext: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'dealer.json') });
    await use(ctx);
    await ctx.close();
  },

  oemContext: async ({ browser }, use) => {
    const ctx = await browser.newContext({ storageState: path.join(AUTH_DIR, 'oem.json') });
    await use(ctx);
    await ctx.close();
  },
});

export { expect };
