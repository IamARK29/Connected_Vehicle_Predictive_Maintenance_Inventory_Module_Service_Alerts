import { test as setup, expect } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const API_BASE = process.env.API_BASE ?? 'http://localhost:8001';
const AUTH_DIR = path.join(__dirname, '.auth');

const ROLES = [
  { username: 'dealer',  password: 'dealer123', file: 'dealer.json'  },
  { username: 'dealer2', password: 'dealer123', file: 'dealer2.json' },
  { username: 'oem',     password: 'oem123',    file: 'oem.json'     },
  { username: 'admin',   password: 'admin123',  file: 'admin.json'   },
];

fs.mkdirSync(AUTH_DIR, { recursive: true });

for (const { username, password, file } of ROLES) {
  setup(`authenticate as ${username}`, async ({ request, page }) => {
    const resp = await request.post(`${API_BASE}/api/auth/token`, {
      data: { username, password },
    });
    expect(resp.ok(), `Auth failed for ${username}: ${resp.status()}`).toBeTruthy();
    const { access_token, role, dealer_code } = await resp.json();

    await page.goto('http://localhost:3000/login');
    await page.evaluate(
      ({ token, r, dc, u }: { token: string; r: string; dc: string; u: string }) => {
        localStorage.setItem('ap_token',       token);
        localStorage.setItem('ap_role',        r);
        localStorage.setItem('ap_dealer_code', dc ?? 'ALL');
        localStorage.setItem('ap_user',        u);
      },
      { token: access_token, r: role, dc: dealer_code, u: username },
    );

    await page.context().storageState({ path: path.join(AUTH_DIR, file) });
  });
}
