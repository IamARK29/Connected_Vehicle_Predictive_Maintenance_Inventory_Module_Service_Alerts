/**
 * API contract tests — verify that what the backend returns matches what
 * the UI expects.  These tests call the API directly (via Playwright's
 * request context) and assert on response shape, not rendered UI.
 *
 * They also verify that role-based access is enforced at the API layer,
 * not just the UI layer.
 */
import { test, expect } from '../fixtures/auth';

const API = process.env.API_BASE ?? 'http://localhost:8001';

// ── Auth endpoint ─────────────────────────────────────────────────────────────

test.describe('Auth API', () => {
  test('dealer token response contains access_token and dealer_code DL001', async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'dealer', password: 'dealer123' },
    });
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body.access_token).toBeTruthy();
    // Role in the HTTP response comes directly from the user store (uppercase);
    // get_current_user() normalises to lowercase when decoding the JWT.
    expect(['dealer', 'DEALER']).toContain(body.role.toLowerCase() === body.role ? body.role : body.role.toLowerCase());
    expect(body.dealer_code).toBe('DL001');
  });

  test('OEM token response contains access_token and dealer_code ALL', async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'oem', password: 'oem123' },
    });
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body.access_token).toBeTruthy();
    expect(body.dealer_code).toBe('ALL');
  });

  test('wrong password returns 401', async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'dealer', password: 'wrong' },
    });
    expect(resp.status()).toBe(401);
  });

  test('unknown user returns 401', async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'ghost', password: 'ghost' },
    });
    expect(resp.status()).toBe(401);
  });
});

// ── OEM endpoint contract ─────────────────────────────────────────────────────

test.describe('OEM fleet-overview API', () => {
  let oemToken: string;

  test.beforeAll(async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'oem', password: 'oem123' },
    });
    oemToken = (await resp.json()).access_token;
  });

  test('returns 200 with groups array for OEM', async ({ request }) => {
    const resp = await request.get(`${API}/api/oem/fleet-overview`, {
      headers: { Authorization: `Bearer ${oemToken}` },
    });
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(Array.isArray(body.groups)).toBeTruthy();
    expect(body.totals).toBeDefined();
  });

  test('returns 403 for dealer role', async ({ request }) => {
    const tokenResp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'dealer', password: 'dealer123' },
    });
    const dealerToken = (await tokenResp.json()).access_token;

    const resp = await request.get(`${API}/api/oem/fleet-overview`, {
      headers: { Authorization: `Bearer ${dealerToken}` },
    });
    expect(resp.status()).toBe(403);
  });

  test('each group in fleet-overview has required fields', async ({ request }) => {
    const resp = await request.get(`${API}/api/oem/fleet-overview`, {
      headers: { Authorization: `Bearer ${oemToken}` },
    });
    const body = await resp.json();
    for (const group of body.groups.slice(0, 3)) {
      expect(group).toHaveProperty('key');
      expect(group).toHaveProperty('vehicle_count');
      expect(group).toHaveProperty('avg_health_score');
    }
  });
});

// ── Dealer demand forecast contract ──────────────────────────────────────────

test.describe('Dealer demand-forecast API', () => {
  let dealerToken: string;
  let oemToken: string;

  test.beforeAll(async ({ request }) => {
    const [dr, or] = await Promise.all([
      request.post(`${API}/api/auth/token`, { data: { username: 'dealer', password: 'dealer123' } }),
      request.post(`${API}/api/auth/token`, { data: { username: 'oem',    password: 'oem123'    } }),
    ]);
    dealerToken = (await dr.json()).access_token;
    oemToken    = (await or.json()).access_token;
  });

  test('dealer can access own dealer demand forecast', async ({ request }) => {
    const resp = await request.get(`${API}/api/dealer/DL001/demand-forecast`, {
      headers: { Authorization: `Bearer ${dealerToken}` },
    });
    expect(resp.status()).not.toBe(401);
    expect(resp.status()).not.toBe(403);
  });

  test('dealer gets 403 accessing another dealer demand forecast', async ({ request }) => {
    const resp = await request.get(`${API}/api/dealer/DL002/demand-forecast`, {
      headers: { Authorization: `Bearer ${dealerToken}` },
    });
    expect(resp.status()).toBe(403);
  });

  test('OEM can access any dealer demand forecast', async ({ request }) => {
    for (const dc of ['DL001', 'DL002', 'DL003']) {
      const resp = await request.get(`${API}/api/dealer/${dc}/demand-forecast`, {
        headers: { Authorization: `Bearer ${oemToken}` },
      });
      expect(resp.status()).not.toBe(403);
    }
  });

  test('forecast response contains parts array', async ({ request }) => {
    const resp = await request.get(`${API}/api/dealer/DL001/demand-forecast`, {
      headers: { Authorization: `Bearer ${dealerToken}` },
    });
    if (resp.ok()) {
      const body = await resp.json();
      expect(Array.isArray(body)).toBeTruthy();
      if (body.length > 0) {
        expect(body[0]).toHaveProperty('part_code');
        // The response uses 'description' for the part display name
        expect(body[0]).toHaveProperty('description');
      }
    }
  });
});

// ── OEM model-health contract ─────────────────────────────────────────────────

test.describe('OEM model-health API', () => {
  let oemToken: string;

  test.beforeAll(async ({ request }) => {
    const resp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'oem', password: 'oem123' },
    });
    oemToken = (await resp.json()).access_token;
  });

  test('returns 200 with models array for OEM', async ({ request }) => {
    const resp = await request.get(`${API}/api/oem/model-health`, {
      headers: { Authorization: `Bearer ${oemToken}` },
    });
    expect(resp.ok()).toBeTruthy();
    const body = await resp.json();
    expect(body).toHaveProperty('models');
    expect(Array.isArray(body.models)).toBeTruthy();
  });

  test('model health response includes status field per model', async ({ request }) => {
    const resp = await request.get(`${API}/api/oem/model-health`, {
      headers: { Authorization: `Bearer ${oemToken}` },
    });
    const { models } = await resp.json();
    for (const m of models.slice(0, 3)) {
      // API returns 'model_name' (not 'model') and 'status' (not 'trained')
      expect(m).toHaveProperty('model_name');
      expect(m).toHaveProperty('status');
    }
  });

  test('model health returns 403 for dealer', async ({ request }) => {
    const tokenResp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'dealer', password: 'dealer123' },
    });
    const dealerToken = (await tokenResp.json()).access_token;
    const resp = await request.get(`${API}/api/oem/model-health`, {
      headers: { Authorization: `Bearer ${dealerToken}` },
    });
    expect(resp.status()).toBe(403);
  });
});

// ── Inventory stock API ───────────────────────────────────────────────────────

test.describe('Inventory stock API — dealer scoping', () => {
  test('dealer stock response is scoped to their dealer_code', async ({ request }) => {
    const tokenResp = await request.post(`${API}/api/auth/token`, {
      data: { username: 'dealer', password: 'dealer123' },
    });
    const token = (await tokenResp.json()).access_token;

    const resp = await request.get(`${API}/api/inventory/stock`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (resp.ok()) {
      const body = await resp.json();
      const items = Array.isArray(body) ? body : body.items ?? [];
      const otherDealers = items.filter(
        (item: any) => item.dealer_code && item.dealer_code !== 'DL001',
      );
      expect(otherDealers.length).toBe(0);
    }
  });
});
