import { defineConfig, devices } from "@playwright/test";

/**
 * E2E tests run against a running stack (web + API). Locally: `docker compose up` (web on :3000) or start the
 * API and `pnpm start`, then `E2E_BASE_URL=http://localhost:3000 pnpm test:e2e`. CI starts both first.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] }, testIgnore: /screenshots\.spec/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testMatch: /responsive\.spec/ },
    { name: "screenshots", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, testMatch: /screenshots\.spec/ },
  ],
});
