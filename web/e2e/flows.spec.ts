import { expect, test } from "@playwright/test";

/** The four core flows from the brief: score an applicant, run a policy simulation, promote a model, view drift. */

test.describe.configure({ mode: "serial" });

test("score an applicant", async ({ page }) => {
  await page.goto("/score");
  await expect(page.getByRole("heading", { name: "Score an applicant" })).toBeVisible();
  await expect(page.getByLabel("EXT_SOURCE_2")).toBeVisible();
  await page.getByTestId("assess").click();
  const result = page.getByTestId("score-result");
  await expect(result).toBeVisible();
  await expect(page.getByTestId("pd-value")).toHaveText(/^\d+\.\d%$/);
  await expect(result.getByText(/Within threshold|Flagged for review/).first()).toBeVisible();
  await expect(page.getByRole("figure", { name: "Why this score" })).toBeVisible();
  await expect(page.getByText(/approve|deny/i)).toHaveCount(0);
});

test("what-if mode re-scores when an input changes", async ({ page }) => {
  await page.goto("/score");
  await expect(page.getByLabel("EXT_SOURCE_2")).toBeVisible();
  await page.getByLabel("What-if mode").click();
  await expect(page.getByTestId("pd-value")).toBeVisible();
  const before = await page.getByTestId("pd-value").textContent();
  await page.getByLabel("EXT_SOURCE_2", { exact: true }).fill("0.02");
  await expect(page.getByTestId("pd-value")).not.toHaveText(before ?? "", { timeout: 10_000 });
});

test("run a policy simulation", async ({ page }) => {
  await page.goto("/policies");
  await expect(page.getByText("Draft policy")).toBeVisible();
  await expect(page.getByText("Valid", { exact: true })).toBeVisible();
  await page.getByTestId("simulate").click();
  const res = page.getByTestId("simulation-result");
  await expect(res).toBeVisible();
  await expect(res.getByRole("cell", { name: /^Flag rate/ })).toBeVisible();
  await expect(page.getByText("Confusion matrices")).toBeVisible();
});

test("promote a model (admin)", async ({ page }) => {
  await page.request.post("/api/session", { data: { role: "admin" } });
  await page.goto("/models");
  const rows = page.locator("[data-testid^=promote-]");
  await expect(rows.first()).toBeVisible();
  const id = (await rows.first().getAttribute("data-testid"))!.replace("promote-", "");
  await rows.first().click();
  await page.getByTestId("confirm-promote").click();
  await expect(page.getByText(new RegExp(`v${id} is now serving`))).toBeVisible();
  await expect(page.getByRole("link", { name: new RegExp(`model v${id}`) }).first()).toBeVisible();
  // restore the previous production version so the suite is repeatable
  const list = await (await page.request.get("/api/proxy/v1/models")).json();
  const previous = list.versions.find((v: { aliases: string[] }) => v.aliases.includes("previous"));
  if (previous) await page.request.post(`/api/proxy/v1/models/${previous.version}/promote`, { data: { alias: "production" } });
  await page.request.post("/api/session", { data: { role: "analyst" } });
});

test("view drift", async ({ page }) => {
  await page.goto("/monitoring");
  await page.getByTestId("replay-demo").first().click();
  await expect(page.getByText(/Drift check complete/)).toBeVisible({ timeout: 30_000 });
  const summary = page.getByTestId("drift-summary");
  await expect(summary.getByText(/Warning|Critical/).first()).toBeVisible();
  await expect(page.getByRole("figure", { name: "PSI by input" })).toBeVisible();
  await expect(page.getByText("Alert history")).toBeVisible();
});
