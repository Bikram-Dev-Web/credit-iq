import { expect, test } from "@playwright/test";

/** 375 px-class viewport: no horizontal page scroll, navigation reachable from the menu. */
for (const path of ["/", "/score", "/portfolio", "/policies", "/models", "/monitoring"]) {
  test(`no horizontal overflow on ${path}`, async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
}

test("mobile navigation opens", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await page.goto("/");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("link", { name: "Portfolio" }).click();
  await expect(page.getByRole("heading", { name: "Portfolio" })).toBeVisible();
});
