import { test } from "@playwright/test";

/** Screenshots for docs/v2/DESIGN.md and the README (both themes). Run: pnpm exec playwright test --project screenshots */
const PAGES = ["/", "/score", "/portfolio", "/policies", "/models", "/monitoring"];

for (const theme of ["dark", "light"] as const) {
  for (const path of PAGES) {
    test(`${theme} ${path}`, async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem("ciq-theme", t), theme);
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      if (path === "/score") {
        await page.getByTestId("assess").click();
        await page.getByTestId("score-result").waitFor();
      }
      if (path === "/policies") {
        await page.getByTestId("simulate").click();
        await page.getByTestId("simulation-result").waitFor();
      }
      await page.waitForTimeout(600);
      const name = path === "/" ? "overview" : path.slice(1);
      await page.screenshot({ path: `../docs/v2/screenshots/${name}-${theme}.png`, fullPage: false });
    });
  }
}
