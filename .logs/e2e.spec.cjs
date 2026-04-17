const { test, expect } = require("@playwright/test");

test("frontend loads and optimization flow works", async ({ page }) => {
  const consoleErrors = [];
  const pageErrors = [];
  const failedRequests = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  page.on("pageerror", (err) => pageErrors.push(String(err)));

  page.on("requestfailed", (req) => {
    failedRequests.push(`${req.method()} ${req.url()} :: ${req.failure()?.errorText || "unknown"}`);
  });

  const resp = await page.goto("http://127.0.0.1:5173/", { waitUntil: "domcontentloaded", timeout: 30000 });
  expect(resp && resp.ok()).toBeTruthy();

  const runBtn = page.getByRole("button", { name: "Run Backend Optimization" });
  await expect(runBtn).toBeVisible({ timeout: 30000 });
  await runBtn.click();

  await expect(runBtn).toBeDisabled();
  await expect(runBtn).toBeEnabled({ timeout: 180000 });

  const depots = page.getByText(/^Depots:\s*\d+$/);
  const routes = page.getByText(/^Routes:\s*\d+$/);

  await expect(depots).toBeVisible();
  await expect(routes).toBeVisible();

  const depotsText = (await depots.textContent()) || "";
  const routesText = (await routes.textContent()) || "";

  const depotsCount = Number((depotsText.match(/\d+/) || ["0"])[0]);
  const routesCount = Number((routesText.match(/\d+/) || ["0"])[0]);

  expect(depotsCount).toBeGreaterThan(0);
  expect(routesCount).toBeGreaterThan(0);

  const errorBanner = page.getByText(/Backend error|Invalid backend response|Failed to run optimization/i);
  await expect(errorBanner).toHaveCount(0);

  if (consoleErrors.length) {
    console.log("CONSOLE_ERRORS_START");
    consoleErrors.forEach((e) => console.log(e));
    console.log("CONSOLE_ERRORS_END");
  }
  if (pageErrors.length) {
    console.log("PAGE_ERRORS_START");
    pageErrors.forEach((e) => console.log(e));
    console.log("PAGE_ERRORS_END");
  }
  if (failedRequests.length) {
    console.log("FAILED_REQUESTS_START");
    failedRequests.forEach((e) => console.log(e));
    console.log("FAILED_REQUESTS_END");
  }

  console.log(`CHECK_SUMMARY depots=${depotsCount} routes=${routesCount} consoleErrors=${consoleErrors.length} pageErrors=${pageErrors.length} failedRequests=${failedRequests.length}`);
});
