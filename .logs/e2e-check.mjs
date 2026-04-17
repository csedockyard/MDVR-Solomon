import { chromium } from "playwright";

const FRONTEND_URL = "http://127.0.0.1:5173/";

const result = {
  ok: false,
  frontendLoaded: false,
  clickedRunButton: false,
  depotsText: null,
  routesText: null,
  errorBannerText: null,
  consoleErrors: [],
  pageErrors: [],
  failedRequests: [],
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext();
const page = await context.newPage();

page.on("console", (msg) => {
  if (msg.type() === "error") {
    result.consoleErrors.push(msg.text());
  }
});

page.on("pageerror", (err) => {
  result.pageErrors.push(String(err));
});

page.on("requestfailed", (req) => {
  result.failedRequests.push(`${req.method()} ${req.url()} :: ${req.failure()?.errorText || "unknown"}`);
});

try {
  const resp = await page.goto(FRONTEND_URL, { waitUntil: "domcontentloaded", timeout: 30000 });
  result.frontendLoaded = !!resp && resp.ok();

  await page.waitForSelector('button:has-text("Run Backend Optimization")', { timeout: 30000 });
  await page.click('button:has-text("Run Backend Optimization")');
  result.clickedRunButton = true;

  await page.waitForFunction(() => {
    const button = Array.from(document.querySelectorAll("button")).find((b) => b.textContent?.includes("Run Backend Optimization") || b.textContent?.includes("Running..."));
    if (!button) return false;
    return button.textContent?.includes("Run Backend Optimization");
  }, null, { timeout: 180000 });

  await page.waitForTimeout(1500);

  const metaTexts = await page.$$eval("div", (nodes) => nodes.map((n) => n.textContent?.trim() || ""));
  result.depotsText = metaTexts.find((t) => /^Depots:\s*\d+/.test(t)) || null;
  result.routesText = metaTexts.find((t) => /^Routes:\s*\d+/.test(t)) || null;

  const errorBanner = await page.$("text=/Backend error|Invalid backend response|Failed to run optimization/i");
  if (errorBanner) {
    result.errorBannerText = (await errorBanner.textContent())?.trim() || "";
  }

  const depotsOk = !!result.depotsText && !/^Depots:\s*0$/.test(result.depotsText);
  const routesOk = !!result.routesText && !/^Routes:\s*0$/.test(result.routesText);

  result.ok = result.frontendLoaded && result.clickedRunButton && depotsOk && routesOk && !result.errorBannerText && result.pageErrors.length === 0;
} catch (e) {
  result.pageErrors.push(`Unhandled test exception: ${String(e)}`);
} finally {
  await browser.close();
}

console.log(JSON.stringify(result, null, 2));
