// Renders the HTML diagrams in this folder to PNG files in assets/diagrams/.
// Usage: node scripts/diagrams/render.js   (needs: npm i playwright)
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

const here = __dirname;
const out = path.resolve(here, "..", "..", "assets", "diagrams");
fs.mkdirSync(out, { recursive: true });

const jobs = [
  { file: "banner.html", sel: "#banner", name: "banner", themes: ["dark"] },
  { file: "architecture.html", sel: "#diagram", name: "architecture", themes: ["light", "dark"] },
];

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1400, height: 900 } });
  for (const job of jobs) {
    await page.goto("file://" + path.join(here, job.file));
    for (const theme of job.themes) {
      await page.evaluate((t) => (document.body.dataset.theme = t), theme);
      await page.evaluate(() => document.fonts.ready);
      const suffix = job.themes.length > 1 ? `-${theme}` : "";
      const target = path.join(out, `${job.name}${suffix}.png`);
      await page.locator(job.sel).screenshot({ path: target, omitBackground: job.name === "banner" });
      console.log("wrote", target);
    }
  }
  await browser.close();
})();
