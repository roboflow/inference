import { defineConfig, devices } from "@playwright/test";

// Frontend-only regression coverage for the /build discovery logic in
// src/app/page.tsx. Runs against `next dev` with the /build and
// /dashboard.html network calls intercepted (see e2e/builder-link.spec.ts),
// so no real inference server is required. Real-server, real-browser
// verification against the actual backend lives in
// roboflow/evidence/inference-builder/ (outside this repo checkout).
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3147",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    // Must be "localhost", not "127.0.0.1": Next.js 16 dev server blocks
    // cross-origin dev requests by default (allowedDevOrigins), and treats
    // those as different origins, which silently breaks client hydration.
    command: "npm run dev -- --port 3147",
    url: "http://localhost:3147",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
