import { defineConfig, devices } from "@playwright/test";

// Keep the suite off a preview that may already be running on the default port.
const port = process.env.VOXDELTA_POC_PORT ?? "5173";
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests",
  timeout: 20_000,
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: baseURL,
    env: { VOXDELTA_POC_PORT: port },
    reuseExistingServer: false,
    timeout: 20_000,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});
