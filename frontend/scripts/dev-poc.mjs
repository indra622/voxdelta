import { randomBytes } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { parse } from "dotenv";

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const backend = resolve(frontend, "../backend");
const viteArguments = process.argv.slice(2);
// A dedicated PoC env keeps experiments off the working backend configuration; without one,
// the launcher uses the same `.env` the backend itself reads rather than inventing settings.
const envFile = [resolve(backend, ".env.poc"), resolve(backend, ".env")].find(existsSync);
if (!envFile) {
  console.error("No backend/.env.poc or backend/.env to launch the PoC with.");
  process.exit(1);
}
const selectedEnv = parse(readFileSync(envFile));
const capabilityToken = randomBytes(32).toString("hex");
const environment = {
  ...process.env,
  ...selectedEnv,
  VOXDELTA_API_CAPABILITY_TOKEN: capabilityToken,
};
// The env file also holds provider API keys, which only the backend needs. The dev server is
// given the application settings and the capability token, and nothing else from that file.
const uiEnvironment = {
  ...process.env,
  ...Object.fromEntries(
    Object.entries(selectedEnv).filter(([name]) => name.startsWith("VOXDELTA_")),
  ),
  VOXDELTA_API_CAPABILITY_TOKEN: capabilityToken,
};

const children = [
  spawn("uv", ["run", "uvicorn", "voxdelta.api.app:app", "--host", "127.0.0.1", "--port", "8765"], {
    cwd: backend,
    env: environment,
    stdio: "inherit",
  }),
  spawn("npm", ["exec", "vite", "--", "--mode", "poc", ...viteArguments], {
    cwd: frontend,
    env: uiEnvironment,
    stdio: "inherit",
  }),
];

let shuttingDown = false;
function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill("SIGTERM");
  }
  setTimeout(() => process.exit(code), 100).unref();
}

for (const child of children) {
  child.on("error", (error) => {
    console.error(error.message);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (!shuttingDown && code !== null && code !== 0) {
      console.error("PoC process exited with code " + code + (signal ? " (" + signal + ")" : ""));
      shutdown(code);
    }
  });
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
