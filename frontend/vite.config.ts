import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const backendEnv = loadEnv(mode, "../backend", "");
  const capability = process.env.VOXDELTA_API_CAPABILITY_TOKEN ?? backendEnv.VOXDELTA_API_CAPABILITY_TOKEN;
  // Names a reverse proxy may present in Host, listed one by one. Vite rejects any other
  // name so a rebound DNS record cannot reach this server; `true` would remove that guard.
  const allowedHosts = (
    process.env.VOXDELTA_POC_ALLOWED_HOSTS ??
    backendEnv.VOXDELTA_POC_ALLOWED_HOSTS ??
    ""
  )
    .split(",")
    .map((host) => host.trim())
    .filter(Boolean);

  // A second copy of this UI may already be serving the default port; tests and side-by-side
  // runs pick their own rather than taking it over.
  const port = Number(process.env.VOXDELTA_POC_PORT ?? backendEnv.VOXDELTA_POC_PORT ?? 5173);

  return {
    envDir: "../backend",
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port,
      strictPort: true,
      allowedHosts,
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8765",
          changeOrigin: true,
          configure(proxy) {
            proxy.on("proxyReq", (proxyRequest) => {
              if (capability) {
                proxyRequest.setHeader("X-VoxDelta-Token", capability);
              }
              if (proxyRequest.hasHeader("Origin")) {
                proxyRequest.setHeader("Origin", `http://127.0.0.1:${port}`);
              }
            });
          },
        },
      },
    },
    test: {
      environment: "jsdom",
      // Testing Library registers its DOM cleanup through the global afterEach hook.
      globals: true,
      setupFiles: "./src/test/setup.ts",
      include: ["src/**/*.test.{ts,tsx}"],
      css: true,
    },
  };
});
