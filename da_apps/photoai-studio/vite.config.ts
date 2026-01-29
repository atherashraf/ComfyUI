import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    base: "/",
    plugins: [react()],

    resolve: {
      alias: {
        "@": resolve(__dirname, "src")
      },
      // react: resolve(__dirname, 'node_modules/react'),
      //'react-dom': resolve(__dirname, 'node_modules/react-dom'),
      //'@emotion/react': resolve(__dirname, 'node_modules/@emotion/react'),
      //'@emotion/styled': resolve(__dirname, 'node_modules/@emotion/styled'),
      // ensure single instances when damap is linked
      //dedupe: ["react", "react-dom", "@emotion/react", "@emotion/styled"]
    },

    // helpful when damap is symlinked outside project root
    server: {
      port: parseInt(env.VITE_DEV_PORT || "5173", 10),
      fs: {
        // allow parent folder so npm/yarn link or workspace sibling can be resolved
        allow: [".."]
      }
    },

    // No externals here (apps should not externalize react/react-dom)
    // No optimizeDeps.exclude for react/react-dom either

    assetsInclude: ["**/*.pdf"]
  };
});
