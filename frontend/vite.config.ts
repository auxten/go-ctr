import type { ConfigEnv } from 'vite'
import { loadEnv, defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

import { resolve } from 'path'

function pathResolve(dir: string) {
  return resolve(process.cwd(), '.', dir)
}

// https://vitejs.dev/config/
export default ({ mode }: ConfigEnv) => {
  const dirRoot = process.cwd()

  const env = loadEnv(mode, dirRoot)

  return defineConfig({
    base: env.VITE_PUBLIC_PATH,
    plugins: [
      vue(),
    ],
    server: {
      host: '0.0.0.0',
      port: 8090,
      proxy: {
        '/service/': {
          target: env.VITE_APP_BASE_API,
          changeOrigin: true,
          // autoRewrite: true,
          rewrite: (path: string) => path.replace(/^\/service/, '/'),
        },
      },
    },
    resolve: {
      alias: {
        '@': pathResolve('./src'),
      },
    },
    define: {
      // setting vue-i18-next
      // Suppress warning
      __INTLIFY_PROD_DEVTOOLS__: false,
      __DEV__: process.env.NODE_ENV !== 'production',
      __PROD__: process.env.NODE_ENV === 'production',
    },
    optimizeDeps: {
      include: [
        'axios',
        'dayjs',
        'echarts',
        'lodash-es',
        'naive-ui',
        'shortid',
        'vue',
        'vue-router',
      ],
      exclude: [],
    },
    build: {
      sourcemap: false,
      outDir: 'website',
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'index.html'),
        },
      },
    },
    esbuild: {
    },
  })
}
