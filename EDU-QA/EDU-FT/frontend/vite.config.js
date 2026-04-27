import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
server: {
    port: 5174,        // 指定新端口
    strictPort: true   // 建议设为 true。这样如果 5174 又被占用，Vite 会直接报错退出，而不是静默分配 5175
  }
})
