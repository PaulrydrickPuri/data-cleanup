// vite.config.js
export default {
  // Base public path when served in production
  base: './',
  
  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: process.env.NODE_ENV !== 'production',
    
    // Generate a manifest file for asset mapping
    manifest: true,
    
    // Customize output file names with content hashing for caching
    rollupOptions: {
      output: {
        entryFileNames: 'assets/js/[name].[hash].js',
        chunkFileNames: 'assets/js/[name].[hash].js',
        assetFileNames: 'assets/[ext]/[name].[hash].[ext]'
      }
    }
  },
  
  // Server configuration for development
  server: {
    port: 3000,
    strictPort: true,
    
    // Configure proxy for API requests in development
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  },
  
  // Optimization options
  optimizeDeps: {
    include: ['chart.js']
  }
}
