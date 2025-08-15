/** @type {import('next').NextConfig} */

const isDev = process.env.NODE_ENV === 'development'

const nextConfig = {
  ...(isDev ? {
    // Development: proxy API calls to inference server on port 9001
    async rewrites() {
      return [
        {
          source: '/model/:path*',
          destination: 'http://localhost:9001/model/:path*'
        },
        {
          source: '/info',
          destination: 'http://localhost:9001/info'
        },
        {
          source: '/healthz', 
          destination: 'http://localhost:9001/healthz'
        },
        {
          source: '/readiness',
          destination: 'http://localhost:9001/readiness'
        },
        {
          source: '/metrics',
          destination: 'http://localhost:9001/metrics'
        },
        {
          source: '/notebook/:path*',
          destination: 'http://localhost:9001/notebook/:path*'
        },
        {
          source: '/workflows/execution_engine/versions',
          destination: 'http://localhost:9001/workflows/execution_engine/versions'
        },
        {
          source: '/workflows/blocks/describe',
          destination: 'http://localhost:9001/workflows/blocks/describe'
        },
        {
          source: '/inference_pipelines/:path*',
          destination: 'http://localhost:9001/inference_pipelines/:path*'
        }
      ]
    }
  } : {
    // Production: static export for macOS app bundle
    output: "export"
  })
}

module.exports = nextConfig
