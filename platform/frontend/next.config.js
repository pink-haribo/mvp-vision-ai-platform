/** @type {import('next').NextConfig} */

// Parse storage endpoint URL to extract hostname, protocol, port
function getStorageRemotePattern() {
  const storageEndpoint = process.env.NEXT_PUBLIC_STORAGE_ENDPOINT;

  if (!storageEndpoint) {
    console.warn('[next.config.js] NEXT_PUBLIC_STORAGE_ENDPOINT not set, using default localhost:30900');
    return {
      protocol: 'http',
      hostname: 'localhost',
      port: '30900',
      pathname: '/**',
    };
  }

  try {
    const url = new URL(storageEndpoint);
    return {
      protocol: url.protocol.replace(':', ''),
      hostname: url.hostname,
      port: url.port,
      pathname: '/**',
    };
  } catch (error) {
    console.error('[next.config.js] Invalid NEXT_PUBLIC_STORAGE_ENDPOINT:', storageEndpoint);
    return {
      protocol: 'http',
      hostname: 'localhost',
      port: '30900',
      pathname: '/**',
    };
  }
}

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
    NEXT_PUBLIC_STORAGE_ENDPOINT: process.env.NEXT_PUBLIC_STORAGE_ENDPOINT || 'http://localhost:30900',
  },
  images: {
    remotePatterns: [
      // Cloudflare R2 (production)
      {
        protocol: 'https',
        hostname: '*.r2.cloudflarestorage.com',
        port: '',
        pathname: '/**',
      },
      // Dynamic storage endpoint (MinIO local / S3 / etc.)
      getStorageRemotePattern(),
    ],
  },
}

module.exports = nextConfig
