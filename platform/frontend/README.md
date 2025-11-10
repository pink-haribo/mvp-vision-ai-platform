# Frontend Application

**Next.js 14 App Router** based frontend with TypeScript and Tailwind CSS.

## Structure

```
frontend/
├── app/              # App router pages
│   ├── page.tsx     # Home page
│   ├── training/    # Training pages
│   └── layout.tsx   # Root layout
├── components/      # React components
│   ├── ui/         # UI primitives
│   └── features/   # Feature components
└── lib/            # Utilities
    ├── api.ts      # API client
    └── websocket.ts # WebSocket client
```

## Development

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev

# Build for production
pnpm build

# Run tests
pnpm test
```

## Features

- Chat-based training configuration
- Real-time training monitoring (WebSocket)
- Dataset management
- Model selection and configuration
- Training metrics visualization
