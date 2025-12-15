# Frontend Dockerfile for Development
# Uses npm run dev with hot reload

ARG REGISTRY_MIRROR=
FROM ${REGISTRY_MIRROR}node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source code
COPY . .

# Expose dev server port
EXPOSE 3000

ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Start dev server with hot reload
CMD ["npm", "run", "dev"]
