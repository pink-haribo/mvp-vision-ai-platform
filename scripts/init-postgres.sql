-- Vision AI Platform - PostgreSQL Initialization Script
-- Platform DB only (vision_platform)

-- =============================================================================
-- Extensions
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Note: Labeler database is managed separately by the Labeler service
-- Labeler will create its own database when it starts
-- =============================================================================
