-- ==============================================================================
-- OphthalmoAI PostgreSQL Initialization Script
-- Executed on container startup by /docker-entrypoint-initdb.d/
-- ==============================================================================

-- Enable standard UUID generation and cryptographic extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Set default timezone to UTC for consistent clinical timestamp audits
SET TIME ZONE 'UTC';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'OphthalmoAI PostgreSQL database initialized with uuid-ossp and pgcrypto extensions in UTC timezone.';
END $$;
