-- TruthSeeQ Database Initialization Script
-- This script runs when the PostgreSQL container starts for the first time

-- Create the main database (already created by POSTGRES_DB env var)
-- Create test database for testing
CREATE DATABASE truthseeq_test;

-- Grant privileges to the truthseeq user
GRANT ALL PRIVILEGES ON DATABASE truthseeq TO truthseeq;
GRANT ALL PRIVILEGES ON DATABASE truthseeq_test TO truthseeq;

-- Connect to the main database
\c truthseeq;

-- Create extensions that might be needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Set timezone
SET timezone = 'UTC';

-- Log successful initialization
SELECT 'TruthSeeQ database initialized successfully' as status; 