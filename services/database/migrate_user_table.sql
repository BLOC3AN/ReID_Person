-- Migration script to add global_id, created_at, updated_at columns to user table
-- Run this script to update the existing user table schema

-- Add global_id column (use id as default value for existing records)
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS global_id INTEGER;

-- Set global_id to id for existing records
UPDATE "user" SET global_id = id WHERE global_id IS NULL;

-- Make global_id NOT NULL and UNIQUE
ALTER TABLE "user" ALTER COLUMN global_id SET NOT NULL;
ALTER TABLE "user" ADD CONSTRAINT user_global_id_unique UNIQUE (global_id);

-- Add created_at column with default value
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();

-- Add updated_at column with default value
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

-- Update existing records to have timestamps
UPDATE "user" SET created_at = NOW() WHERE created_at IS NULL;
UPDATE "user" SET updated_at = NOW() WHERE updated_at IS NULL;

-- Create index on global_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_global_id ON "user"(global_id);

-- Display updated table structure
\d "user"

-- Display all records
SELECT * FROM "user";

