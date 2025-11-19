-- Migration script to add Zone-User 1:N relationship
-- This adds zone_id foreign key to user table
-- Run this AFTER creating working_zone table

-- Step 1: Add zone_id column to user table (nullable initially)
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS zone_id VARCHAR(50);

-- Step 2: Add foreign key constraint
-- This ensures zone_id in user table references zone_id in working_zone table
ALTER TABLE "user" 
ADD CONSTRAINT fk_user_zone 
FOREIGN KEY (zone_id) 
REFERENCES working_zone(zone_id) 
ON DELETE SET NULL  -- If zone is deleted, set user's zone_id to NULL
ON UPDATE CASCADE;  -- If zone_id is updated, update user's zone_id too

-- Step 3: Create index on zone_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_zone_id ON "user"(zone_id);

-- Step 4: Display updated table structure
\d "user"

-- Step 5: Display all users with their zones
SELECT 
    u.id,
    u.global_id,
    u.name,
    u.zone_id,
    z.zone_name,
    u.created_at,
    u.updated_at
FROM "user" u
LEFT JOIN working_zone z ON u.zone_id = z.zone_id
ORDER BY u.global_id;

-- Optional: Update existing users to assign them to zones
-- Uncomment and modify as needed:
-- UPDATE "user" SET zone_id = 'ZONE_001' WHERE global_id IN (1, 2, 3);

