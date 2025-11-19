-- Drop and recreate user table with correct schema
-- IMPORTANT: global_id can be duplicated (NO UNIQUE constraint)
-- id is PRIMARY KEY and auto-increment
-- Multiple users can have the same global_id

-- Drop existing table
DROP TABLE IF EXISTS "user" CASCADE;

-- Drop sequence if exists
DROP SEQUENCE IF EXISTS user_id_seq CASCADE;

-- Create sequence for id
CREATE SEQUENCE user_id_seq START 1;

-- Create user table with quoted name (user is a reserved keyword in PostgreSQL)
-- Note: global_id does NOT have UNIQUE constraint - duplicates are allowed
CREATE TABLE "user" (
    id INTEGER PRIMARY KEY DEFAULT nextval('user_id_seq'),
    global_id INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    zone_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes (but NOT unique constraint on global_id)
CREATE INDEX idx_user_global_id ON "user"(global_id);
CREATE INDEX idx_user_zone_id ON "user"(zone_id);

-- Add foreign key to working_zone
ALTER TABLE "user"
ADD CONSTRAINT fk_user_zone
FOREIGN KEY (zone_id)
REFERENCES working_zone(zone_id)
ON DELETE SET NULL
ON UPDATE CASCADE;

-- Insert sample data
INSERT INTO "user" (global_id, name, zone_id) VALUES
    (1, 'Duong', 'ZONE_001'),
    (2, 'Khiem', 'ZONE_001'),
    (3, 'Dang', 'ZONE_002'),
    (4, 'Thanh', 'ZONE_002');

