-- Migration script to create working_zone table in PostgreSQL
-- Run this script to create the working_zone table for zone management

-- Create working_zone table
CREATE TABLE IF NOT EXISTS working_zone (
    zone_id VARCHAR(50) PRIMARY KEY,
    zone_name VARCHAR(255) NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    x3 FLOAT NOT NULL,
    y3 FLOAT NOT NULL,
    x4 FLOAT NOT NULL,
    y4 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index on zone_name for searching
CREATE INDEX IF NOT EXISTS idx_working_zone_zone_name ON working_zone(zone_name);

-- Display table structure
\d working_zone

-- Display all records
SELECT * FROM working_zone;

