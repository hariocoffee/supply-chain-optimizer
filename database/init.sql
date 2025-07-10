-- Create tables for the optimizer application

-- Table to store data uploads
CREATE TABLE IF NOT EXISTS data_uploads (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    unique_plants INTEGER NOT NULL,
    unique_suppliers INTEGER NOT NULL,
    total_volume FLOAT NOT NULL,
    data_sample JSONB
);

-- Table to store optimization runs
CREATE TABLE IF NOT EXISTS optimization_runs (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) NOT NULL,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supplier_constraints JSONB,
    plant_constraints JSONB,
    optimization_status VARCHAR(50) DEFAULT 'pending',
    total_savings FLOAT,
    savings_percentage FLOAT,
    execution_time FLOAT,
    airflow_dag_run_id VARCHAR(255)
);

-- Table to store optimization results
CREATE TABLE IF NOT EXISTS optimization_results (
    id SERIAL PRIMARY KEY,
    optimization_run_id INTEGER NOT NULL,
    plant VARCHAR(100) NOT NULL,
    supplier VARCHAR(100) NOT NULL,
    baseline_volume FLOAT NOT NULL,
    optimized_volume FLOAT NOT NULL,
    baseline_cost FLOAT NOT NULL,
    optimized_cost FLOAT NOT NULL,
    cost_savings FLOAT NOT NULL,
    volume_split FLOAT NOT NULL,
    selection_flag BOOLEAN DEFAULT FALSE
);

-- Table for caching entries
CREATE TABLE IF NOT EXISTS cache_entries (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(64) UNIQUE NOT NULL,
    cached_data JSONB,
    expiry_timestamp TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_data_uploads_file_hash ON data_uploads(file_hash);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_file_hash ON optimization_runs(file_hash);
CREATE INDEX IF NOT EXISTS idx_optimization_results_run_id ON optimization_results(optimization_run_id);
CREATE INDEX IF NOT EXISTS idx_cache_entries_expires ON cache_entries(expiry_timestamp);

-- Create function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM cache_entries WHERE expiry_timestamp < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;