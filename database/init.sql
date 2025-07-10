-- Create tables for the optimizer application

-- Table to store optimization runs
CREATE TABLE IF NOT EXISTS optimization_runs (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    total_savings DECIMAL(15,2),
    optimization_timestamp TIMESTAMP,
    constraints_json TEXT,
    results_json TEXT
);

-- Table to store baseline data
CREATE TABLE IF NOT EXISTS baseline_data (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id),
    plant VARCHAR(255) NOT NULL,
    product VARCHAR(255) NOT NULL,
    volume_lbs BIGINT NOT NULL,
    supplier VARCHAR(255) NOT NULL,
    plant_location VARCHAR(255) NOT NULL,
    ddp_price DECIMAL(10,2) NOT NULL,
    baseline_allocated_volume BIGINT,
    baseline_price_paid DECIMAL(15,2),
    selection VARCHAR(10),
    split_percentage DECIMAL(5,2)
);

-- Table to store optimized results
CREATE TABLE IF NOT EXISTS optimized_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id),
    plant VARCHAR(255) NOT NULL,
    product VARCHAR(255) NOT NULL,
    volume_lbs BIGINT NOT NULL,
    supplier VARCHAR(255) NOT NULL,
    plant_location VARCHAR(255) NOT NULL,
    ddp_price DECIMAL(10,2) NOT NULL,
    optimized_volume BIGINT,
    optimized_price DECIMAL(15,2),
    optimized_selection VARCHAR(10),
    optimized_split DECIMAL(5,2),
    cost_savings DECIMAL(15,2)
);

-- Table to store supplier constraints
CREATE TABLE IF NOT EXISTS supplier_constraints (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id),
    supplier VARCHAR(255) NOT NULL,
    min_volume BIGINT NOT NULL,
    max_volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store plant constraints
CREATE TABLE IF NOT EXISTS plant_constraints (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES optimization_runs(id),
    plant VARCHAR(255) NOT NULL,
    max_volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for caching optimization results
CREATE TABLE IF NOT EXISTS cache_results (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(128) UNIQUE NOT NULL,
    result_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_optimization_runs_file_hash ON optimization_runs(file_hash);
CREATE INDEX IF NOT EXISTS idx_baseline_data_run_id ON baseline_data(run_id);
CREATE INDEX IF NOT EXISTS idx_optimized_results_run_id ON optimized_results(run_id);
CREATE INDEX IF NOT EXISTS idx_supplier_constraints_run_id ON supplier_constraints(run_id);
CREATE INDEX IF NOT EXISTS idx_plant_constraints_run_id ON plant_constraints(run_id);
CREATE INDEX IF NOT EXISTS idx_cache_results_expires_at ON cache_results(expires_at);

-- Create function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM cache_results WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;