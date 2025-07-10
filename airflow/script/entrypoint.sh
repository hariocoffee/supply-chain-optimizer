#!/bin/bash

# Airflow entrypoint script
set -e

# Wait for database to be ready
echo "Waiting for database..."
sleep 10

# Initialize Airflow database if needed
if [ "$1" = "webserver" ]; then
    echo "Initializing Airflow database..."
    airflow db init || echo "Database already initialized"
    
    # Create admin user if it doesn't exist
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || echo "Admin user already exists"
fi

# Start the requested service
echo "Starting Airflow $1..."
exec airflow "$@"