"""
Database Manager for Supply Chain Optimization
Handles data storage and retrieval for optimization results
"""

import pandas as pd
import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for the supply chain optimizer."""
    
    def __init__(self, db_path: str = "data/optimization.db"):
        """Initialize database manager."""
        self.db_path = db_path
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def _create_tables(self):
        """Create necessary database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Baseline data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    data_json TEXT NOT NULL
                )
            """)
            
            # Optimization results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    optimization_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_savings REAL,
                    savings_percentage REAL,
                    execution_time REAL,
                    results_json TEXT NOT NULL,
                    FOREIGN KEY (file_hash) REFERENCES baseline_data (file_hash)
                )
            """)
            
            # Constraints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS constraints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    constraint_type TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_hash) REFERENCES baseline_data (file_hash)
                )
            """)
            
            conn.commit()
    
    def store_baseline_data(self, data: pd.DataFrame, file_hash: str, filename: str) -> bool:
        """Store baseline data in the database."""
        try:
            data_json = data.to_json(orient='records')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO baseline_data 
                    (file_hash, filename, data_json) 
                    VALUES (?, ?, ?)
                """, (file_hash, filename, data_json))
                conn.commit()
            
            logger.info(f"Baseline data stored for file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing baseline data: {str(e)}")
            return False
    
    def store_optimization_results(self, file_hash: str, results: Dict[str, Any]) -> bool:
        """Store optimization results in the database."""
        try:
            # Remove the dataframe from results for JSON serialization
            results_copy = results.copy()
            if 'results_dataframe' in results_copy:
                results_copy['results_dataframe'] = results_copy['results_dataframe'].to_json(orient='records')
            
            results_json = json.dumps(results_copy, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO optimization_results 
                    (file_hash, total_savings, savings_percentage, execution_time, results_json) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    file_hash,
                    results.get('total_savings', 0),
                    results.get('savings_percentage', 0),
                    results.get('execution_time', 0),
                    results_json
                ))
                conn.commit()
            
            logger.info(f"Optimization results stored for hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
            return False
    
    def store_constraints(self, file_hash: str, supplier_constraints: Dict, plant_constraints: Dict) -> bool:
        """Store constraints in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store supplier constraints
                supplier_json = json.dumps(supplier_constraints)
                conn.execute("""
                    INSERT OR REPLACE INTO constraints 
                    (file_hash, constraint_type, constraints_json) 
                    VALUES (?, ?, ?)
                """, (file_hash, 'supplier', supplier_json))
                
                # Store plant constraints
                plant_json = json.dumps(plant_constraints)
                conn.execute("""
                    INSERT OR REPLACE INTO constraints 
                    (file_hash, constraint_type, constraints_json) 
                    VALUES (?, ?, ?)
                """, (file_hash, 'plant', plant_json))
                
                conn.commit()
            
            logger.info(f"Constraints stored for hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing constraints: {str(e)}")
            return False
    
    def get_baseline_data(self, file_hash: str) -> Optional[pd.DataFrame]:
        """Retrieve baseline data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data_json FROM baseline_data WHERE file_hash = ?
                """, (file_hash,))
                
                result = cursor.fetchone()
                if result:
                    data_json = result[0]
                    return pd.read_json(data_json, orient='records')
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving baseline data: {str(e)}")
            return None
    
    def get_optimization_results(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve optimization results from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT results_json FROM optimization_results 
                    WHERE file_hash = ? 
                    ORDER BY optimization_timestamp DESC 
                    LIMIT 1
                """, (file_hash,))
                
                result = cursor.fetchone()
                if result:
                    results_json = result[0]
                    results = json.loads(results_json)
                    
                    # Convert dataframe back
                    if 'results_dataframe' in results and isinstance(results['results_dataframe'], str):
                        results['results_dataframe'] = pd.read_json(results['results_dataframe'], orient='records')
                    
                    return results
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving optimization results: {str(e)}")
            return None