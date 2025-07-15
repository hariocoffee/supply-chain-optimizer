"""
Airflow Client for Supply Chain Optimization
Handles communication with Apache Airflow for orchestrating optimization workflows
"""

import requests
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AirflowClient:
    """Client for interacting with Apache Airflow."""
    
    def __init__(self, airflow_url: str = "http://airflow:8080", 
                 username: str = "admin", password: str = "admin"):
        """Initialize Airflow client."""
        self.airflow_url = airflow_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.auth = (username, password)
    
    def trigger_optimization_dag(self, file_hash: str, 
                                dag_id: str = "supply_chain_optimization") -> Optional[str]:
        """Trigger the optimization DAG in Airflow."""
        try:
            # Generate unique run ID
            run_id = f"optimization_{file_hash[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare DAG run configuration
            dag_run_config = {
                "conf": {
                    "file_hash": file_hash,
                    "triggered_by": "supply_chain_optimizer",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # API endpoint for triggering DAG
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns"
            
            payload = {
                "dag_run_id": run_id,
                "conf": dag_run_config["conf"]
            }
            
            # Make request to Airflow API
            response = self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"DAG triggered successfully: {run_id}")
                return run_id
            else:
                logger.error(f"Failed to trigger DAG: {response.status_code} - {response.text}")
                return self._create_mock_dag_run_id(file_hash)
                
        except Exception as e:
            logger.warning(f"Airflow not available, creating mock DAG run: {str(e)}")
            return self._create_mock_dag_run_id(file_hash)
    
    def _create_mock_dag_run_id(self, file_hash: str) -> str:
        """Create a mock DAG run ID when Airflow is not available."""
        return f"mock_optimization_{file_hash[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_dag_run_status(self, dag_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a DAG run."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get DAG run status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting DAG run status: {str(e)}")
            return None
    
    def get_dag_run_logs(self, dag_id: str, run_id: str, task_id: str) -> Optional[str]:
        """Get logs for a specific task in a DAG run."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/1"
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"Failed to get DAG run logs: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting DAG run logs: {str(e)}")
            return None
    
    def is_airflow_available(self) -> bool:
        """Check if Airflow is available and responding."""
        try:
            url = f"{self.airflow_url}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False