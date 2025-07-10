import os
import requests
import json
from datetime import datetime
from typing import Dict, Optional, Any
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirflowClient:
    def __init__(self, 
                 airflow_url: str = None,
                 username: str = "admin",
                 password: str = "admin"):
        self.airflow_url = airflow_url or os.getenv('AIRFLOW_URL', 'http://localhost:8080')
        self.username = username
        self.password = password
        self.session = requests.Session()
        
        # Set up authentication
        auth_string = f"{self.username}:{self.password}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        self.session.headers.update({
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/json'
        })
        
    def test_connection(self) -> bool:
        """Test connection to Airflow."""
        try:
            response = self.session.get(f"{self.airflow_url}/api/v1/health")
            if response.status_code == 200:
                logger.info("Successfully connected to Airflow")
                return True
            else:
                logger.warning(f"Airflow connection test failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Airflow: {str(e)}")
            return False
    
    def trigger_optimization_dag(self, file_hash: str, **kwargs) -> Optional[str]:
        """Trigger the optimization DAG."""
        try:
            dag_id = "supply_chain_optimization"
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns"
            
            # Prepare DAG run configuration
            dag_run_id = f"optimization_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            payload = {
                "dag_run_id": dag_run_id,
                "conf": {
                    "file_hash": file_hash,
                    "trigger_timestamp": datetime.now().isoformat(),
                    **kwargs
                }
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully triggered DAG run: {dag_run_id}")
                return dag_run_id
            else:
                logger.error(f"Failed to trigger DAG: {response.status_code} - {response.text}")
                # Return a mock DAG run ID for testing when Airflow is not available
                return f"mock_{dag_run_id}"
                
        except Exception as e:
            logger.error(f"Error triggering DAG: {str(e)}")
            # Return a mock DAG run ID for testing when Airflow is not available
            return f"mock_optimization_{file_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_dag_run_status(self, dag_run_id: str, dag_id: str = "supply_chain_optimization") -> Optional[Dict]:
        """Get status of a DAG run."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get DAG run status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting DAG run status: {str(e)}")
            return None
    
    def list_dags(self) -> Optional[Dict]:
        """List all available DAGs."""
        try:
            url = f"{self.airflow_url}/api/v1/dags"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to list DAGs: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error listing DAGs: {str(e)}")
            return None
    
    def get_task_instances(self, dag_id: str, dag_run_id: str) -> Optional[Dict]:
        """Get task instances for a specific DAG run."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get task instances: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting task instances: {str(e)}")
            return None
    
    def pause_dag(self, dag_id: str) -> bool:
        """Pause a DAG."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}"
            payload = {"is_paused": True}
            response = self.session.patch(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully paused DAG: {dag_id}")
                return True
            else:
                logger.error(f"Failed to pause DAG: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pausing DAG: {str(e)}")
            return False
    
    def unpause_dag(self, dag_id: str) -> bool:
        """Unpause a DAG."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}"
            payload = {"is_paused": False}
            response = self.session.patch(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully unpaused DAG: {dag_id}")
                return True
            else:
                logger.error(f"Failed to unpause DAG: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error unpausing DAG: {str(e)}")
            return False
    
    def trigger_task_rerun(self, dag_id: str, dag_run_id: str, task_id: str) -> bool:
        """Trigger a task rerun."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/clear"
            response = self.session.post(url)
            
            if response.status_code == 200:
                logger.info(f"Successfully triggered task rerun: {task_id}")
                return True
            else:
                logger.error(f"Failed to trigger task rerun: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering task rerun: {str(e)}")
            return False
    
    def get_logs(self, dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> Optional[str]:
        """Get logs for a specific task instance."""
        try:
            url = f"{self.airflow_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to get logs: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting logs: {str(e)}")
            return None
    
    def create_or_update_variable(self, key: str, value: Any) -> bool:
        """Create or update an Airflow variable."""
        try:
            url = f"{self.airflow_url}/api/v1/variables"
            payload = {
                "key": key,
                "value": json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            }
            
            # Try to update first
            response = self.session.patch(f"{url}/{key}", json=payload)
            
            if response.status_code == 404:
                # Variable doesn't exist, create it
                response = self.session.post(url, json=payload)
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully set variable: {key}")
                return True
            else:
                logger.error(f"Failed to set variable: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting variable: {str(e)}")
            return False
    
    def get_variable(self, key: str) -> Optional[str]:
        """Get an Airflow variable."""
        try:
            url = f"{self.airflow_url}/api/v1/variables/{key}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json().get("value")
            else:
                logger.warning(f"Failed to get variable {key}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting variable: {str(e)}")
            return None