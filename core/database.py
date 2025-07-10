import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DataUpload(Base):
    __tablename__ = 'data_uploads'
    
    id = Column(Integer, primary_key=True)
    file_hash = Column(String(64), unique=True, nullable=False)
    filename = Column(String(255), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    unique_plants = Column(Integer, nullable=False)
    unique_suppliers = Column(Integer, nullable=False)
    total_volume = Column(Float, nullable=False)
    data_sample = Column(JSONB)

class OptimizationRun(Base):
    __tablename__ = 'optimization_runs'
    
    id = Column(Integer, primary_key=True)
    file_hash = Column(String(64), nullable=False)
    run_timestamp = Column(DateTime, default=datetime.utcnow)
    supplier_constraints = Column(JSONB)
    plant_constraints = Column(JSONB)
    optimization_status = Column(String(50), default='pending')
    total_savings = Column(Float)
    savings_percentage = Column(Float)
    execution_time = Column(Float)
    airflow_dag_run_id = Column(String(255))

class OptimizationResults(Base):
    __tablename__ = 'optimization_results'
    
    id = Column(Integer, primary_key=True)
    optimization_run_id = Column(Integer, nullable=False)
    plant = Column(String(100), nullable=False)
    supplier = Column(String(100), nullable=False)
    baseline_volume = Column(Float, nullable=False)
    optimized_volume = Column(Float, nullable=False)
    baseline_cost = Column(Float, nullable=False)
    optimized_cost = Column(Float, nullable=False)
    cost_savings = Column(Float, nullable=False)
    volume_split = Column(Float, nullable=False)
    selection_flag = Column(Boolean, default=False)

class CacheEntry(Base):
    __tablename__ = 'cache_entries'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(64), unique=True, nullable=False)
    cached_data = Column(JSONB)
    expiry_timestamp = Column(DateTime, nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://optimizer:optimizer123@localhost:5432/optimizer_db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def store_baseline_data(self, data: pd.DataFrame, file_hash: str, filename: str) -> bool:
        """Store baseline data information."""
        try:
            session = self.get_session()
            
            # Check if file already exists
            existing = session.query(DataUpload).filter_by(file_hash=file_hash).first()
            if existing:
                logger.info(f"File hash {file_hash} already exists in database")
                session.close()
                return True
            
            # Calculate metrics
            total_volume = data['2024 Volume (lbs)'].sum()
            unique_plants = data['Plant'].nunique()
            unique_suppliers = data['Supplier'].nunique()
            
            # Create sample data for storage (first 5 rows)
            sample_data = data.head(5).to_dict('records')
            
            # Create new record
            upload_record = DataUpload(
                file_hash=file_hash,
                filename=filename,
                row_count=len(data),
                column_count=len(data.columns),
                unique_plants=unique_plants,
                unique_suppliers=unique_suppliers,
                total_volume=total_volume,
                data_sample=sample_data
            )
            
            session.add(upload_record)
            session.commit()
            session.close()
            
            logger.info(f"Stored baseline data for file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing baseline data: {str(e)}")
            return False
    
    def store_constraints(self, file_hash: str, supplier_constraints: Dict, plant_constraints: Dict) -> bool:
        """Store optimization constraints."""
        try:
            session = self.get_session()
            
            # Check if optimization run already exists
            existing = session.query(OptimizationRun).filter_by(file_hash=file_hash).first()
            if existing:
                # Update existing constraints
                existing.supplier_constraints = supplier_constraints
                existing.plant_constraints = plant_constraints
                existing.run_timestamp = datetime.utcnow()
            else:
                # Create new optimization run
                opt_run = OptimizationRun(
                    file_hash=file_hash,
                    supplier_constraints=supplier_constraints,
                    plant_constraints=plant_constraints
                )
                session.add(opt_run)
            
            session.commit()
            session.close()
            
            logger.info(f"Stored constraints for file hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing constraints: {str(e)}")
            return False
    
    def store_optimization_results(self, file_hash: str, results: Dict) -> bool:
        """Store optimization results."""
        try:
            session = self.get_session()
            
            # Get optimization run
            opt_run = session.query(OptimizationRun).filter_by(file_hash=file_hash).first()
            if not opt_run:
                logger.error(f"No optimization run found for file hash: {file_hash}")
                return False
            
            # Update optimization run with results
            opt_run.optimization_status = 'completed'
            opt_run.total_savings = results.get('total_savings', 0)
            opt_run.savings_percentage = results.get('savings_percentage', 0)
            opt_run.execution_time = results.get('execution_time', 0)
            opt_run.airflow_dag_run_id = results.get('dag_run_id', '')
            
            # Delete existing results for this run
            session.query(OptimizationResults).filter_by(optimization_run_id=opt_run.id).delete()
            
            # Store detailed results
            if 'detailed_results' in results:
                for result in results['detailed_results']:
                    result_record = OptimizationResults(
                        optimization_run_id=opt_run.id,
                        plant=result['plant'],
                        supplier=result['supplier'],
                        baseline_volume=result['baseline_volume'],
                        optimized_volume=result['optimized_volume'],
                        baseline_cost=result['baseline_cost'],
                        optimized_cost=result['optimized_cost'],
                        cost_savings=result['cost_savings'],
                        volume_split=result['volume_split'],
                        selection_flag=result['selection_flag']
                    )
                    session.add(result_record)
            
            session.commit()
            session.close()
            
            logger.info(f"Stored optimization results for file hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
            return False
    
    def get_optimization_history(self, limit: int = 50) -> List[Dict]:
        """Get optimization history."""
        try:
            session = self.get_session()
            
            runs = session.query(OptimizationRun).order_by(OptimizationRun.run_timestamp.desc()).limit(limit).all()
            
            history = []
            for run in runs:
                # Get associated data upload info
                upload = session.query(DataUpload).filter_by(file_hash=run.file_hash).first()
                
                history.append({
                    'run_id': run.id,
                    'filename': upload.filename if upload else 'Unknown',
                    'run_timestamp': run.run_timestamp,
                    'total_savings': run.total_savings,
                    'savings_percentage': run.savings_percentage,
                    'execution_time': run.execution_time,
                    'status': run.optimization_status
                })
            
            session.close()
            return history
            
        except Exception as e:
            logger.error(f"Error getting optimization history: {str(e)}")
            return []
    
    def get_file_info(self, file_hash: str) -> Optional[Dict]:
        """Get file information by hash."""
        try:
            session = self.get_session()
            
            upload = session.query(DataUpload).filter_by(file_hash=file_hash).first()
            if not upload:
                return None
            
            info = {
                'filename': upload.filename,
                'upload_timestamp': upload.upload_timestamp,
                'row_count': upload.row_count,
                'column_count': upload.column_count,
                'unique_plants': upload.unique_plants,
                'unique_suppliers': upload.unique_suppliers,
                'total_volume': upload.total_volume
            }
            
            session.close()
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def cleanup_old_records(self, days_old: int = 30) -> bool:
        """Clean up old records."""
        try:
            session = self.get_session()
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete old optimization runs and their results
            old_runs = session.query(OptimizationRun).filter(OptimizationRun.run_timestamp < cutoff_date).all()
            
            for run in old_runs:
                session.query(OptimizationResults).filter_by(optimization_run_id=run.id).delete()
                session.delete(run)
            
            # Delete old data uploads
            session.query(DataUpload).filter(DataUpload.upload_timestamp < cutoff_date).delete()
            
            # Delete expired cache entries
            session.query(CacheEntry).filter(CacheEntry.expiry_timestamp < datetime.utcnow()).delete()
            
            session.commit()
            session.close()
            
            logger.info(f"Cleaned up records older than {days_old} days")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {str(e)}")
            return False