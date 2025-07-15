"""
Base Optimization Engine for Supply Chain Optimization Platform
Abstract base class for all optimization engines.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Standardized optimization result structure."""
    success: bool
    total_savings: float
    savings_percentage: float
    baseline_total_cost: float
    optimized_total_cost: float
    execution_time: float
    results_dataframe: Optional[pd.DataFrame] = None
    detailed_results: Optional[list] = None
    solver_status: str = "unknown"
    error_message: Optional[str] = None
    
    # Optional fields for specific optimizers
    company_volume_used: Optional[float] = None
    company_volume_limit: Optional[float] = None
    constraint_violations: Optional[list] = None


@dataclass
class OptimizationInput:
    """Standardized optimization input structure."""
    data: pd.DataFrame
    supplier_constraints: Dict[str, Dict[str, Any]]
    plant_constraints: Dict[str, Dict[str, Any]]
    company_volume_limit: Optional[float] = None
    solver_options: Optional[Dict[str, Any]] = None


class BaseOptimizationEngine(ABC):
    """Abstract base class for optimization engines."""
    
    def __init__(self, engine_name: str = "base"):
        """Initialize optimization engine."""
        self.engine_name = engine_name
        self.logger = logging.getLogger(f"{__name__}.{engine_name}")
    
    @abstractmethod
    def optimize(self, optimization_input: OptimizationInput) -> OptimizationResult:
        """
        Execute optimization with given input.
        
        Args:
            optimization_input: Standardized optimization input
            
        Returns:
            OptimizationResult with standardized structure
        """
        pass
    
    def validate_input(self, optimization_input: OptimizationInput) -> tuple[bool, list]:
        """
        Validate optimization input.
        
        Args:
            optimization_input: Input to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check data
        if optimization_input.data is None or optimization_input.data.empty:
            errors.append("Data is empty or None")
        
        # Check required columns
        required_columns = [
            'Plant_Product_Location_ID', 'Supplier', '2024 Volume (lbs)', 
            'DDP (USD)', 'Is_Baseline_Supplier'
        ]
        
        if optimization_input.data is not None:
            missing_columns = [col for col in required_columns 
                             if col not in optimization_input.data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # Check constraints
        if not optimization_input.supplier_constraints:
            errors.append("No supplier constraints provided")
        
        return len(errors) == 0, errors
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for optimization.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed data ready for optimization
        """
        df = data.copy()
        
        # Clean numerical data
        df = df[df['2024 Volume (lbs)'] > 0]
        df = df[df['DDP (USD)'] > 0]
        
        # Create unique indices for optimization
        if 'location_supplier_idx' not in df.columns:
            df['location_supplier_idx'] = range(len(df))
        
        return df
    
    def calculate_baseline_cost(self, data: pd.DataFrame) -> float:
        """Calculate baseline cost from data."""
        baseline_data = data[data['Is_Baseline_Supplier'] == 1].copy()
        baseline_total_cost = 0
        
        for _, row in baseline_data.iterrows():
            baseline_allocated_volume = row.get('Baseline Allocated Volume', 0)
            if baseline_allocated_volume > 0:
                baseline_total_cost += row.get('Baseline Price Paid', 
                                             baseline_allocated_volume * row['DDP (USD)'])
            else:
                baseline_total_cost += row['2024 Volume (lbs)'] * row['DDP (USD)']
        
        return baseline_total_cost
    
    def create_failure_result(self, error_message: str = "Optimization failed") -> OptimizationResult:
        """Create a failure result."""
        return OptimizationResult(
            success=False,
            total_savings=0,
            savings_percentage=0,
            baseline_total_cost=0,
            optimized_total_cost=0,
            execution_time=0,
            solver_status="failed",
            error_message=error_message
        )
    
    def log_optimization_start(self, optimization_input: OptimizationInput):
        """Log optimization start."""
        self.logger.info(f"Starting {self.engine_name} optimization...")
        self.logger.info(f"Data rows: {len(optimization_input.data)}")
        self.logger.info(f"Suppliers: {len(optimization_input.supplier_constraints)}")
        self.logger.info(f"Plants: {len(optimization_input.plant_constraints)}")
    
    def log_optimization_complete(self, result: OptimizationResult):
        """Log optimization completion."""
        if result.success:
            self.logger.info(f"{self.engine_name} optimization completed successfully")
            self.logger.info(f"Total savings: ${result.total_savings:,.2f}")
            self.logger.info(f"Execution time: {result.execution_time:.2f} seconds")
        else:
            self.logger.error(f"{self.engine_name} optimization failed: {result.error_message}")


class OptimizationEngineFactory:
    """Factory for creating optimization engines."""
    
    _engines = {}
    
    @classmethod
    def register_engine(cls, name: str, engine_class):
        """Register an optimization engine."""
        cls._engines[name] = engine_class
    
    @classmethod
    def create_engine(cls, name: str, **kwargs) -> BaseOptimizationEngine:
        """Create an optimization engine by name."""
        if name not in cls._engines:
            raise ValueError(f"Unknown optimization engine: {name}")
        
        return cls._engines[name](**kwargs)
    
    @classmethod
    def get_available_engines(cls) -> list:
        """Get list of available engine names."""
        return list(cls._engines.keys())