"""
Data Processing Service for Supply Chain Optimization
Handles data validation, preparation, and transformation.
"""

import pandas as pd
import numpy as np
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    processed_data: Optional[pd.DataFrame] = None


@dataclass
class ConstraintAnalysis:
    """Analysis of supplier constraints."""
    supplier_constraints: Dict[str, Dict[str, Any]]
    total_volume: float
    is_demo_file: bool
    file_type: str


class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self):
        """Initialize data processor."""
        self.required_columns = settings.get_data_columns()
    
    def generate_file_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash for the uploaded data for caching purposes."""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def validate_data(self, data: pd.DataFrame, filename: str = "") -> ValidationResult:
        """Validate uploaded data for optimization requirements."""
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append("Data file is empty")
            return ValidationResult(False, errors, warnings)
        
        # Check for required columns based on file type
        if settings.is_demo_file(filename):
            required_cols = self.required_columns['required_optimization']
        else:
            required_cols = self.required_columns['required_baseline']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate data types and values
        if '2024 Volume (lbs)' in data.columns:
            if data['2024 Volume (lbs)'].isnull().any():
                warnings.append("Some volume values are missing")
            if (data['2024 Volume (lbs)'] <= 0).any():
                warnings.append("Some volume values are zero or negative")
        
        if 'DDP (USD)' in data.columns:
            if data['DDP (USD)'].isnull().any():
                warnings.append("Some price values are missing")
            if (data['DDP (USD)'] <= 0).any():
                warnings.append("Some price values are zero or negative")
        
        # Check for duplicate supplier-location combinations
        if all(col in data.columns for col in ['Supplier', 'Plant_Product_Location_ID']):
            duplicates = data.duplicated(subset=['Supplier', 'Plant_Product_Location_ID'])
            if duplicates.any():
                warnings.append(f"Found {duplicates.sum()} duplicate supplier-location combinations")
        
        # Process data if no critical errors
        processed_data = None
        if not errors:
            processed_data = self._prepare_data(data, filename)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, processed_data)
    
    def _prepare_data(self, data: pd.DataFrame, filename: str = "") -> pd.DataFrame:
        """Prepare and clean data for optimization."""
        df = data.copy()
        
        # Create Plant_Product_ID if it doesn't exist
        if 'Plant_Product_ID' not in df.columns and all(col in df.columns for col in ['Plant', 'Product']):
            df['Plant_Product_ID'] = df['Plant'] + '_' + df['Product']
        
        # Create Plant_Product_Location_ID if it doesn't exist
        if 'Plant_Product_Location_ID' not in df.columns:
            if all(col in df.columns for col in ['Plant', 'Product', 'Plant Location']):
                df['Plant_Product_Location_ID'] = (df['Plant'] + '_' + 
                                                   df['Plant Location'] + '_' + 
                                                   df['Product'])
            elif 'Plant_Product_ID' in df.columns:
                df['Plant_Product_Location_ID'] = df['Plant_Product_ID']
        
        # Add location_supplier_idx for optimization
        df['location_supplier_idx'] = range(len(df))
        
        # Clean numerical data
        if '2024 Volume (lbs)' in df.columns:
            df = df[df['2024 Volume (lbs)'] > 0]
        
        if 'DDP (USD)' in df.columns:
            df = df[df['DDP (USD)'] > 0]
        
        # Handle missing baseline data
        if 'Is_Baseline_Supplier' not in df.columns:
            # Infer from Selection column if available
            if 'Selection' in df.columns:
                df['Is_Baseline_Supplier'] = df['Selection'].apply(
                    lambda x: 1 if str(x).strip().upper() == 'X' else 0
                )
            else:
                df['Is_Baseline_Supplier'] = 0
        
        # Handle missing baseline allocation
        if 'Baseline Allocated Volume' not in df.columns:
            if 'Is_Baseline_Supplier' in df.columns:
                df['Baseline Allocated Volume'] = df.apply(
                    lambda row: row['2024 Volume (lbs)'] if row['Is_Baseline_Supplier'] == 1 else 0,
                    axis=1
                )
            else:
                df['Baseline Allocated Volume'] = 0
        
        # Handle missing baseline price
        if 'Baseline Price Paid' not in df.columns:
            df['Baseline Price Paid'] = df['Baseline Allocated Volume'] * df['DDP (USD)']
        
        logger.info(f"Prepared {len(df)} records for optimization")
        return df
    
    def analyze_constraints(self, data: pd.DataFrame, filename: str = "") -> ConstraintAnalysis:
        """Analyze data to determine supplier constraints."""
        supplier_constraints = {}
        total_volume = data['2024 Volume (lbs)'].sum() if '2024 Volume (lbs)' in data.columns else 0
        is_demo_file = settings.is_demo_file(filename)
        
        # Determine file type and constraint strategy
        if 'demo_data' in filename.lower():
            file_type = 'demo_data'
            supplier_constraints = self._get_demo_data_constraints()
        elif 'DEMO.csv' in filename:
            file_type = 'DEMO'
            supplier_constraints = self._get_demo_csv_constraints(data)
        else:
            file_type = 'standard'
            supplier_constraints = self._calculate_standard_constraints(data)
        
        return ConstraintAnalysis(
            supplier_constraints=supplier_constraints,
            total_volume=total_volume,
            is_demo_file=is_demo_file,
            file_type=file_type
        )
    
    def _get_demo_data_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined constraints for demo_data.csv files."""
        predefined_constraints = {
            'Aunt Baker': {
                'min': 190_000_000,
                'max': 208_800_000,
                'display_max': 208_800_000,
                'baseline_volume': 208_800_000,
                'baseline_cost': 0
            },
            'Aunt Bethany': {
                'min': 2_400_000_000,
                'max': 2_758_300_000,
                'display_max': 2_758_300_000,
                'baseline_volume': 2_758_300_000,
                'baseline_cost': 0
            },
            'Aunt Celine': {
                'min': 500_000_000,
                'max': 628_200_000,
                'display_max': 628_200_000,
                'baseline_volume': 628_200_000,
                'baseline_cost': 0
            },
            'Aunt Smith': {
                'min': 2_000_000_000,
                'max': 2_507_100_000,
                'display_max': 2_507_100_000,
                'baseline_volume': 2_507_100_000,
                'baseline_cost': 0
            }
        }
        return predefined_constraints
    
    def _get_demo_csv_constraints(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get dynamic constraints for DEMO.csv files."""
        supplier_constraints = {}
        
        # Validate required columns exist
        required_columns = ['Supplier', 'Baseline Allocated Volume', 'Baseline Price Paid']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns for DEMO.csv constraint calculation: {missing_columns}")
            return {}
        
        # Group by supplier and sum baseline data
        supplier_data = data.groupby('Supplier').agg({
            'Baseline Allocated Volume': 'sum',
            'Baseline Price Paid': 'sum'
        }).reset_index()
        
        # Use predefined minimums with dynamic maximums
        demo_min_constraints = settings.constraints.demo_min_constraints
        
        for _, row in supplier_data.iterrows():
            supplier = row['Supplier']
            baseline_volume = row['Baseline Allocated Volume']
            baseline_cost = row['Baseline Price Paid']
            
            min_capacity = demo_min_constraints.get(supplier, 0)
            max_capacity = baseline_volume  # Use actual data for max
            
            supplier_constraints[supplier] = {
                'min': min_capacity,
                'max': max_capacity,
                'display_max': max_capacity,
                'baseline_volume': baseline_volume,
                'baseline_cost': baseline_cost
            }
        
        return supplier_constraints
    
    def _calculate_standard_constraints(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate constraints for standard data files."""
        supplier_constraints = {}
        
        # Validate required columns
        required_columns = ['Supplier', 'Baseline Allocated Volume', 'Baseline Price Paid', '2024 Volume (lbs)']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns for constraint calculation: {missing_columns}")
            return {}
        
        # Group by supplier
        supplier_data = data.groupby('Supplier').agg({
            'Baseline Allocated Volume': 'sum',
            'Baseline Price Paid': 'sum'
        }).reset_index()
        
        total_volume = data['2024 Volume (lbs)'].sum()
        
        for _, row in supplier_data.iterrows():
            supplier = row['Supplier']
            baseline_volume = row['Baseline Allocated Volume']
            baseline_cost = row['Baseline Price Paid']
            
            # Set flexible constraints for optimization
            if baseline_volume > 0:
                max_capacity = total_volume  # Allow full reallocation
            else:
                max_capacity = total_volume  # New suppliers can compete for full volume
            
            supplier_constraints[supplier] = {
                'min': 0,  # Minimum can be 0
                'max': max_capacity,
                'display_max': baseline_volume if baseline_volume > 0 else total_volume,
                'baseline_volume': baseline_volume,
                'baseline_cost': baseline_cost
            }
        
        return supplier_constraints
    
    def create_template_data(self) -> pd.DataFrame:
        """Create template data for download."""
        template_data = {
            'Plant': ['Alpha', 'Beta', 'Charlie', 'Charlie', 'Delta', 'Delta'],
            'Product': ['Brownies', 'Brownies', 'Brownies', 'Brownies', 'Brownies', 'Brownies'],
            '2024 Volume (lbs)': [5900000, 15900000, 56800000, 56800000, 75500000, 75500000],
            'Supplier': ['Aunt Smith', 'Aunt Smith', 'Aunt Bethany', 'Aunt Smith', 'Aunt Bethany', 'Aunt Baker'],
            'Plant Location': ['Location1', 'Location2', 'Location3', 'Location3', 'Location4', 'Location4'],
            'DDP (USD)': [8.05, 6.91, 9.96, 9.28, 5.47, 7.29],
            'Baseline Allocated Volume': [5900000, 15900000, 56800000, 0, 75500000, 0],
            'Baseline Price Paid': [47495000, 109869000, 565728000, 0, 412985000, 0],
            'Selection': ['X', 'X', 'X', '', 'X', ''],
            'Split': ['100%', '100%', '100%', '0%', '100%', '0%'],
            'Optimized Volume': ['', '', '', '', '', ''],
            'Optimized Price': ['', '', '', '', '', ''],
            'Optimized Selection': ['', '', '', '', '', ''],
            'Optimized Split': ['', '', '', '', '', ''],
            'Cost Savings': ['', '', '', '', '', ''],
            'Plant_Product_Location_ID': ['Alpha_Location1_Brownies', 'Beta_Location2_Brownies', 'Charlie_Location3_Brownies', 'Charlie_Location3_Brownies', 'Delta_Location4_Brownies', 'Delta_Location4_Brownies'],
            'Supplier_Plant_Product_Location_ID': ['Alpha_Location1_Brownies_Aunt Smith', 'Beta_Location2_Brownies_Aunt Smith', 'Charlie_Location3_Brownies_Aunt Bethany', 'Charlie_Location3_Brownies_Aunt Smith', 'Delta_Location4_Brownies_Aunt Bethany', 'Delta_Location4_Brownies_Aunt Baker'],
            'Plant_Product_ID': ['Alpha_Brownies', 'Beta_Brownies', 'Charlie_Brownies', 'Charlie_Brownies', 'Delta_Brownies', 'Delta_Brownies'],
            'Supplier_Plant_Product_ID': ['Alpha_Brownies_Aunt Smith', 'Beta_Brownies_Aunt Smith', 'Charlie_Brownies_Aunt Bethany', 'Charlie_Brownies_Aunt Smith', 'Delta_Brownies_Aunt Bethany', 'Delta_Brownies_Aunt Baker'],
            'Plant_Product_Baseline_Cost': [47495000, 109869000, 565728000, 565728000, 412985000, 412985000],
            'Plant_Product_Optimized_Cost': ['', '', '', '', '', ''],
            'Plant_Product_Total_Savings': ['', '', '', '', '', ''],
            'Allocated_Cost_Savings': ['', '', '', '', '', ''],
            'Is_Baseline_Supplier': [1, 1, 1, 0, 1, 0],
            'Is_Optimized_Supplier': ['', '', '', '', '', ''],
            'Volume_Fraction': ['1.000', '1.000', '1.000', '0.000', '1.000', '0.000']
        }
        
        return pd.DataFrame(template_data)
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the data."""
        if data.empty:
            return {}
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'unique_plants': data['Plant'].nunique() if 'Plant' in data.columns else 0,
            'unique_suppliers': data['Supplier'].nunique() if 'Supplier' in data.columns else 0,
            'unique_locations': data['Plant_Product_Location_ID'].nunique() if 'Plant_Product_Location_ID' in data.columns else 0,
            'total_volume': data['2024 Volume (lbs)'].sum() if '2024 Volume (lbs)' in data.columns else 0,
            'baseline_volume': data['Baseline Allocated Volume'].sum() if 'Baseline Allocated Volume' in data.columns else 0,
            'baseline_cost': data['Baseline Price Paid'].sum() if 'Baseline Price Paid' in data.columns else 0
        }
        
        return summary