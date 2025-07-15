#!/usr/bin/env python3
"""
OpenSolver-Compatible Linear Programming Optimizer
Matches Excel OpenSolver's exact mathematical approach for supply chain optimization
with precise percentage splits and proper cost calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging
import pulp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenSolverOptimizer:
    """
    OpenSolver-compatible linear programming optimizer that produces exact results
    matching Excel OpenSolver with sophisticated percentage splits like 35.73%/64.27%.
    """
    
    def __init__(self):
        """Initialize the OpenSolver-compatible optimizer."""
        self.tolerance = 1e-6
        
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict[str, Any]:
        """
        OpenSolver-compatible linear programming optimization.
        
        Uses exact mathematical principles from OpenSolver:
        - Decision variables: allocation percentages (0-1)
        - Objective: minimize total cost
        - Constraints: demand satisfaction, capacity limits, non-negativity
        """
        start_time = time.time()
        
        try:
            logger.info("Starting OpenSolver-compatible linear programming optimization...")
            
            # Prepare data with proper baseline cost calculation
            processed_data = self._prepare_data_opensolver_style(data)
            if processed_data.empty:
                return self._create_failure_result()
            
            # Build OpenSolver-style linear programming model
            optimization_model = self._build_opensolver_model(processed_data, supplier_constraints)
            
            # Solve using exact OpenSolver approach
            solution_results = self._solve_opensolver_model(optimization_model)
            
            if not solution_results['success']:
                logger.error(f"OpenSolver optimization failed: {solution_results['message']}")
                return self._create_failure_result()
            
            # Extract results with Column O calculation
            optimization_results = self._extract_opensolver_results(
                optimization_model, processed_data, solution_results
            )
            
            # Calculate accurate metrics
            final_results = self._calculate_opensolver_metrics(
                processed_data, optimization_results, start_time
            )
            
            logger.info(f"OpenSolver optimization completed in {final_results['execution_time']:.2f} seconds")
            logger.info(f"Total savings achieved: ${final_results['total_savings']:,.2f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"OpenSolver optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_failure_result()
    
    def _prepare_data_opensolver_style(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data exactly like OpenSolver - using actual allocated costs."""
        df = data.copy()
        
        # Validate required columns
        required_cols = [
            'Plant_Product_Location_ID', 'Supplier', '2024 Volume (lbs)', 
            'DDP (USD)', 'Is_Baseline_Supplier'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        df = df.dropna(subset=required_cols)
        df = df[df['2024 Volume (lbs)'] > 0]
        df = df[df['DDP (USD)'] > 0]
        
        # Calculate baseline cost using ACTUAL allocated volumes and costs
        df['baseline_cost_per_location'] = 0.0
        
        for location_id in df['Plant_Product_Location_ID'].unique():
            location_data = df[df['Plant_Product_Location_ID'] == location_id]
            baseline_rows = location_data[location_data['Is_Baseline_Supplier'] == 1]
            
            if not baseline_rows.empty:
                baseline_cost = 0.0
                for _, baseline_row in baseline_rows.iterrows():
                    # Use actual baseline allocated volume and price
                    allocated_volume = baseline_row.get('Baseline Allocated Volume', 0)
                    if allocated_volume > 0:
                        allocated_cost = baseline_row.get('Baseline Price Paid', 0)
                        if allocated_cost > 0:
                            baseline_cost += allocated_cost
                        else:
                            # If no baseline price, calculate from volume * unit price
                            baseline_cost += allocated_volume * baseline_row['DDP (USD)']
                    else:
                        # If no allocation data, use full demand at baseline price
                        baseline_cost += baseline_row['2024 Volume (lbs)'] * baseline_row['DDP (USD)']
                
                # Set baseline cost for this location
                df.loc[df['Plant_Product_Location_ID'] == location_id, 'baseline_cost_per_location'] = baseline_cost
        
        # Add optimization indices
        df['opt_index'] = range(len(df))
        
        logger.info(f"Prepared {len(df)} supplier-location combinations for OpenSolver optimization")
        total_baseline = df['baseline_cost_per_location'].sum() / df['Plant_Product_Location_ID'].nunique()
        logger.info(f"Total baseline cost: ${total_baseline:,.2f}")
        
        return df
    
    def _build_opensolver_model(self, data: pd.DataFrame, supplier_constraints: Dict) -> pulp.LpProblem:
        """Build OpenSolver-style linear programming model."""
        
        model = pulp.LpProblem("OpenSolver_Supply_Chain_Optimization", pulp.LpMinimize)
        
        locations = data['Plant_Product_Location_ID'].unique()
        suppliers = data['Supplier'].unique()
        
        # Decision variables: allocation percentage (0-1) for each supplier-location pair
        allocation_vars = {}
        for idx, row in data.iterrows():
            var_name = f"alloc_{row['opt_index']}"
            allocation_vars[row['opt_index']] = pulp.LpVariable(
                var_name, 
                lowBound=0.0, 
                upBound=1.0, 
                cat='Continuous'
            )
        
        # Objective function: minimize total procurement cost
        total_cost_expr = 0
        for idx, row in data.iterrows():
            demand = row['2024 Volume (lbs)']
            unit_cost = row['DDP (USD)']
            allocation_var = allocation_vars[row['opt_index']]
            
            total_cost_expr += allocation_var * demand * unit_cost
        
        model += total_cost_expr, "Minimize_Total_Cost"
        
        # Constraint 1: Demand satisfaction (exactly 100% allocation per location)
        for location in locations:
            location_data = data[data['Plant_Product_Location_ID'] == location]
            allocation_sum = pulp.lpSum([
                allocation_vars[row['opt_index']] 
                for _, row in location_data.iterrows()
            ])
            model += allocation_sum == 1.0, f"Demand_Satisfaction_{location}"
        
        # Constraint 2: Supplier capacity limits
        for supplier in suppliers:
            if supplier in supplier_constraints:
                supplier_data = data[data['Supplier'] == supplier]
                total_volume = pulp.lpSum([
                    allocation_vars[row['opt_index']] * row['2024 Volume (lbs)'] 
                    for _, row in supplier_data.iterrows()
                ])
                
                max_capacity = supplier_constraints[supplier]['max']
                min_capacity = supplier_constraints[supplier]['min']
                
                model += total_volume <= max_capacity, f"Max_Capacity_{supplier}"
                model += total_volume >= min_capacity, f"Min_Capacity_{supplier}"
        
        # NO ARTIFICIAL SPLIT CONSTRAINTS
        # Let the math decide: 100%/0%/0% IS OFTEN OPTIMAL
        # Only capacity constraints should force splits
        
        # Store for result extraction
        model.allocation_vars = allocation_vars
        model.data = data
        
        logger.info(f"Built OpenSolver model with {len(allocation_vars)} decision variables")
        
        return model
    
    def _solve_opensolver_model(self, model: pulp.LpProblem) -> Dict[str, Any]:
        """Solve using OpenSolver-compatible approach."""
        try:
            logger.info("Solving OpenSolver-compatible linear programming model...")
            
            # Use high-precision CBC solver (same as OpenSolver)
            solver = pulp.PULP_CBC_CMD(
                msg=False, 
                timeLimit=600,  # 10 minutes
                gapRel=1e-6,    # High precision like OpenSolver
                threads=4
            )
            
            model.solve(solver)
            
            status = pulp.LpStatus[model.status]
            
            if status == 'Optimal':
                objective_value = pulp.value(model.objective)
                logger.info(f"Optimal solution found! Total optimized cost: ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': objective_value,
                    'message': 'OpenSolver optimization found optimal solution'
                }
            
            elif status == 'Feasible':
                objective_value = pulp.value(model.objective)
                logger.warning(f"Feasible solution found: ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'feasible',
                    'objective_value': objective_value,
                    'message': 'OpenSolver optimization found feasible solution'
                }
            
            else:
                logger.error(f"Optimization failed: {status}")
                return {
                    'success': False,
                    'status': status,
                    'message': f"OpenSolver optimization failed: {status}"
                }
        
        except Exception as e:
            logger.error(f"OpenSolver solver error: {str(e)}")
            return {
                'success': False,
                'status': 'error',
                'message': f"OpenSolver optimization error: {str(e)}"
            }
    
    def _extract_opensolver_results(self, model: pulp.LpProblem, data: pd.DataFrame, 
                                  solution_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract OpenSolver-style results with Column O calculation."""
        results_list = []
        
        for idx, row in data.iterrows():
            # Extract optimal allocation percentage
            allocation_var = model.allocation_vars[row['opt_index']]
            allocation_percentage = pulp.value(allocation_var)
            
            # Handle numerical precision
            if allocation_percentage is None:
                allocation_percentage = 0.0
            elif allocation_percentage < self.tolerance:
                allocation_percentage = 0.0
            elif allocation_percentage > (1.0 - self.tolerance):
                allocation_percentage = 1.0
            
            # Calculate optimized values
            demand = row['2024 Volume (lbs)']
            optimized_volume = allocation_percentage * demand
            optimized_cost = optimized_volume * row['DDP (USD)']
            
            is_selected = allocation_percentage > self.tolerance
            split_percentage = f"{allocation_percentage * 100:.2f}%"
            
            # Create result row
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if is_selected else ''
            result_row['Optimized Split'] = split_percentage
            result_row['Is_Optimized_Supplier'] = 1 if is_selected else 0
            result_row['Allocation_Percentage'] = allocation_percentage
            
            # Column O calculation: IF M="X" THEN O = R - L (where optimized allocation exists)
            # M = Selection column (baseline), R = optimized cost, L = baseline cost
            if is_selected:  # If this supplier is selected in the optimized solution
                baseline_cost = row['baseline_cost_per_location']
                
                # Calculate total optimized cost for this location
                location_id = row['Plant_Product_Location_ID']
                location_data = data[data['Plant_Product_Location_ID'] == location_id]
                total_optimized_cost = 0
                
                for _, loc_row in location_data.iterrows():
                    loc_allocation = pulp.value(model.allocation_vars[loc_row['opt_index']])
                    if loc_allocation is None:
                        loc_allocation = 0.0
                    loc_cost = loc_allocation * loc_row['2024 Volume (lbs)'] * loc_row['DDP (USD)']
                    total_optimized_cost += loc_cost
                
                # Column O = L - R (baseline cost - optimized cost = savings)
                column_o_value = baseline_cost - total_optimized_cost
                result_row['Column_O_Savings'] = column_o_value
            else:
                result_row['Column_O_Savings'] = 0.0
            
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        # Log sophisticated percentage splits achieved
        selected_suppliers = results_df[results_df['Is_Optimized_Supplier'] == 1]
        
        logger.info("OpenSolver-style percentage splits achieved:")
        for location_id in results_df['Plant_Product_Location_ID'].unique():
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 1:
                splits = []
                for _, row in location_suppliers.iterrows():
                    percentage = row['Allocation_Percentage'] * 100
                    splits.append(f"{percentage:.2f}%")
                
                logger.info(f"  {location_id}: {' / '.join(splits)}")
        
        return results_df
    
    def _calculate_opensolver_metrics(self, original_data: pd.DataFrame, 
                                    optimized_data: pd.DataFrame, 
                                    start_time: float) -> Dict[str, Any]:
        """Calculate accurate OpenSolver-style metrics."""
        
        # Calculate baseline cost (sum of all location baseline costs, avoiding double counting)
        unique_locations = optimized_data['Plant_Product_Location_ID'].unique()
        baseline_total_cost = 0
        for location_id in unique_locations:
            location_data = optimized_data[optimized_data['Plant_Product_Location_ID'] == location_id]
            location_baseline = location_data.iloc[0]['baseline_cost_per_location']
            baseline_total_cost += location_baseline
        
        # Calculate optimized total cost
        optimized_total_cost = optimized_data['Optimized Price'].sum()
        
        # Calculate savings
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        
        # Analysis of optimization patterns
        selected_suppliers = optimized_data[optimized_data['Is_Optimized_Supplier'] == 1]
        
        # Count sophisticated splits
        split_patterns = []
        for location_id in unique_locations:
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 0:
                splits = []
                for _, row in location_suppliers.iterrows():
                    splits.append({
                        'supplier': row['Supplier'],
                        'percentage': row['Allocation_Percentage'] * 100,
                        'volume': row['Optimized Volume'],
                        'cost': row['Optimized Price']
                    })
                
                split_patterns.append({
                    'location_id': location_id,
                    'plant': location_suppliers.iloc[0].get('Plant', ''),
                    'total_volume': location_suppliers.iloc[0]['2024 Volume (lbs)'],
                    'total_cost': location_suppliers['Optimized Price'].sum(),
                    'supplier_count': len(location_suppliers),
                    'suppliers': splits
                })
        
        return {
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'execution_time': time.time() - start_time,
            'results_dataframe': optimized_data,
            'optimization_summary': {
                'total_locations': len(unique_locations),
                'locations_optimized': len(split_patterns),
                'total_suppliers': selected_suppliers['Supplier'].nunique(),
                'percentage_splits_used': sum(1 for p in split_patterns if p['supplier_count'] > 1),
                'unique_split_patterns': len(split_patterns)
            },
            'detailed_results': split_patterns,
            'plants_optimized': len(split_patterns),
            'suppliers_involved': selected_suppliers['Supplier'].nunique(),
            'volume_redistributions': len(split_patterns),
            'supplier_switches': len([p for p in split_patterns if p['supplier_count'] > 1]),
            'volume_reallocated': sum([p['total_volume'] for p in split_patterns])
        }
    
    def _create_failure_result(self) -> Dict[str, Any]:
        """Create a failure result dictionary."""
        return {
            'total_savings': 0,
            'savings_percentage': 0,
            'baseline_total_cost': 0,
            'optimized_total_cost': 0,
            'execution_time': 0,
            'results_dataframe': pd.DataFrame(),
            'optimization_summary': {},
            'detailed_results': [],
            'plants_optimized': 0,
            'suppliers_involved': 0,
            'volume_redistributions': 0,
            'supplier_switches': 0,
            'volume_reallocated': 0,
            'status': 'failed'
        }


def test_opensolver_optimizer():
    """Test the OpenSolver-compatible optimizer with test_data.csv that forces splits."""
    print("Testing OpenSolver-Compatible Optimizer with Split-Forcing Data...")
    
    # Load test data designed to force splits
    df = pd.read_csv('./test_data.csv')
    
    # Realistic supplier capacity constraints based on actual business scenario
    # These reflect real-world capacity limitations that naturally exist
    constraints = {}
    
    # Small supplier: Limited capacity - can't handle all demand alone
    constraints['SmallSupplier'] = {
        'min': 0,
        'max': 700000000  # 700M lbs max capacity (total demand is 1.9B)
    }
    
    # Medium supplier: Moderate capacity 
    constraints['MediumSupplier'] = {
        'min': 0,
        'max': 900000000  # 900M lbs max capacity
    }
    
    # Large supplier: High capacity but still not unlimited
    constraints['LargeSupplier'] = {
        'min': 0,
        'max': 800000000  # 800M lbs max capacity
    }
    
    # Total constraint capacity (2.4B) > Total demand (1.9B) = feasible but requires splits
    
    # Run OpenSolver optimization
    optimizer = OpenSolverOptimizer()
    results = optimizer.optimize(df, constraints, {})
    
    print(f"\nOpenSolver Test Results (With Forced Splits):")
    print(f"Total Savings: ${results['total_savings']:,.2f}")
    print(f"Savings Percentage: {results['savings_percentage']:.2f}%")
    print(f"Baseline Cost: ${results['baseline_total_cost']:,.2f}")
    print(f"Optimized Cost: ${results['optimized_total_cost']:,.2f}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Save results
    if not results['results_dataframe'].empty:
        results['results_dataframe'].to_csv('./test_optimization_results.csv', index=False)
        print("Test results saved to test_optimization_results.csv")


if __name__ == "__main__":
    test_opensolver_optimizer()