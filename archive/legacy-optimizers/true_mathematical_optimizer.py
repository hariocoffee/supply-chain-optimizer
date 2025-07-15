#!/usr/bin/env python3
"""
True Mathematical Optimization Algorithm
Implements proper linear programming optimization exactly like OpenSolver
to find optimal allocations to the exact penny with flexible percentage splits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging
from scipy.optimize import linprog
import pulp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueMathematicalOptimizer:
    """
    True mathematical optimization using linear programming.
    Finds exact optimal allocations that minimize total cost, 
    producing precise percentage splits like 35.73%/64.27% or 30%/55.89%/14.11%
    """
    
    def __init__(self):
        """Initialize the true mathematical optimizer."""
        self.tolerance = 1e-8  # High precision for exact optimization
        self.solver = 'PULP_CBC_CMD'  # Use CBC solver for linear programming
        
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict[str, Any]:
        """
        True mathematical optimization using linear programming.
        
        This implements the exact same approach as OpenSolver:
        - Decision variables: allocation percentages for each supplier-location pair
        - Objective function: minimize total procurement cost
        - Constraints: demand satisfaction, supplier capacity, non-negativity
        
        Results in precise optimal allocations like 35.73%/64.27% or 30%/55.89%/14.11%
        """
        start_time = time.time()
        
        try:
            logger.info("Starting true mathematical optimization (Linear Programming)...")
            
            # Prepare optimization data
            processed_data = self._prepare_optimization_data(data)
            if processed_data.empty:
                return self._create_failure_result()
            
            # Build mathematical model using linear programming
            optimization_model = self._build_linear_programming_model(processed_data, supplier_constraints)
            
            # Solve using mathematical optimization
            solution_results = self._solve_mathematical_model(optimization_model, processed_data)
            
            if not solution_results['success']:
                logger.error(f"Mathematical optimization failed: {solution_results['message']}")
                return self._create_failure_result()
            
            # Extract precise optimization results
            optimization_results = self._extract_precise_results(
                optimization_model, processed_data, solution_results
            )
            
            # Calculate comprehensive metrics with cost savings
            final_results = self._calculate_comprehensive_metrics(
                processed_data, optimization_results, start_time
            )
            
            logger.info(f"Mathematical optimization completed in {final_results['execution_time']:.2f} seconds")
            logger.info(f"Total savings achieved: ${final_results['total_savings']:,.2f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Mathematical optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_failure_result()
    
    def _prepare_optimization_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for mathematical optimization."""
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
        
        # Clean and validate data
        df = df.dropna(subset=required_cols)
        df = df[df['2024 Volume (lbs)'] > 0]
        df = df[df['DDP (USD)'] > 0]
        
        # Calculate baseline cost for each location
        df['baseline_cost'] = 0.0
        for location_id in df['Plant_Product_Location_ID'].unique():
            location_data = df[df['Plant_Product_Location_ID'] == location_id]
            baseline_row = location_data[location_data['Is_Baseline_Supplier'] == 1]
            
            if not baseline_row.empty:
                baseline_cost = baseline_row.iloc[0].get('Baseline Price Paid', 0)
                if baseline_cost == 0:
                    baseline_cost = baseline_row.iloc[0]['2024 Volume (lbs)'] * baseline_row.iloc[0]['DDP (USD)']
                
                df.loc[df['Plant_Product_Location_ID'] == location_id, 'baseline_cost'] = baseline_cost
        
        # Create unique indices for optimization variables
        df['optimization_index'] = range(len(df))
        
        logger.info(f"Prepared {len(df)} supplier-location combinations for mathematical optimization")
        logger.info(f"Optimizing across {df['Plant_Product_Location_ID'].nunique()} locations with {df['Supplier'].nunique()} suppliers")
        
        return df
    
    def _build_linear_programming_model(self, data: pd.DataFrame, supplier_constraints: Dict) -> pulp.LpProblem:
        """
        Build linear programming model exactly like OpenSolver.
        
        Decision Variables: x[i] = allocation percentage for supplier-location pair i
        Objective: Minimize Σ(x[i] * demand[location] * cost_per_unit[i])
        Constraints:
        1. Demand satisfaction: Σ(x[i] for same location) = 1.0
        2. Supplier capacity: Σ(x[i] * demand[location] for same supplier) ≤ supplier_max
        3. Non-negativity: x[i] ≥ 0
        4. Upper bound: x[i] ≤ 1.0
        """
        
        # Create linear programming problem
        model = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)
        
        # Get unique locations and suppliers
        locations = data['Plant_Product_Location_ID'].unique()
        suppliers = data['Supplier'].unique()
        
        # Create decision variables: allocation percentage for each supplier-location pair
        allocation_vars = {}
        for idx, row in data.iterrows():
            var_name = f"alloc_{row['optimization_index']}"
            allocation_vars[row['optimization_index']] = pulp.LpVariable(
                var_name, 
                lowBound=0.0, 
                upBound=1.0, 
                cat='Continuous'
            )
        
        # Objective Function: Minimize total cost
        total_cost = 0
        for idx, row in data.iterrows():
            demand = row['2024 Volume (lbs)']
            cost_per_unit = row['DDP (USD)']
            allocation_var = allocation_vars[row['optimization_index']]
            
            total_cost += allocation_var * demand * cost_per_unit
        
        model += total_cost, "Total_Procurement_Cost"
        
        # Constraint 1: Demand satisfaction - each location must have exactly 100% allocation
        for location in locations:
            location_data = data[data['Plant_Product_Location_ID'] == location]
            demand_constraint = 0
            
            for idx, row in location_data.iterrows():
                demand_constraint += allocation_vars[row['optimization_index']]
            
            model += demand_constraint == 1.0, f"Demand_Satisfaction_{location}"
        
        # Constraint 2: Supplier capacity limits
        for supplier in suppliers:
            if supplier in supplier_constraints:
                supplier_data = data[data['Supplier'] == supplier]
                supplier_total_volume = 0
                
                for idx, row in supplier_data.iterrows():
                    demand = row['2024 Volume (lbs)']
                    allocation_var = allocation_vars[row['optimization_index']]
                    supplier_total_volume += allocation_var * demand
                
                max_capacity = supplier_constraints[supplier]['max']
                min_capacity = supplier_constraints[supplier]['min']
                
                # Maximum capacity constraint
                model += supplier_total_volume <= max_capacity, f"Supplier_Max_Capacity_{supplier}"
                
                # Minimum capacity constraint (if supplier is used at all)
                model += supplier_total_volume >= min_capacity, f"Supplier_Min_Capacity_{supplier}"
        
        # Store variables in model for later access
        model.allocation_vars = allocation_vars
        model.data = data
        
        logger.info(f"Built linear programming model with {len(allocation_vars)} decision variables")
        logger.info(f"Model has {len(locations)} demand constraints and {len(suppliers)} capacity constraints")
        
        return model
    
    def _solve_mathematical_model(self, model: pulp.LpProblem, data: pd.DataFrame) -> Dict[str, Any]:
        """Solve the linear programming model using mathematical optimization."""
        try:
            logger.info("Solving linear programming model...")
            
            # Solve using CBC solver (same as OpenSolver)
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300, gapRel=1e-6)
            model.solve(solver)
            
            # Check solution status
            status = pulp.LpStatus[model.status]
            
            if status == 'Optimal':
                objective_value = pulp.value(model.objective)
                logger.info(f"Optimal solution found! Objective value: ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': objective_value,
                    'message': 'Mathematical optimization found optimal solution'
                }
            
            elif status == 'Feasible':
                objective_value = pulp.value(model.objective)
                logger.warning(f"Feasible solution found (may not be optimal): ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'feasible', 
                    'objective_value': objective_value,
                    'message': 'Mathematical optimization found feasible solution'
                }
            
            else:
                logger.error(f"Optimization failed with status: {status}")
                return {
                    'success': False,
                    'status': status,
                    'message': f"Linear programming solver failed: {status}"
                }
        
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            return {
                'success': False,
                'status': 'error',
                'message': f"Mathematical optimization error: {str(e)}"
            }
    
    def _extract_precise_results(self, model: pulp.LpProblem, data: pd.DataFrame, 
                               solution_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract precise optimization results with exact percentage allocations."""
        results_list = []
        
        for idx, row in data.iterrows():
            # Get optimal allocation percentage
            allocation_var = model.allocation_vars[row['optimization_index']]
            allocation_percentage = pulp.value(allocation_var)
            
            # Handle numerical precision
            if allocation_percentage is None:
                allocation_percentage = 0.0
            elif allocation_percentage < self.tolerance:
                allocation_percentage = 0.0
            elif allocation_percentage > (1.0 - self.tolerance):
                allocation_percentage = 1.0
            
            # Calculate optimized volumes and costs
            demand = row['2024 Volume (lbs)']
            optimized_volume = allocation_percentage * demand
            optimized_cost = optimized_volume * row['DDP (USD)']
            
            # Determine if supplier is selected (allocation > tolerance)
            is_selected = allocation_percentage > self.tolerance
            
            # Format percentage split with high precision
            split_percentage = f"{allocation_percentage * 100:.2f}%"
            
            # Create result row
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if is_selected else ''
            result_row['Optimized Split'] = split_percentage
            result_row['Is_Optimized_Supplier'] = 1 if is_selected else 0
            result_row['Allocation_Percentage'] = allocation_percentage
            
            # Calculate cost savings (Column O) - if baseline supplier was selected, show savings
            if row['Is_Baseline_Supplier'] == 1 and is_selected:
                baseline_cost = row.get('Baseline Price Paid', demand * row['DDP (USD)'])
                cost_savings = baseline_cost - optimized_cost
                result_row['Cost_Savings'] = cost_savings
            else:
                result_row['Cost_Savings'] = 0.0
            
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        # Log optimization statistics
        selected_suppliers = results_df[results_df['Is_Optimized_Supplier'] == 1]
        unique_splits = set()
        
        for location_id in results_df['Plant_Product_Location_ID'].unique():
            location_data = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            if len(location_data) > 1:
                split_pattern = tuple(sorted([f"{row['Allocation_Percentage']*100:.2f}%" 
                                            for _, row in location_data.iterrows()]))
                unique_splits.add(split_pattern)
        
        logger.info(f"Mathematical optimization produced {len(unique_splits)} unique percentage split patterns")
        logger.info(f"Examples of precise splits found:")
        for i, split in enumerate(list(unique_splits)[:5]):
            logger.info(f"  Pattern {i+1}: {' / '.join(split)}")
        
        return results_df
    
    def _calculate_comprehensive_metrics(self, original_data: pd.DataFrame, 
                                       optimized_data: pd.DataFrame, 
                                       start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive optimization metrics and savings."""
        
        # Calculate baseline costs
        baseline_total_cost = 0
        for location_id in optimized_data['Plant_Product_Location_ID'].unique():
            location_data = optimized_data[optimized_data['Plant_Product_Location_ID'] == location_id]
            baseline_cost = location_data.iloc[0]['baseline_cost']
            baseline_total_cost += baseline_cost
        
        # Calculate optimized costs
        optimized_total_cost = optimized_data['Optimized Price'].sum()
        
        # Calculate savings
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        
        # Calculate detailed metrics
        selected_suppliers = optimized_data[optimized_data['Is_Optimized_Supplier'] == 1]
        
        # Count allocation types
        percentage_splits = selected_suppliers['Allocation_Percentage'].tolist()
        full_allocations = sum(1 for p in percentage_splits if abs(p - 1.0) < self.tolerance)
        partial_allocations = sum(1 for p in percentage_splits if self.tolerance < p < (1.0 - self.tolerance))
        
        # Analyze percentage split patterns
        split_patterns = []
        for location_id in selected_suppliers['Plant_Product_Location_ID'].unique():
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 1:
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
                    'plant': location_suppliers.iloc[0]['Plant'],
                    'total_volume': location_suppliers.iloc[0]['2024 Volume (lbs)'],
                    'total_cost': location_suppliers['Optimized Price'].sum(),
                    'supplier_count': len(location_suppliers),
                    'suppliers': splits
                })
        
        # Count supplier switches and reallocations
        supplier_switches = 0
        volume_reallocated = 0
        
        for location_id in optimized_data['Plant_Product_Location_ID'].unique():
            location_data = optimized_data[optimized_data['Plant_Product_Location_ID'] == location_id]
            
            baseline_supplier = location_data[location_data['Is_Baseline_Supplier'] == 1]
            optimized_suppliers = location_data[location_data['Is_Optimized_Supplier'] == 1]
            
            if not baseline_supplier.empty and not optimized_suppliers.empty:
                baseline_sup_name = baseline_supplier.iloc[0]['Supplier']
                optimized_sup_names = set(optimized_suppliers['Supplier'].tolist())
                
                # Check for changes in supplier allocation
                baseline_in_optimized = baseline_sup_name in optimized_sup_names
                multiple_suppliers = len(optimized_sup_names) > 1
                
                if not baseline_in_optimized or multiple_suppliers:
                    supplier_switches += 1
                    volume_reallocated += location_data.iloc[0]['2024 Volume (lbs)']
        
        return {
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'execution_time': time.time() - start_time,
            'results_dataframe': optimized_data,
            'optimization_summary': {
                'total_locations': optimized_data['Plant_Product_Location_ID'].nunique(),
                'locations_optimized': len(split_patterns) + supplier_switches,
                'total_suppliers': selected_suppliers['Supplier'].nunique(),
                'supplier_switches': supplier_switches,
                'volume_reallocated': volume_reallocated,
                'percentage_splits_used': partial_allocations,
                'full_allocations': full_allocations,
                'unique_split_patterns': len(split_patterns)
            },
            'detailed_results': split_patterns,
            'plants_optimized': len(split_patterns) + supplier_switches,
            'suppliers_involved': selected_suppliers['Supplier'].nunique(),
            'volume_redistributions': len(split_patterns),
            'supplier_switches': supplier_switches,
            'volume_reallocated': volume_reallocated
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


def test_true_mathematical_optimizer():
    """Test the true mathematical optimizer with real data."""
    print("Testing True Mathematical Optimizer (Linear Programming)...")
    
    # Load real data
    df = pd.read_csv('./sco_data_cleaned.csv')
    
    # Create realistic constraints
    constraints = {}
    supplier_data = df.groupby('Supplier').agg({
        'Baseline Allocated Volume': 'sum',
        '2024 Volume (lbs)': 'sum'
    }).reset_index()
    
    total_volume = df['2024 Volume (lbs)'].sum()
    
    for _, row in supplier_data.iterrows():
        supplier = row['Supplier']
        baseline_volume = row['Baseline Allocated Volume']
        
        constraints[supplier] = {
            'min': 0,  # Allow zero allocation
            'max': total_volume,  # Allow full volume if most cost-effective
            'baseline_volume': baseline_volume,
            'baseline_cost': 0
        }
    
    # Run true mathematical optimization
    optimizer = TrueMathematicalOptimizer()
    results = optimizer.optimize(df, constraints, {})
    
    print(f"\nTrue Mathematical Optimization Results:")
    print(f"Total Savings: ${results['total_savings']:,.2f}")
    print(f"Savings Percentage: {results['savings_percentage']:.2f}%")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Percentage Splits Used: {results['optimization_summary']['percentage_splits_used']}")
    print(f"Unique Split Patterns: {results['optimization_summary']['unique_split_patterns']}")
    
    # Show example precise splits
    if results['detailed_results']:
        print(f"\nExamples of Precise Mathematical Splits:")
        for i, result in enumerate(results['detailed_results'][:5]):
            splits = [f"{s['supplier']}: {s['percentage']:.2f}%" for s in result['suppliers']]
            print(f"  {i+1}. {result['location_id']}: {' / '.join(splits)}")
    
    # Save results
    results['results_dataframe'].to_csv('./true_mathematical_optimization_results.csv', index=False)
    print("Results saved to true_mathematical_optimization_results.csv")


if __name__ == "__main__":
    test_true_mathematical_optimizer()