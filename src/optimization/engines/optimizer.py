#!/usr/bin/env python3
"""
Enhanced Supply Chain Optimization Algorithm
Implements advanced mathematical optimization with percentage splits and continuous variables
for maximum cost savings in supplier selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPyomoOptimizer:
    """
    Enhanced optimization engine using Pyomo with continuous variables for percentage splits.
    Implements sophisticated mathematical programming to achieve maximum cost savings.
    """
    
    def __init__(self, solver_name: str = 'ipopt'):
        """Initialize the optimizer with the specified solver."""
        # Try different solvers in order of preference
        available_solvers = ['cbc', 'glpk', 'ipopt', 'cplex', 'gurobi']
        self.solver_name = None
        
        for solver in available_solvers:
            try:
                test_solver = pyo.SolverFactory(solver)
                if test_solver.available():
                    self.solver_name = solver
                    logger.info(f"Using solver: {solver}")
                    break
            except:
                continue
        
        if not self.solver_name:
            # Fallback to a simple linear programming approach
            logger.warning("No optimization solver found, will use simplified approach")
            self.solver_name = 'simple'
        
        self.model = None
        self.results = None
        
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict[str, Any]:
        """
        Enhanced optimization with continuous percentage splits for maximum savings.
        
        Args:
            data: Supply chain data with supplier options
            supplier_constraints: Min/max volume constraints per supplier
            plant_constraints: Plant-specific constraints (unused but maintained for compatibility)
            
        Returns:
            Dictionary with optimization results including enhanced savings
        """
        start_time = time.time()
        
        try:
            logger.info("Starting enhanced optimization with percentage splits...")
            
            # Validate and prepare data
            processed_data = self._prepare_data(data)
            if processed_data.empty:
                logger.error("No valid data to optimize")
                return self._create_failure_result()
            
            # Build enhanced mathematical model
            model = self._build_enhanced_model(processed_data, supplier_constraints)
            
            # Solve the optimization problem
            solver_results = self._solve_model(model)
            
            if not solver_results['success']:
                logger.error(f"Optimization failed: {solver_results['message']}")
                return self._create_failure_result()
            
            # Extract and process results with percentage splits
            optimization_results = self._extract_enhanced_results(model, processed_data)
            
            # Calculate comprehensive savings and metrics
            final_results = self._calculate_comprehensive_metrics(
                processed_data, optimization_results, start_time
            )
            
            # Add solver status information to final results
            final_results['solver_status'] = solver_results['status']
            final_results['solver_message'] = solver_results['message']
            final_results['optimization_method'] = 'mathematical' if solver_results['status'] in ['optimal', 'feasible'] else 'greedy'
            
            logger.info(f"Enhanced optimization completed successfully in {final_results['execution_time']:.2f} seconds")
            logger.info(f"Total savings achieved: ${final_results['total_savings']:,.2f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return self._create_failure_result()
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data for enhanced optimization."""
        df = data.copy()
        
        # Required columns for optimization
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
        
        # Create unique indices for optimization
        df['location_supplier_idx'] = range(len(df))
        
        logger.info(f"Prepared {len(df)} supplier-location combinations for optimization")
        
        return df
    
    def _build_enhanced_model(self, data: pd.DataFrame, supplier_constraints: Dict) -> pyo.ConcreteModel:
        """
        Build enhanced mathematical optimization model with continuous percentage variables.
        This allows for sophisticated percentage splits that maximize cost savings.
        """
        model = pyo.ConcreteModel()
        
        # Sets
        locations = data['Plant_Product_Location_ID'].unique()
        suppliers = data['Supplier'].unique()
        supplier_location_pairs = list(zip(data['location_supplier_idx'], 
                                         data['Plant_Product_Location_ID'], 
                                         data['Supplier']))
        
        model.locations = pyo.Set(initialize=locations)
        model.suppliers = pyo.Set(initialize=suppliers)
        model.supplier_location_pairs = pyo.Set(initialize=supplier_location_pairs)
        
        # Parameters
        model.demand = pyo.Param(model.locations, initialize={
            loc: data[data['Plant_Product_Location_ID'] == loc]['2024 Volume (lbs)'].iloc[0]
            for loc in locations
        })
        
        model.cost_per_unit = pyo.Param(model.supplier_location_pairs, initialize={
            (idx, loc, sup): data.iloc[idx]['DDP (USD)']
            for idx, loc, sup in supplier_location_pairs
        })
        
        # Enhanced Decision Variables - Continuous percentage allocation (0 to 1)
        model.allocation_percentage = pyo.Var(
            model.supplier_location_pairs, 
            domain=pyo.NonNegativeReals, 
            bounds=(0, 1)
        )
        
        # Binary variables for supplier selection (for logical constraints)
        model.supplier_selected = pyo.Var(
            model.supplier_location_pairs,
            domain=pyo.Binary
        )
        
        # Objective Function - Minimize total cost
        def total_cost_rule(model):
            return sum(
                model.allocation_percentage[idx, loc, sup] * 
                model.demand[loc] * 
                model.cost_per_unit[idx, loc, sup]
                for idx, loc, sup in model.supplier_location_pairs
            )
        
        model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
        
        # Constraints
        
        # 1. Demand satisfaction constraint - Each location must receive exactly its required volume
        def demand_satisfaction_rule(model, loc):
            return sum(
                model.allocation_percentage[idx, loc, sup]
                for idx, location, sup in model.supplier_location_pairs
                if location == loc
            ) == 1.0  # Sum of percentages must equal 100%
        
        model.demand_satisfaction = pyo.Constraint(
            model.locations, 
            rule=demand_satisfaction_rule
        )
        
        # 2. Enhanced supplier capacity constraints with flexibility
        def supplier_capacity_rule(model, supplier):
            if supplier not in supplier_constraints:
                return pyo.Constraint.Skip
            
            total_volume = sum(
                model.allocation_percentage[idx, loc, sup] * model.demand[loc]
                for idx, loc, sup in model.supplier_location_pairs
                if sup == supplier
            )
            
            min_vol = supplier_constraints[supplier]['min']
            max_vol = supplier_constraints[supplier]['max']
            
            return (min_vol, total_volume, max_vol)
        
        model.supplier_capacity = pyo.Constraint(
            model.suppliers,
            rule=supplier_capacity_rule
        )
        
        # 3. Logical constraint - allocation only if supplier is selected
        def selection_logic_rule(model, idx, loc, sup):
            return model.allocation_percentage[idx, loc, sup] <= model.supplier_selected[idx, loc, sup]
        
        model.selection_logic = pyo.Constraint(
            model.supplier_location_pairs,
            rule=selection_logic_rule
        )
        
        # 4. Minimum allocation constraint - if selected, minimum meaningful allocation
        def minimum_allocation_rule(model, idx, loc, sup):
            return model.allocation_percentage[idx, loc, sup] >= 0.01 * model.supplier_selected[idx, loc, sup]
        
        model.minimum_allocation = pyo.Constraint(
            model.supplier_location_pairs,
            rule=minimum_allocation_rule
        )
        
        # 5. Enhanced diversification constraint - allow up to 3 suppliers per location for risk management
        def diversification_rule(model, loc):
            return sum(
                model.supplier_selected[idx, location, sup]
                for idx, location, sup in model.supplier_location_pairs
                if location == loc
            ) <= 3
        
        model.diversification = pyo.Constraint(
            model.locations,
            rule=diversification_rule
        )
        
        self.model = model
        logger.info(f"Enhanced mathematical model built with {len(supplier_location_pairs)} decision variables")
        
        return model
    
    def _solve_model(self, model: pyo.ConcreteModel) -> Dict[str, Any]:
        """Solve the enhanced optimization model."""
        try:
            if self.solver_name == 'simple':
                # Use fallback simple optimization approach
                return self._solve_simple_optimization(model)
            
            # Use available solver with enhanced settings for better performance
            solver = pyo.SolverFactory(self.solver_name)
            
            # Enhanced solver options for better solutions
            if self.solver_name == 'cbc':
                solver.options['seconds'] = 300  # 5 minute time limit
                solver.options['ratio'] = 0.01   # 1% optimality gap
                solver.options['threads'] = 4    # Use multiple threads
                solver.options['cuts'] = 'on'    # Enable cutting planes
                solver.options['heuristics'] = 'on'  # Enable heuristics
            elif self.solver_name == 'glpk':
                solver.options['tmlim'] = 300    # 5 minute time limit
                solver.options['mipgap'] = 0.01  # 1% optimality gap
            
            logger.info("Solving enhanced optimization model...")
            # Solve without automatic solution loading to avoid warning status issues
            results = solver.solve(model, tee=False, load_solutions=False)
            
            # Check solution status BEFORE loading solutions
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                # Manually load the optimal solution
                model.solutions.load_from(results)
                
                logger.info("Optimal solution found!")
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': pyo.value(model.total_cost),
                    'message': 'Optimization completed successfully'
                }
            
            elif results.solver.termination_condition == TerminationCondition.feasible:
                # Load the feasible solution
                model.solutions.load_from(results)
                
                logger.warning("Feasible solution found (may not be optimal)")
                return {
                    'success': True,
                    'status': 'feasible',
                    'objective_value': pyo.value(model.total_cost),
                    'message': 'Feasible solution found'
                }
            
            else:
                logger.error(f"Solver failed: {results.solver.termination_condition}")
                return self._solve_simple_optimization(model)
        
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            return self._solve_simple_optimization(model)
    
    def _solve_simple_optimization(self, model: pyo.ConcreteModel) -> Dict[str, Any]:
        """Fallback simple optimization using greedy approach."""
        try:
            logger.info("Using fallback greedy optimization approach...")
            
            # Extract data from model for greedy solution
            locations = list(model.locations)
            
            # Simple greedy approach - for each location, choose cheapest supplier
            total_cost = 0
            
            for loc in locations:
                # Find all supplier options for this location
                location_pairs = [(idx, supplier) for idx, location, supplier in model.supplier_location_pairs 
                                if location == loc]
                
                if location_pairs:
                    # Choose the cheapest supplier
                    best_pair = min(location_pairs, 
                                  key=lambda x: model.cost_per_unit[x[0], loc, x[1]])
                    
                    best_idx, best_supplier = best_pair
                    demand = pyo.value(model.demand[loc])
                    cost = pyo.value(model.cost_per_unit[best_idx, loc, best_supplier])
                    
                    # Set solution values
                    for idx, supplier in location_pairs:
                        if (idx, supplier) == (best_idx, best_supplier):
                            model.allocation_percentage[idx, loc, supplier].set_value(1.0)
                            model.supplier_selected[idx, loc, supplier].set_value(1)
                        else:
                            model.allocation_percentage[idx, loc, supplier].set_value(0.0)
                            model.supplier_selected[idx, loc, supplier].set_value(0)
                    
                    total_cost += demand * cost
            
            logger.info(f"Simple optimization completed with cost: ${total_cost:,.2f}")
            return {
                'success': True,
                'status': 'simple_optimal',
                'objective_value': total_cost,
                'message': 'Simple greedy optimization completed'
            }
            
        except Exception as e:
            logger.error(f"Simple optimization error: {str(e)}")
            return {
                'success': False,
                'status': 'error',
                'message': f"Simple optimization error: {str(e)}"
            }
    
    def _extract_enhanced_results(self, model: pyo.ConcreteModel, data: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced optimization results with percentage splits."""
        results_list = []
        
        for idx, row in data.iterrows():
            location_id = row['Plant_Product_Location_ID']
            supplier = row['Supplier']
            pair_key = (row['location_supplier_idx'], location_id, supplier)
            
            # Get optimization results
            allocation_pct = pyo.value(model.allocation_percentage[pair_key])
            is_selected = pyo.value(model.supplier_selected[pair_key]) > 0.5
            
            # Calculate optimized volumes and costs
            demand = row['2024 Volume (lbs)']
            optimized_volume = allocation_pct * demand
            optimized_cost = optimized_volume * row['DDP (USD)']
            
            # Format percentage split
            split_percentage = f"{allocation_pct * 100:.2f}%"
            
            # Create result row
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if is_selected else ''
            result_row['Optimized Split'] = split_percentage
            result_row['Is_Optimized_Supplier'] = 1 if is_selected else 0
            result_row['Allocation_Percentage'] = allocation_pct
            
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        # Log summary of percentage splits found
        selected_suppliers = results_df[results_df['Is_Optimized_Supplier'] == 1]
        percentage_splits = selected_suppliers['Allocation_Percentage'].tolist()
        
        # Count different types of allocations
        full_allocations = sum(1 for p in percentage_splits if p > 0.99)
        partial_allocations = sum(1 for p in percentage_splits if 0.01 <= p <= 0.99)
        
        logger.info(f"Optimization results: {full_allocations} full allocations, {partial_allocations} percentage splits")
        
        return results_df
    
    def _calculate_comprehensive_metrics(self, original_data: pd.DataFrame, 
                                       optimized_data: pd.DataFrame, 
                                       start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive optimization metrics and savings."""
        
        # Calculate baseline costs
        baseline_data = original_data[original_data['Is_Baseline_Supplier'] == 1].copy()
        baseline_total_cost = 0
        
        for _, row in baseline_data.iterrows():
            baseline_allocated_volume = row.get('Baseline Allocated Volume', 0)
            if baseline_allocated_volume > 0:
                baseline_total_cost += row.get('Baseline Price Paid', 
                                             baseline_allocated_volume * row['DDP (USD)'])
            else:
                # If no baseline allocation, use full volume at baseline price
                baseline_total_cost += row['2024 Volume (lbs)'] * row['DDP (USD)']
        
        # Calculate optimized costs
        optimized_total_cost = optimized_data['Optimized Price'].sum()
        
        # Calculate savings
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        
        # Calculate detailed metrics
        selected_suppliers = optimized_data[optimized_data['Is_Optimized_Supplier'] == 1]
        unique_locations_optimized = selected_suppliers['Plant_Product_Location_ID'].nunique()
        unique_suppliers_involved = selected_suppliers['Supplier'].nunique()
        
        # Count supplier switches
        supplier_switches = 0
        volume_reallocated = 0
        
        for location_id in optimized_data['Plant_Product_Location_ID'].unique():
            location_data = optimized_data[optimized_data['Plant_Product_Location_ID'] == location_id]
            
            # Check if baseline supplier changed
            baseline_supplier = location_data[location_data['Is_Baseline_Supplier'] == 1]
            optimized_suppliers = location_data[location_data['Is_Optimized_Supplier'] == 1]
            
            if not baseline_supplier.empty and not optimized_suppliers.empty:
                baseline_sup_name = baseline_supplier.iloc[0]['Supplier']
                optimized_sup_names = set(optimized_suppliers['Supplier'].tolist())
                
                # Check if baseline supplier is no longer the only supplier or has reduced allocation
                if len(optimized_sup_names) > 1 or baseline_sup_name not in optimized_sup_names:
                    supplier_switches += 1
                    volume_reallocated += location_data.iloc[0]['2024 Volume (lbs)']
        
        # Enhanced detailed results with percentage information
        detailed_results = []
        for location_id in selected_suppliers['Plant_Product_Location_ID'].unique():
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 0:
                total_cost_location = location_suppliers['Optimized Price'].sum()
                supplier_details = []
                
                for _, supplier_row in location_suppliers.iterrows():
                    allocation_pct = supplier_row['Allocation_Percentage']
                    supplier_details.append({
                        'supplier': supplier_row['Supplier'],
                        'allocation_percentage': allocation_pct,
                        'volume': supplier_row['Optimized Volume'],
                        'cost': supplier_row['Optimized Price'],
                        'price_per_unit': supplier_row['DDP (USD)']
                    })
                
                detailed_results.append({
                    'location_id': location_id,
                    'plant': location_suppliers.iloc[0]['Plant'],
                    'total_volume': location_suppliers.iloc[0]['2024 Volume (lbs)'],
                    'total_cost': total_cost_location,
                    'supplier_count': len(location_suppliers),
                    'suppliers': supplier_details
                })
        
        # Create final results dictionary
        final_results = {
            'success': True,  # Add success flag for interface compatibility
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'execution_time': time.time() - start_time,
            'results_dataframe': optimized_data,
            'optimization_summary': {
                'total_locations': optimized_data['Plant_Product_Location_ID'].nunique(),
                'locations_optimized': unique_locations_optimized,
                'total_suppliers': unique_suppliers_involved,
                'supplier_switches': supplier_switches,
                'volume_reallocated': volume_reallocated,
                'percentage_splits_used': len([s for s in selected_suppliers['Allocation_Percentage'] 
                                             if 0.01 < s < 0.99])
            },
            'detailed_results': detailed_results,
            'plants_optimized': unique_locations_optimized,
            'suppliers_involved': unique_suppliers_involved,
            'volume_redistributions': len(detailed_results),
            'supplier_switches': supplier_switches,
            'volume_reallocated': volume_reallocated
        }
        
        return final_results
    
    def _create_failure_result(self) -> Dict[str, Any]:
        """Create a failure result dictionary."""
        return {
            'success': False,  # Add success flag for interface compatibility
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


# Import the linear programming optimizer for true mathematical optimization
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Use the enhanced Pyomo optimizer as the production optimizer
# OpenSolver and other legacy optimizers have been archived for backwards compatibility
PyomoOptimizer = EnhancedPyomoOptimizer
logger.info("Using production EnhancedPyomoOptimizer for supply chain optimization")


def test_enhanced_optimizer():
    """Test the enhanced optimizer with sample data."""
    print("Testing Enhanced Pyomo Optimizer...")
    
    # Create sample data
    sample_data = {
        'Plant_Product_Location_ID': ['TestPlant_TestLocation_TestProduct'] * 3,
        'Supplier': ['Supplier_A', 'Supplier_B', 'Supplier_C'],
        '2024 Volume (lbs)': [1000000, 1000000, 1000000],
        'DDP (USD)': [5.0, 4.5, 5.5],
        'Is_Baseline_Supplier': [1, 0, 0],
        'Baseline Allocated Volume': [1000000, 0, 0],
        'Baseline Price Paid': [5000000, 0, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Sample constraints
    supplier_constraints = {
        'Supplier_A': {'min': 0, 'max': 1000000},
        'Supplier_B': {'min': 0, 'max': 1000000},
        'Supplier_C': {'min': 0, 'max': 1000000}
    }
    
    # Test optimization
    optimizer = EnhancedPyomoOptimizer()
    results = optimizer.optimize(df, supplier_constraints, {})
    
    print(f"Test Results:")
    print(f"Total Savings: ${results['total_savings']:,.2f}")
    print(f"Savings Percentage: {results['savings_percentage']:.2f}%")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print("Enhanced optimization test completed!")


if __name__ == "__main__":
    test_enhanced_optimizer()