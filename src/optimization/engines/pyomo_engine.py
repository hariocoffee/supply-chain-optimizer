"""
Pyomo Optimization Engine for Supply Chain Optimization Platform
Enhanced mathematical optimization using Pyomo with continuous variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import time
import logging

from optimization.engines.base_engine import BaseOptimizationEngine, OptimizationResult, OptimizationInput
from config import settings

logger = logging.getLogger(__name__)


class PyomoOptimizationEngine(BaseOptimizationEngine):
    """Enhanced optimization engine using Pyomo with continuous variables."""
    
    def __init__(self, solver_name: str = None):
        """Initialize the Pyomo optimizer."""
        super().__init__("pyomo")
        
        # Determine best available solver
        self.solver_name = self._find_best_solver(solver_name)
        self.model = None
        
        logger.info(f"PyomoOptimizationEngine initialized with solver: {self.solver_name}")
    
    def _find_best_solver(self, preferred_solver: str = None) -> str:
        """Find the best available solver."""
        if preferred_solver:
            candidate_solvers = [preferred_solver] + settings.optimization.available_solvers
        else:
            candidate_solvers = settings.optimization.available_solvers
        
        for solver in candidate_solvers:
            try:
                test_solver = pyo.SolverFactory(solver)
                if test_solver.available():
                    return solver
            except:
                continue
        
        # Fallback to simple approach
        logger.warning("No optimization solver found, will use simplified approach")
        return 'simple'
    
    def optimize(self, optimization_input: OptimizationInput) -> OptimizationResult:
        """Execute Pyomo optimization."""
        start_time = time.time()
        
        try:
            self.log_optimization_start(optimization_input)
            
            # Validate input
            is_valid, errors = self.validate_input(optimization_input)
            if not is_valid:
                error_msg = f"Input validation failed: {', '.join(errors)}"
                logger.error(error_msg)
                return self.create_failure_result(error_msg)
            
            # Prepare data
            processed_data = self.prepare_data(optimization_input.data)
            if processed_data.empty:
                return self.create_failure_result("No valid data to optimize")
            
            # Build mathematical model
            model = self._build_mathematical_model(processed_data, optimization_input.supplier_constraints)
            
            # Solve the optimization problem
            solver_results = self._solve_model(model)
            
            if not solver_results['success']:
                error_msg = f"Solver failed: {solver_results['message']}"
                logger.error(error_msg)
                return self.create_failure_result(error_msg)
            
            # Extract results
            optimization_results = self._extract_results(model, processed_data)
            
            # Calculate comprehensive metrics
            final_results = self._calculate_metrics(
                processed_data, optimization_results, start_time
            )
            
            result = OptimizationResult(
                success=True,
                total_savings=final_results['total_savings'],
                savings_percentage=final_results['savings_percentage'],
                baseline_total_cost=final_results['baseline_total_cost'],
                optimized_total_cost=final_results['optimized_total_cost'],
                execution_time=final_results['execution_time'],
                results_dataframe=final_results['results_dataframe'],
                detailed_results=final_results['detailed_results'],
                solver_status=solver_results['status']
            )
            
            self.log_optimization_complete(result)
            return result
            
        except Exception as e:
            error_msg = f"Optimization error: {str(e)}"
            logger.error(error_msg)
            return self.create_failure_result(error_msg)
    
    def _build_mathematical_model(self, data: pd.DataFrame, 
                                 supplier_constraints: Dict) -> pyo.ConcreteModel:
        """Build enhanced mathematical optimization model."""
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
        
        # Decision Variables - Continuous percentage allocation (0 to 1)
        model.allocation_percentage = pyo.Var(
            model.supplier_location_pairs, 
            domain=pyo.NonNegativeReals, 
            bounds=(0, 1)
        )
        
        # Binary variables for supplier selection
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
        self._add_constraints(model, supplier_constraints)
        
        self.model = model
        logger.info(f"Mathematical model built with {len(supplier_location_pairs)} decision variables")
        
        return model
    
    def _add_constraints(self, model: pyo.ConcreteModel, supplier_constraints: Dict):
        """Add constraints to the optimization model."""
        
        # 1. Demand satisfaction constraint
        def demand_satisfaction_rule(model, loc):
            return sum(
                model.allocation_percentage[idx, loc, sup]
                for idx, location, sup in model.supplier_location_pairs
                if location == loc
            ) == 1.0
        
        model.demand_satisfaction = pyo.Constraint(
            model.locations, 
            rule=demand_satisfaction_rule
        )
        
        # 2. Supplier capacity constraints
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
        
        # 3. Selection logic constraint
        def selection_logic_rule(model, idx, loc, sup):
            return model.allocation_percentage[idx, loc, sup] <= model.supplier_selected[idx, loc, sup]
        
        model.selection_logic = pyo.Constraint(
            model.supplier_location_pairs,
            rule=selection_logic_rule
        )
        
        # 4. Minimum allocation constraint
        def minimum_allocation_rule(model, idx, loc, sup):
            return model.allocation_percentage[idx, loc, sup] >= 0.01 * model.supplier_selected[idx, loc, sup]
        
        model.minimum_allocation = pyo.Constraint(
            model.supplier_location_pairs,
            rule=minimum_allocation_rule
        )
        
        # 5. Diversification constraint - up to 3 suppliers per location
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
    
    def _solve_model(self, model: pyo.ConcreteModel) -> Dict[str, Any]:
        """Solve the optimization model."""
        try:
            if self.solver_name == 'simple':
                return self._solve_simple_optimization(model)
            
            # Use available solver with enhanced settings
            solver = pyo.SolverFactory(self.solver_name)
            
            # Apply solver-specific options
            solver_config = settings.get_solver_config(self.solver_name)
            for key, value in solver_config.items():
                solver.options[key] = value
            
            logger.info("Solving mathematical optimization model...")
            results = solver.solve(model, tee=False)
            
            # Check solution status
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                logger.info("Optimal solution found!")
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': pyo.value(model.total_cost),
                    'message': 'Optimization completed successfully'
                }
            
            elif results.solver.termination_condition == TerminationCondition.feasible:
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
            
            locations = list(model.locations)
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
    
    def _extract_results(self, model: pyo.ConcreteModel, data: pd.DataFrame) -> pd.DataFrame:
        """Extract optimization results with percentage splits."""
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
        
        # Log summary of percentage splits
        selected_suppliers = results_df[results_df['Is_Optimized_Supplier'] == 1]
        percentage_splits = selected_suppliers['Allocation_Percentage'].tolist()
        
        full_allocations = sum(1 for p in percentage_splits if p > 0.99)
        partial_allocations = sum(1 for p in percentage_splits if 0.01 <= p <= 0.99)
        
        logger.info(f"Optimization results: {full_allocations} full allocations, {partial_allocations} percentage splits")
        
        return results_df
    
    def _calculate_metrics(self, original_data: pd.DataFrame, 
                          optimized_data: pd.DataFrame, 
                          start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive optimization metrics."""
        
        # Calculate baseline costs
        baseline_total_cost = self.calculate_baseline_cost(original_data)
        
        # Calculate optimized costs
        optimized_total_cost = optimized_data['Optimized Price'].sum()
        
        # Calculate savings
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        
        # Calculate detailed metrics
        selected_suppliers = optimized_data[optimized_data['Is_Optimized_Supplier'] == 1]
        unique_locations_optimized = selected_suppliers['Plant_Product_Location_ID'].nunique()
        unique_suppliers_involved = selected_suppliers['Supplier'].nunique()
        
        # Create detailed results
        detailed_results = []
        for location_id in selected_suppliers['Plant_Product_Location_ID'].unique():
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 0:
                supplier_details = []
                for _, supplier_row in location_suppliers.iterrows():
                    supplier_details.append({
                        'supplier': supplier_row['Supplier'],
                        'allocation_percentage': supplier_row['Allocation_Percentage'],
                        'volume': supplier_row['Optimized Volume'],
                        'cost': supplier_row['Optimized Price'],
                        'price_per_unit': supplier_row['DDP (USD)']
                    })
                
                detailed_results.append({
                    'location_id': location_id,
                    'plant': location_suppliers.iloc[0]['Plant'],
                    'total_volume': location_suppliers.iloc[0]['2024 Volume (lbs)'],
                    'total_cost': location_suppliers['Optimized Price'].sum(),
                    'supplier_count': len(location_suppliers),
                    'suppliers': supplier_details
                })
        
        return {
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'execution_time': time.time() - start_time,
            'results_dataframe': optimized_data,
            'detailed_results': detailed_results,
            'optimization_summary': {
                'total_locations': optimized_data['Plant_Product_Location_ID'].nunique(),
                'locations_optimized': unique_locations_optimized,
                'total_suppliers': unique_suppliers_involved,
                'percentage_splits_used': len([s for s in selected_suppliers['Allocation_Percentage'] 
                                             if 0.01 < s < 0.99])
            }
        }