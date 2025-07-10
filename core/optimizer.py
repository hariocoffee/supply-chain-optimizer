import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyomoOptimizer:
    def __init__(self, solver_name: str = 'cbc'):
        self.solver_name = solver_name
        self.model = None
        self.solver = None
        
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Optional[Dict]:
        """
        Main optimization function that mimics OpenSolver behavior.
        Minimizes total cost while respecting supplier and plant constraints.
        """
        try:
            start_time = time.time()
            
            # Validate and prepare data
            if not self._validate_data(data):
                logger.error("Data validation failed")
                return None
            
            # Prepare optimization data
            opt_data = self._prepare_optimization_data(data, supplier_constraints, plant_constraints)
            
            # Build optimization model
            model = self._build_model(opt_data)
            
            # Solve the model
            solution = self._solve_model(model)
            
            if solution is None:
                logger.error("Optimization failed")
                return None
            
            # Process results
            results = self._process_results(data, solution, opt_data)
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return None
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = [
            'Plant', 'Supplier', '2024 Volume (lbs)', 'DDP (USD)',
            'Baseline Allocated Volume', 'Baseline Price Paid'
        ]
        
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for negative values
        if (data['2024 Volume (lbs)'] < 0).any():
            logger.error("Found negative volume values")
            return False
        
        if (data['DDP (USD)'] < 0).any():
            logger.error("Found negative price values")
            return False
        
        return True
    
    def _prepare_optimization_data(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict:
        """Prepare data for optimization."""
        
        # Create plant-supplier combinations with their constraints
        combinations = []
        plant_demands = {}
        
        # Group by plant to get total demand per plant (Column C - 2024 Volume)
        for plant in data['Plant'].unique():
            plant_data = data[data['Plant'] == plant]
            total_demand = plant_data['2024 Volume (lbs)'].iloc[0]  # Each plant must receive this exact volume
            plant_demands[plant] = total_demand
        
        # Create valid combinations (plant-supplier pairs)
        # Each row represents a potential supply relationship
        for _, row in data.iterrows():
            plant = row['Plant']
            supplier = row['Supplier']
            plant_demand = row['2024 Volume (lbs)']  # Total demand for this plant
            price = row['DDP (USD)']
            
            # The capacity for this supplier-plant combination is the full plant demand
            # (the supplier could potentially supply the entire plant demand)
            combinations.append({
                'plant': plant,
                'supplier': supplier,
                'capacity': plant_demand,  # Max this supplier can supply to this plant
                'price': price,
                'baseline_volume': row.get('Baseline Allocated Volume', 0),
                'baseline_cost': row.get('Baseline Price Paid', 0)
            })
        
        plants = list(plant_demands.keys())
        suppliers = list(set([combo['supplier'] for combo in combinations]))
        
        return {
            'combinations': combinations,
            'plants': plants,
            'suppliers': suppliers,
            'plant_demands': plant_demands,
            'supplier_constraints': supplier_constraints,
            'plant_constraints': plant_constraints
        }
    
    def _build_model(self, opt_data: Dict) -> pyo.ConcreteModel:
        """Build the Pyomo optimization model."""
        
        model = pyo.ConcreteModel()
        
        # Sets
        model.plants = pyo.Set(initialize=opt_data['plants'])
        model.suppliers = pyo.Set(initialize=opt_data['suppliers'])
        
        # Create valid combinations
        valid_combinations = []
        capacity_dict = {}
        price_dict = {}
        
        for combo in opt_data['combinations']:
            key = (combo['plant'], combo['supplier'])
            valid_combinations.append(key)
            capacity_dict[key] = combo['capacity']
            price_dict[key] = combo['price']
        
        model.combinations = pyo.Set(initialize=valid_combinations)
        
        # Parameters
        model.capacity = pyo.Param(model.combinations, initialize=capacity_dict)
        model.price = pyo.Param(model.combinations, initialize=price_dict)
        model.demand = pyo.Param(model.plants, initialize=opt_data['plant_demands'])
        
        # Decision variables
        model.allocation = pyo.Var(model.combinations, domain=pyo.NonNegativeReals)
        
        # Objective function: minimize total cost
        def objective_rule(model):
            return sum(model.allocation[p, s] * model.price[p, s] 
                      for p, s in model.combinations)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Constraints
        
        # 1. Demand satisfaction: each plant must receive its required volume
        def demand_constraint_rule(model, plant):
            return sum(model.allocation[plant, s] for s in model.suppliers 
                      if (plant, s) in model.combinations) == model.demand[plant]
        
        model.demand_constraint = pyo.Constraint(model.plants, rule=demand_constraint_rule)
        
        # 2. Capacity constraints: allocation cannot exceed supplier capacity
        def capacity_constraint_rule(model, plant, supplier):
            return model.allocation[plant, supplier] <= model.capacity[plant, supplier]
        
        model.capacity_constraint = pyo.Constraint(model.combinations, rule=capacity_constraint_rule)
        
        # 3. Supplier volume constraints
        def supplier_min_constraint_rule(model, supplier):
            total_supply = sum(model.allocation[p, supplier] for p in model.plants 
                             if (p, supplier) in model.combinations)
            min_volume = opt_data['supplier_constraints'].get(supplier, {}).get('min', 0)
            return total_supply >= min_volume
        
        def supplier_max_constraint_rule(model, supplier):
            total_supply = sum(model.allocation[p, supplier] for p in model.plants 
                             if (p, supplier) in model.combinations)
            max_volume = opt_data['supplier_constraints'].get(supplier, {}).get('max', float('inf'))
            return total_supply <= max_volume
        
        model.supplier_min_constraint = pyo.Constraint(model.suppliers, rule=supplier_min_constraint_rule)
        model.supplier_max_constraint = pyo.Constraint(model.suppliers, rule=supplier_max_constraint_rule)
        
        return model
    
    def _solve_model(self, model: pyo.ConcreteModel) -> Optional[Dict]:
        """Solve the optimization model."""
        try:
            # Create solver
            solver = pyo.SolverFactory(self.solver_name)
            
            # Solve
            results = solver.solve(model, tee=False)
            
            # Check solver status
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                logger.info("Optimization solved successfully")
                
                # Extract solution
                solution = {}
                for plant, supplier in model.combinations:
                    allocation = pyo.value(model.allocation[plant, supplier])
                    if allocation > 0.001:  # Only include non-zero allocations
                        solution[(plant, supplier)] = allocation
                
                return {
                    'status': 'optimal',
                    'objective_value': pyo.value(model.objective),
                    'allocations': solution
                }
            
            else:
                logger.error(f"Solver failed: {results.solver.status}")
                return None
                
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            return None
    
    def _process_results(self, original_data: pd.DataFrame, solution: Dict, opt_data: Dict) -> Dict:
        """Process optimization results and create output."""
        
        # Create results DataFrame
        results_df = original_data.copy()
        
        # Initialize optimization columns
        results_df['Optimized Volume'] = 0
        results_df['Optimized Price'] = 0
        results_df['Optimized Selection'] = ''
        results_df['Optimized Split'] = 0
        
        # Fill in optimized allocations
        allocations = solution['allocations']
        
        # Calculate plant totals for split percentages
        plant_totals = {}
        for (plant, supplier), volume in allocations.items():
            if plant not in plant_totals:
                plant_totals[plant] = 0
            plant_totals[plant] += volume
        
        for idx, row in results_df.iterrows():
            plant = row['Plant']
            supplier = row['Supplier']
            
            if (plant, supplier) in allocations:
                optimized_volume = allocations[(plant, supplier)]
                optimized_price = optimized_volume * row['DDP (USD)']
                
                results_df.loc[idx, 'Optimized Volume'] = optimized_volume
                results_df.loc[idx, 'Optimized Price'] = optimized_price
                results_df.loc[idx, 'Optimized Selection'] = 'X'
                
                # Calculate split percentage based on plant total
                plant_total = plant_totals.get(plant, 0)
                if plant_total > 0:
                    split_percentage = (optimized_volume / plant_total) * 100
                    results_df.loc[idx, 'Optimized Split'] = f"{split_percentage:.0f}%"
                else:
                    results_df.loc[idx, 'Optimized Split'] = "0%"
            else:
                # No allocation for this supplier-plant combination
                results_df.loc[idx, 'Optimized Volume'] = 0
                results_df.loc[idx, 'Optimized Price'] = 0
                results_df.loc[idx, 'Optimized Selection'] = ''
                results_df.loc[idx, 'Optimized Split'] = "0%"
        
        # Calculate cost savings for each row
        results_df['Cost Savings'] = results_df['Baseline Price Paid'] - results_df['Optimized Price']
        
        # Calculate total savings
        baseline_total_cost = results_df['Baseline Price Paid'].sum()
        optimized_total_cost = results_df['Optimized Price'].sum()
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0
        
        # Create detailed results for analysis
        detailed_results = []
        for idx, row in results_df.iterrows():
            if row['Optimized Volume'] > 0:
                detailed_results.append({
                    'plant': row['Plant'],
                    'supplier': row['Supplier'],
                    'baseline_volume': row['Baseline Allocated Volume'],
                    'optimized_volume': row['Optimized Volume'],
                    'baseline_cost': row['Baseline Price Paid'],
                    'optimized_cost': row['Optimized Price'],
                    'cost_savings': row['Baseline Price Paid'] - row['Optimized Price'],
                    'volume_split': row['Optimized Split'],
                    'selection_flag': row['Optimized Selection'] == 'X'
                })
        
        return {
            'status': 'success',
            'results_dataframe': results_df,
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'detailed_results': detailed_results,
            'optimization_summary': {
                'total_plants': len(opt_data['plants']),
                'total_suppliers': len(opt_data['suppliers']),
                'total_combinations': len(opt_data['combinations']),
                'selected_combinations': len([r for r in detailed_results if r['selection_flag']])
            }
        }
    
    def generate_optimization_insights(self, results: Dict) -> List[str]:
        """Generate insights about the optimization results."""
        insights = []
        
        if results['savings_percentage'] > 0:
            insights.append(f"✅ Achieved {results['savings_percentage']:.1f}% cost savings (${results['total_savings']:,.2f})")
        else:
            insights.append("⚠️ No cost savings achieved - current allocation may already be optimal")
        
        # Analyze supplier utilization
        detailed_results = results['detailed_results']
        supplier_utilization = {}
        
        for result in detailed_results:
            supplier = result['supplier']
            if supplier not in supplier_utilization:
                supplier_utilization[supplier] = {'volume': 0, 'cost': 0}
            supplier_utilization[supplier]['volume'] += result['optimized_volume']
            supplier_utilization[supplier]['cost'] += result['optimized_cost']
        
        # Find most utilized supplier
        if supplier_utilization:
            max_supplier = max(supplier_utilization.keys(), 
                             key=lambda s: supplier_utilization[s]['volume'])
            insights.append(f"🏆 Primary supplier: {max_supplier} ({supplier_utilization[max_supplier]['volume']:,.0f} lbs)")
        
        # Analyze plant distribution
        plant_distribution = {}
        for result in detailed_results:
            plant = result['plant']
            if plant not in plant_distribution:
                plant_distribution[plant] = 0
            plant_distribution[plant] += result['optimized_volume']
        
        if len(plant_distribution) > 1:
            insights.append(f"📍 Optimized across {len(plant_distribution)} plants")
        
        return insights