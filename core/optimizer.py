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
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for supplier selection optimization."""
        required_columns = [
            'Plant', 'Product', '2024 Volume (lbs)', 'Supplier', 'Plant Location',
            'DDP (USD)', 'Baseline Allocated Volume', 'Baseline Price Paid',
            'Plant_Product_Location_ID', 'Supplier_Plant_Product_Location_ID'
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
        
        # Check that each plant-product-location combination has consistent demand
        plant_product_locations = data.groupby('Plant_Product_Location_ID')['2024 Volume (lbs)'].nunique()
        if (plant_product_locations > 1).any():
            # Log which combinations have inconsistent volumes for debugging
            inconsistent = plant_product_locations[plant_product_locations > 1]
            for location_id in inconsistent.index:
                location_data = data[data['Plant_Product_Location_ID'] == location_id]
                volumes = location_data['2024 Volume (lbs)'].unique()
                logger.error(f"Location {location_id} has inconsistent volumes: {volumes}")
            logger.error("Inconsistent volume demands found for plant-product-location combinations")
            return False
        
        # Check for duplicate supplier entries (same Plant+Product+Location+Supplier combination)
        duplicate_check = data.groupby(['Plant_Product_Location_ID', 'Supplier']).size()
        if (duplicate_check > 1).any():
            duplicates = duplicate_check[duplicate_check > 1]
            for (location_id, supplier), count in duplicates.items():
                logger.error(f"Duplicate supplier entry: {supplier} appears {count} times for {location_id}")
            logger.error("Found duplicate supplier entries for same plant-product-location combinations")
            return False
        
        # Check that each plant-product-location has at least one supplier option
        plant_locations = data['Plant_Product_Location_ID'].unique()
        for plant_location in plant_locations:
            suppliers = data[data['Plant_Product_Location_ID'] == plant_location]['Supplier'].unique()
            if len(suppliers) == 0:
                logger.error(f"No suppliers found for {plant_location}")
                return False
        
        return True
    
    def _prepare_optimization_data(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict:
        """Prepare data for optimization with plant-product-location structure."""
        
        # CRITICAL FIX: Optimize at Plant+Product+Location level, not Plant+Product level
        # Each location has its own demand and must be satisfied exactly
        
        combinations = []
        location_demands = {}
        
        # Group by Plant+Product+Location to get demand for each unique location
        for location_id in data['Plant_Product_Location_ID'].unique():
            location_data = data[data['Plant_Product_Location_ID'] == location_id]
            # Each location has its own specific demand (Column C)
            demand = location_data['2024 Volume (lbs)'].iloc[0]
            location_demands[location_id] = demand
        
        # Create valid combinations (location-supplier pairs)
        # Each row represents a potential supply relationship at the location level
        for _, row in data.iterrows():
            location_id = row['Plant_Product_Location_ID']
            supplier = row['Supplier']
            demand = location_demands[location_id]  # Demand for this specific location
            price = row['DDP (USD)']
            
            # The capacity for this supplier-location combination is the full location demand
            combinations.append({
                'location': location_id,
                'supplier': supplier,
                'capacity': demand,  # Max this supplier can supply to this location
                'price': price,
                'baseline_volume': row.get('Baseline Allocated Volume', 0),
                'baseline_cost': row.get('Baseline Price Paid', 0),
                'plant': row['Plant'],
                'product': row['Product'],
                'plant_product': row['Plant_Product_ID']
            })
        
        locations = list(location_demands.keys())
        suppliers = list(set([combo['supplier'] for combo in combinations]))
        
        return {
            'combinations': combinations,
            'locations': locations,
            'suppliers': suppliers,
            'location_demands': location_demands,
            'supplier_constraints': supplier_constraints,
            'plant_constraints': plant_constraints
        }
    
    def _build_model(self, opt_data: Dict) -> pyo.ConcreteModel:
        """Build the Pyomo optimization model for location-supplier selection."""
        
        model = pyo.ConcreteModel()
        
        # Sets
        model.locations = pyo.Set(initialize=opt_data['locations'])
        model.suppliers = pyo.Set(initialize=opt_data['suppliers'])
        
        # Create valid combinations
        valid_combinations = []
        capacity_dict = {}
        price_dict = {}
        
        for combo in opt_data['combinations']:
            key = (combo['location'], combo['supplier'])
            valid_combinations.append(key)
            capacity_dict[key] = combo['capacity']
            price_dict[key] = combo['price']
        
        model.combinations = pyo.Set(initialize=valid_combinations)
        
        # Parameters
        model.capacity = pyo.Param(model.combinations, initialize=capacity_dict)
        model.price = pyo.Param(model.combinations, initialize=price_dict)
        model.demand = pyo.Param(model.locations, initialize=opt_data['location_demands'])
        
        # Decision variables
        model.allocation = pyo.Var(model.combinations, domain=pyo.NonNegativeReals)
        
        # Objective function: minimize total cost
        def objective_rule(model):
            return sum(model.allocation[loc, s] * model.price[loc, s] 
                      for loc, s in model.combinations)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Constraints
        
        # 1. Demand satisfaction: each location must receive its required volume (Column C)
        def demand_constraint_rule(model, location):
            return sum(model.allocation[location, s] for s in model.suppliers 
                      if (location, s) in model.combinations) == model.demand[location]
        
        model.demand_constraint = pyo.Constraint(model.locations, rule=demand_constraint_rule)
        
        # 2. Capacity constraints: allocation cannot exceed supplier capacity for each location
        def capacity_constraint_rule(model, location, supplier):
            return model.allocation[location, supplier] <= model.capacity[location, supplier]
        
        model.capacity_constraint = pyo.Constraint(model.combinations, rule=capacity_constraint_rule)
        
        # 3. Supplier volume constraints
        def supplier_min_constraint_rule(model, supplier):
            total_supply = sum(model.allocation[loc, supplier] for loc in model.locations 
                             if (loc, supplier) in model.combinations)
            min_volume = opt_data['supplier_constraints'].get(supplier, {}).get('min', 0)
            return total_supply >= min_volume
        
        def supplier_max_constraint_rule(model, supplier):
            total_supply = sum(model.allocation[loc, supplier] for loc in model.locations 
                             if (loc, supplier) in model.combinations)
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
                for location, supplier in model.combinations:
                    allocation = pyo.value(model.allocation[location, supplier])
                    if allocation > 0.001:  # Only include non-zero allocations
                        solution[(location, supplier)] = allocation
                
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
        """Process optimization results with proper location-level optimization and Plant+Product aggregation."""
        
        # Create results DataFrame
        results_df = original_data.copy()
        
        # Initialize optimization columns
        results_df['Optimized Volume'] = 0.0
        results_df['Optimized Price'] = 0.0
        results_df['Optimized Selection'] = ''
        results_df['Optimized Split'] = '0%'
        results_df['Is_Optimized_Supplier'] = 0
        results_df['Cost Savings'] = 0.0
        results_df['Allocated_Cost_Savings'] = 0.0
        
        # Get optimized allocations (now at location level)
        allocations = solution['allocations']
        
        # Step 1: Fill in optimized volumes at location level
        for idx, row in results_df.iterrows():
            location_id = row['Plant_Product_Location_ID']
            supplier = row['Supplier']
            location_volume = row['2024 Volume (lbs)']  # Column C for this location
            
            if (location_id, supplier) in allocations:
                # This supplier gets allocation for this location
                optimized_volume = allocations[(location_id, supplier)]
                
                # Column K = Optimized Volume (from optimization)
                results_df.loc[idx, 'Optimized Volume'] = optimized_volume
                
                # Column L = Column K × Column F (DDP)
                optimized_price = optimized_volume * row['DDP (USD)']
                results_df.loc[idx, 'Optimized Price'] = optimized_price
                
                # Mark as selected
                results_df.loc[idx, 'Optimized Selection'] = 'X'
                results_df.loc[idx, 'Is_Optimized_Supplier'] = 1
                
                # Column N = Optimized Split as percentage of location volume
                if location_volume > 0:
                    split_percentage = (optimized_volume / location_volume) * 100
                    results_df.loc[idx, 'Optimized Split'] = f"{split_percentage:.0f}%"
                    results_df.loc[idx, 'Volume_Fraction'] = f"{split_percentage/100:.3f}"
                else:
                    results_df.loc[idx, 'Optimized Split'] = "0%"
                    results_df.loc[idx, 'Volume_Fraction'] = "0.000"
            else:
                # This supplier gets no allocation
                results_df.loc[idx, 'Optimized Volume'] = 0.0
                results_df.loc[idx, 'Optimized Price'] = 0.0
                results_df.loc[idx, 'Optimized Selection'] = ''
                results_df.loc[idx, 'Optimized Split'] = "0%"
                results_df.loc[idx, 'Is_Optimized_Supplier'] = 0
                results_df.loc[idx, 'Volume_Fraction'] = "0.000"
        
        # Step 2: Calculate Plant-Product level aggregated costs and savings
        plant_product_costs = {}
        
        for plant_product in results_df['Plant_Product_ID'].unique():
            pp_rows = results_df[results_df['Plant_Product_ID'] == plant_product]
            
            # Calculate baseline and optimized costs for this plant-product (sum across all locations)
            baseline_cost = pp_rows['Baseline Price Paid'].sum()
            optimized_cost = pp_rows['Optimized Price'].sum()
            total_savings = baseline_cost - optimized_cost
            
            plant_product_costs[plant_product] = {
                'baseline_cost': baseline_cost,
                'optimized_cost': optimized_cost,
                'total_savings': total_savings
            }
            
            # Update all rows for this plant-product with the same aggregated values
            results_df.loc[results_df['Plant_Product_ID'] == plant_product, 'Plant_Product_Baseline_Cost'] = baseline_cost
            results_df.loc[results_df['Plant_Product_ID'] == plant_product, 'Plant_Product_Optimized_Cost'] = optimized_cost
            results_df.loc[results_df['Plant_Product_ID'] == plant_product, 'Plant_Product_Total_Savings'] = total_savings
        
        # Step 3: Allocate Plant-Product level savings to individual location-supplier combinations
        for idx, row in results_df.iterrows():
            plant_product = row['Plant_Product_ID']
            optimized_volume = row['Optimized Volume']
            total_savings = plant_product_costs[plant_product]['total_savings']
            
            # Calculate total Plant-Product optimized volume for proportion calculation
            pp_rows = results_df[results_df['Plant_Product_ID'] == plant_product]
            total_plant_product_optimized_volume = pp_rows['Optimized Volume'].sum()
            
            # Cost Savings at row level is generally 0 unless specified otherwise
            results_df.loc[idx, 'Cost Savings'] = 0.0
            
            # Allocated Cost Savings is proportional to this location's volume within the Plant-Product
            if optimized_volume > 0 and total_plant_product_optimized_volume > 0:
                volume_proportion = optimized_volume / total_plant_product_optimized_volume
                allocated_savings = total_savings * volume_proportion
                results_df.loc[idx, 'Allocated_Cost_Savings'] = allocated_savings
            else:
                results_df.loc[idx, 'Allocated_Cost_Savings'] = 0.0
        
        # Calculate overall totals
        baseline_total_cost = results_df['Baseline Price Paid'].sum()
        optimized_total_cost = results_df['Optimized Price'].sum()
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0
        
        # Create detailed results for analysis
        detailed_results = []
        for idx, row in results_df.iterrows():
            if row['Optimized Volume'] > 0:
                detailed_results.append({
                    'location': row['Plant_Product_Location_ID'],
                    'plant_product': row['Plant_Product_ID'],
                    'plant': row['Plant'],
                    'supplier': row['Supplier'],
                    'baseline_volume': row['Baseline Allocated Volume'],
                    'optimized_volume': row['Optimized Volume'],
                    'baseline_cost': row['Baseline Price Paid'],
                    'optimized_cost': row['Optimized Price'],
                    'allocated_savings': row['Allocated_Cost_Savings'],
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
                'total_locations': len(opt_data['locations']),
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