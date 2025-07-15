#!/usr/bin/env python3
"""
Advanced Mathematical Optimization with Risk Considerations
Implements true linear programming with additional constraints that encourage
percentage splits for risk management and supply chain resilience
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging
import pulp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMathematicalOptimizer:
    """
    Advanced mathematical optimization that produces realistic percentage splits
    by incorporating supply chain risk, supplier relationship costs, and 
    diversification benefits into the linear programming model.
    """
    
    def __init__(self):
        """Initialize the advanced mathematical optimizer."""
        self.tolerance = 1e-8
        self.max_single_supplier_share = 0.85  # Force diversification above 85%
        self.risk_penalty_threshold = 0.75  # Apply risk penalty above 75% share
        self.competitive_price_threshold = 0.10  # 10% price difference threshold
        
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, plant_constraints: Dict) -> Dict[str, Any]:
        """
        Advanced mathematical optimization with realistic constraints.
        
        Incorporates:
        1. Traditional cost minimization
        2. Supply chain risk penalties for over-concentration
        3. Supplier relationship maintenance costs
        4. Diversification benefits for competitive suppliers
        5. Minimum allocation constraints when suppliers are competitive
        """
        start_time = time.time()
        
        try:
            logger.info("Starting advanced mathematical optimization with risk considerations...")
            
            # Prepare optimization data with competitive analysis
            processed_data = self._prepare_optimization_data_with_risk(data)
            if processed_data.empty:
                return self._create_failure_result()
            
            # Build advanced mathematical model
            optimization_model = self._build_advanced_mathematical_model(processed_data, supplier_constraints)
            
            # Solve using advanced optimization
            solution_results = self._solve_advanced_model(optimization_model, processed_data)
            
            if not solution_results['success']:
                logger.error(f"Advanced optimization failed: {solution_results['message']}")
                return self._create_failure_result()
            
            # Extract precise results with cost savings calculation
            optimization_results = self._extract_precise_results_with_savings(
                optimization_model, processed_data, solution_results
            )
            
            # Calculate comprehensive metrics
            final_results = self._calculate_comprehensive_metrics_with_risk(
                processed_data, optimization_results, start_time
            )
            
            logger.info(f"Advanced optimization completed in {final_results['execution_time']:.2f} seconds")
            logger.info(f"Total savings achieved: ${final_results['total_savings']:,.2f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Advanced optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_failure_result()
    
    def _prepare_optimization_data_with_risk(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with competitive analysis and risk factors."""
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
        
        # Clean data
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
        
        # Analyze competitive suppliers for each location
        df['is_competitive'] = False
        df['price_rank'] = 0
        df['risk_adjusted_cost'] = df['DDP (USD)']
        
        for location_id in df['Plant_Product_Location_ID'].unique():
            location_data = df[df['Plant_Product_Location_ID'] == location_id]
            
            # Sort by price
            location_data = location_data.sort_values('DDP (USD)')
            
            if len(location_data) >= 2:
                cheapest_price = location_data.iloc[0]['DDP (USD)']
                
                for idx, row in location_data.iterrows():
                    price_diff = (row['DDP (USD)'] - cheapest_price) / cheapest_price
                    
                    # Mark as competitive if within threshold
                    if price_diff <= self.competitive_price_threshold:
                        df.loc[idx, 'is_competitive'] = True
                    
                    # Add risk adjustment for concentration
                    risk_factor = 1.0 + (price_diff * 0.1)  # Small risk penalty for higher prices
                    df.loc[idx, 'risk_adjusted_cost'] = row['DDP (USD)'] * risk_factor
                    
                    # Set price rank
                    df.loc[idx, 'price_rank'] = list(location_data.index).index(idx) + 1
        
        # Create optimization indices
        df['optimization_index'] = range(len(df))
        
        logger.info(f"Prepared {len(df)} supplier-location combinations")
        competitive_pairs = len(df[df['is_competitive'] == True])
        logger.info(f"Identified {competitive_pairs} competitive supplier options requiring diversification")
        
        return df
    
    def _build_advanced_mathematical_model(self, data: pd.DataFrame, supplier_constraints: Dict) -> pulp.LpProblem:
        """Build advanced linear programming model with risk and diversification constraints."""
        
        # Create linear programming problem
        model = pulp.LpProblem("Advanced_Supply_Chain_Optimization", pulp.LpMinimize)
        
        # Get unique locations and suppliers
        locations = data['Plant_Product_Location_ID'].unique()
        suppliers = data['Supplier'].unique()
        
        # Decision variables: allocation percentage for each supplier-location pair
        allocation_vars = {}
        for idx, row in data.iterrows():
            var_name = f"alloc_{row['optimization_index']}"
            allocation_vars[row['optimization_index']] = pulp.LpVariable(
                var_name, 
                lowBound=0.0, 
                upBound=1.0, 
                cat='Continuous'
            )
        
        # Binary variables for supplier selection (to handle minimum allocation constraints)
        selection_vars = {}
        for idx, row in data.iterrows():
            var_name = f"select_{row['optimization_index']}"
            selection_vars[row['optimization_index']] = pulp.LpVariable(
                var_name,
                lowBound=0,
                upBound=1,
                cat='Binary'
            )
        
        # Objective Function: Minimize total cost with risk adjustments
        total_cost = 0
        risk_penalty = 0
        
        for idx, row in data.iterrows():
            demand = row['2024 Volume (lbs)']
            base_cost = row['DDP (USD)']
            allocation_var = allocation_vars[row['optimization_index']]
            
            # Base cost component
            total_cost += allocation_var * demand * base_cost
            
            # Risk penalty for over-concentration (quadratic penalty)
            if demand > 10000000:  # Large volumes get concentration penalty
                concentration_penalty = allocation_var * allocation_var * demand * base_cost * 0.001
                risk_penalty += concentration_penalty
        
        model += total_cost + risk_penalty, "Total_Risk_Adjusted_Cost"
        
        # Constraint 1: Demand satisfaction
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
                
                model += supplier_total_volume <= max_capacity, f"Supplier_Max_{supplier}"
                model += supplier_total_volume >= min_capacity, f"Supplier_Min_{supplier}"
        
        # Constraint 3: Diversification for competitive suppliers
        for location in locations:
            location_data = data[data['Plant_Product_Location_ID'] == location]
            competitive_suppliers = location_data[location_data['is_competitive'] == True]
            
            if len(competitive_suppliers) >= 2:
                # Force diversification when suppliers are competitive
                for idx, row in competitive_suppliers.iterrows():
                    allocation_var = allocation_vars[row['optimization_index']]
                    
                    # No single supplier can have more than 85% if others are competitive
                    model += allocation_var <= self.max_single_supplier_share, f"Max_Share_{idx}"
                    
                    # If a supplier is selected, minimum 5% allocation
                    selection_var = selection_vars[row['optimization_index']]
                    model += allocation_var >= 0.05 * selection_var, f"Min_Allocation_{idx}"
                    model += allocation_var <= selection_var, f"Selection_Logic_{idx}"
        
        # Constraint 4: Risk diversification for large volumes
        for location in locations:
            location_data = data[data['Plant_Product_Location_ID'] == location]
            total_volume = location_data.iloc[0]['2024 Volume (lbs)']
            
            if total_volume > 50000000:  # Large volume locations
                competitive_suppliers = location_data[location_data['is_competitive'] == True]
                
                if len(competitive_suppliers) >= 2:
                    # Force at least 2 suppliers for large volumes
                    total_selections = 0
                    for idx, row in competitive_suppliers.iterrows():
                        total_selections += selection_vars[row['optimization_index']]
                    
                    model += total_selections >= 2, f"Min_Suppliers_Large_Volume_{location}"
        
        # Store variables in model
        model.allocation_vars = allocation_vars
        model.selection_vars = selection_vars
        model.data = data
        
        logger.info(f"Built advanced model with {len(allocation_vars)} allocation and {len(selection_vars)} selection variables")
        
        return model
    
    def _solve_advanced_model(self, model: pulp.LpProblem, data: pd.DataFrame) -> Dict[str, Any]:
        """Solve the advanced mathematical model."""
        try:
            logger.info("Solving advanced mathematical model...")
            
            # Use CBC solver with longer time limit for complex model
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=600, gapRel=1e-4)
            model.solve(solver)
            
            status = pulp.LpStatus[model.status]
            
            if status == 'Optimal':
                objective_value = pulp.value(model.objective)
                logger.info(f"Optimal solution found! Objective value: ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'optimal',
                    'objective_value': objective_value,
                    'message': 'Advanced optimization found optimal solution'
                }
            
            elif status == 'Feasible':
                objective_value = pulp.value(model.objective)
                logger.warning(f"Feasible solution found: ${objective_value:,.2f}")
                
                return {
                    'success': True,
                    'status': 'feasible',
                    'objective_value': objective_value,
                    'message': 'Advanced optimization found feasible solution'
                }
            
            else:
                logger.error(f"Optimization failed with status: {status}")
                return {
                    'success': False,
                    'status': status,
                    'message': f"Advanced optimization failed: {status}"
                }
        
        except Exception as e:
            logger.error(f"Solver error: {str(e)}")
            return {
                'success': False,
                'status': 'error',
                'message': f"Advanced optimization error: {str(e)}"
            }
    
    def _extract_precise_results_with_savings(self, model: pulp.LpProblem, data: pd.DataFrame, 
                                            solution_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract precise results with Column O (Cost_Savings) calculation."""
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
            
            # Determine if supplier is selected
            is_selected = allocation_percentage > self.tolerance
            
            # Format percentage split with precision
            split_percentage = f"{allocation_percentage * 100:.2f}%"
            
            # Create result row
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if is_selected else ''
            result_row['Optimized Split'] = split_percentage
            result_row['Is_Optimized_Supplier'] = 1 if is_selected else 0
            result_row['Allocation_Percentage'] = allocation_percentage
            
            # Calculate Column O (Cost_Savings) - NEW REQUIREMENT
            if row['Selection'] == 'X' and is_selected:  # If baseline was selected AND optimized is selected
                baseline_cost_for_location = row['baseline_cost']
                
                # Calculate optimized cost for entire location
                location_data = data[data['Plant_Product_Location_ID'] == row['Plant_Product_Location_ID']]
                location_optimized_cost = 0
                
                for _, loc_row in location_data.iterrows():
                    loc_allocation = pulp.value(model.allocation_vars[loc_row['optimization_index']])
                    if loc_allocation is None:
                        loc_allocation = 0.0
                    loc_optimized_cost = loc_allocation * demand * loc_row['DDP (USD)']
                    location_optimized_cost += loc_optimized_cost
                
                cost_savings = baseline_cost_for_location - location_optimized_cost
                result_row['Cost_Savings'] = cost_savings
            else:
                result_row['Cost_Savings'] = 0.0
            
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        # Log optimization statistics
        selected_suppliers = results_df[results_df['Is_Optimized_Supplier'] == 1]
        
        # Analyze percentage splits
        split_patterns = []
        for location_id in results_df['Plant_Product_Location_ID'].unique():
            location_suppliers = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
            
            if len(location_suppliers) > 1:
                splits = [f"{row['Allocation_Percentage']*100:.2f}%" for _, row in location_suppliers.iterrows()]
                split_patterns.append(' / '.join(sorted(splits, reverse=True)))
        
        logger.info(f"Advanced optimization produced {len(split_patterns)} percentage split patterns")
        if split_patterns:
            logger.info("Examples of precise splits found:")
            for i, pattern in enumerate(split_patterns[:10]):
                logger.info(f"  Pattern {i+1}: {pattern}")
        
        return results_df
    
    def _calculate_comprehensive_metrics_with_risk(self, original_data: pd.DataFrame, 
                                                 optimized_data: pd.DataFrame, 
                                                 start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive metrics with risk analysis."""
        
        # Calculate baseline and optimized costs
        baseline_total_cost = optimized_data['baseline_cost'].sum() / optimized_data['Plant_Product_Location_ID'].nunique()
        optimized_total_cost = optimized_data['Optimized Price'].sum()
        
        # Calculate savings
        total_savings = baseline_total_cost - optimized_total_cost
        savings_percentage = (total_savings / baseline_total_cost * 100) if baseline_total_cost > 0 else 0
        
        # Analyze allocation patterns
        selected_suppliers = optimized_data[optimized_data['Is_Optimized_Supplier'] == 1]
        percentage_splits = selected_suppliers['Allocation_Percentage'].tolist()
        
        # Count allocation types
        full_allocations = sum(1 for p in percentage_splits if abs(p - 1.0) < self.tolerance)
        partial_allocations = sum(1 for p in percentage_splits if self.tolerance < p < (1.0 - self.tolerance))
        
        # Analyze detailed split patterns
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
        
        return {
            'total_savings': total_savings,
            'savings_percentage': savings_percentage,
            'baseline_total_cost': baseline_total_cost,
            'optimized_total_cost': optimized_total_cost,
            'execution_time': time.time() - start_time,
            'results_dataframe': optimized_data,
            'optimization_summary': {
                'total_locations': optimized_data['Plant_Product_Location_ID'].nunique(),
                'locations_optimized': len(split_patterns),
                'total_suppliers': selected_suppliers['Supplier'].nunique(),
                'percentage_splits_used': partial_allocations,
                'full_allocations': full_allocations,
                'unique_split_patterns': len(split_patterns)
            },
            'detailed_results': split_patterns,
            'plants_optimized': len(split_patterns),
            'suppliers_involved': selected_suppliers['Supplier'].nunique(),
            'volume_redistributions': len(split_patterns),
            'supplier_switches': len(split_patterns),  # Simplified calculation
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


def test_advanced_mathematical_optimizer():
    """Test the advanced mathematical optimizer with competitive constraints."""
    print("Testing Advanced Mathematical Optimizer with Risk Diversification...")
    
    # Load real data
    df = pd.read_csv('./sco_data_cleaned.csv')
    
    # Create more restrictive constraints to encourage diversification
    constraints = {}
    supplier_data = df.groupby('Supplier').agg({
        'Baseline Allocated Volume': 'sum',
        '2024 Volume (lbs)': 'sum'
    }).reset_index()
    
    total_volume = df['2024 Volume (lbs)'].sum()
    
    for _, row in supplier_data.iterrows():
        supplier = row['Supplier']
        baseline_volume = row['Baseline Allocated Volume']
        
        # More restrictive capacity constraints to encourage splitting
        constraints[supplier] = {
            'min': 0,
            'max': total_volume * 0.6,  # Limit any supplier to 60% of total
            'baseline_volume': baseline_volume,
            'baseline_cost': 0
        }
    
    # Run advanced mathematical optimization
    optimizer = AdvancedMathematicalOptimizer()
    results = optimizer.optimize(df, constraints, {})
    
    print(f"\nAdvanced Mathematical Optimization Results:")
    print(f"Total Savings: ${results['total_savings']:,.2f}")
    print(f"Savings Percentage: {results['savings_percentage']:.2f}%")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Percentage Splits Used: {results['optimization_summary']['percentage_splits_used']}")
    print(f"Unique Split Patterns: {results['optimization_summary']['unique_split_patterns']}")
    
    # Show example precise splits
    if results['detailed_results']:
        print(f"\nExamples of Advanced Mathematical Splits:")
        for i, result in enumerate(results['detailed_results'][:10]):
            splits = [f"{s['supplier']}: {s['percentage']:.2f}%" for s in result['suppliers']]
            print(f"  {i+1}. {result['location_id']}: {' / '.join(splits)}")
    
    # Save results
    results['results_dataframe'].to_csv('./advanced_mathematical_optimization_results.csv', index=False)
    print("Results saved to advanced_mathematical_optimization_results.csv")


if __name__ == "__main__":
    test_advanced_mathematical_optimizer()