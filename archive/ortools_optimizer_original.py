"""
Advanced Supply Chain Optimization using OR-Tools
==================================================
Implements the most powerful optimization algorithm for supply chain optimization
with sophisticated constraint handling and company volume requirements.

Key Features:
- OR-Tools integration with fallback solvers
- Company volume constraint optimization
- Plant-level demand satisfaction
- Supplier capacity constraints
- Cost minimization objective
"""

from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SupplierConstraint:
    """Supplier capacity and pricing constraints"""
    min_capacity: float
    max_capacity: float
    current_allocation: float
    avg_price: float

@dataclass
class OptimizationResult:
    """Complete optimization results"""
    success: bool
    solver_status: str
    results_dataframe: pd.DataFrame
    total_savings: float
    savings_percentage: float
    baseline_total_cost: float
    optimized_total_cost: float
    execution_time: float
    company_volume_used: float
    company_volume_limit: float
    constraint_violations: List[str]

class ORToolsOptimizer:
    """
    OR-Tools based supply chain optimizer with company volume constraints.
    
    Solves the complex optimization problem:
    Minimize: Total procurement cost
    Subject to:
    - Plant demand satisfaction (each plant 100% allocated)
    - Supplier capacity constraints
    - Company volume constraint
    - Non-negativity constraints
    """
    
    def __init__(self):
        self.solver = None
        self.tolerance = 1e-6
        self._initialize_solver()
    
    def _initialize_solver(self):
        """Initialize the most powerful OR-Tools solver available"""
        solver_preferences = [
            'GUROBI_MIXED_INTEGER_PROGRAMMING',
            'CPLEX_MIXED_INTEGER_PROGRAMMING', 
            'SCIP_MIXED_INTEGER_PROGRAMMING',
            'CBC_MIXED_INTEGER_PROGRAMMING',
            'GLOP_LINEAR_PROGRAMMING'
        ]
        
        for solver_name in solver_preferences:
            try:
                self.solver = pywraplp.Solver.CreateSolver(solver_name)
                if self.solver:
                    logger.info(f"Using {solver_name} for optimization")
                    break
            except Exception as e:
                logger.debug(f"Failed to initialize {solver_name}: {e}")
                continue
        
        if not self.solver:
            raise Exception("No suitable OR-Tools solver found")
    
    def optimize(self, data: pd.DataFrame, supplier_constraints: Dict, 
                plant_constraints: Dict, company_volume_limit: Optional[float] = None) -> OptimizationResult:
        """
        Optimize supply chain allocation using OR-Tools.
        
        Args:
            data: Supply chain data with plants, suppliers, volumes, prices
            supplier_constraints: Supplier capacity constraints
            plant_constraints: Plant-level constraints (unused for now)
            company_volume_limit: Total company buying limit in lbs
        
        Returns:
            OptimizationResult with complete optimization solution
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("Starting OR-Tools supply chain optimization...")
            
            # Prepare data
            processed_data = self._prepare_data(data)
            if processed_data.empty:
                return self._create_failure_result("No valid data to optimize")
            
            # Build optimization model
            if company_volume_limit:
                result = self._solve_with_company_constraint(
                    processed_data, supplier_constraints, company_volume_limit
                )
            else:
                result = self._solve_standard_optimization(
                    processed_data, supplier_constraints
                )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            logger.info(f"OR-Tools optimization completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"OR-Tools optimization failed: {e}")
            return self._create_failure_result(str(e))
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate optimization data"""
        # Ensure required columns exist
        required_cols = ['Plant', 'Supplier', '2024 Volume (lbs)', 'DDP (USD)', 'Plant Location']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create unique identifiers
        processed_data = data.copy()
        processed_data['opt_index'] = range(len(processed_data))
        processed_data['plant_supplier_id'] = (
            processed_data['Plant'] + '_' + 
            processed_data['Plant Location'] + '_' +
            processed_data['Supplier']
        )
        
        # Calculate baseline costs properly
        # Use Baseline Price Paid when available, otherwise calculate from current allocation
        processed_data['baseline_cost_per_location'] = processed_data.apply(
            lambda row: row['Baseline Price Paid'] if pd.notna(row['Baseline Price Paid']) and row['Baseline Price Paid'] > 0 
            else row['2024 Volume (lbs)'] * row['DDP (USD)'], axis=1
        )
        
        return processed_data
    
    def _solve_with_company_constraint(self, data: pd.DataFrame, 
                                     supplier_constraints: Dict,
                                     company_volume_limit: float) -> OptimizationResult:
        """Enhanced optimization with company volume constraint and perfect cost minimization"""
        
        logger.info(f"🚀 Starting enhanced optimization with company volume limit: {company_volume_limit:,.0f} lbs")
        
        self.solver.Clear()
        
        # Create plant-product-location groups (unique demand points)
        data['demand_point'] = data['Plant'] + '_' + data['Product'] + '_' + data['Plant Location']
        demand_points = data['demand_point'].unique()
        suppliers = data['Supplier'].unique()
        
        logger.info(f"📊 Found {len(demand_points)} demand points and {len(suppliers)} suppliers")
        
        # Decision variables: continuous allocation percentage (0.0 to 1.0) for each demand-supplier combination
        allocation_vars = {}
        
        for _, row in data.iterrows():
            var_name = f"alloc_{row['opt_index']}"
            allocation_vars[row['opt_index']] = self.solver.NumVar(0.0, 1.0, var_name)
        
        logger.info(f"✓ Created {len(allocation_vars)} continuous allocation variables (supports multi-supplier splits)")
        
        # Objective: Minimize total cost (this will find the Aunt Smith vs Aunt Bethany savings!)
        objective = self.solver.Objective()
        for _, row in data.iterrows():
            cost_per_unit = row['DDP (USD)']
            volume = row['2024 Volume (lbs)']
            total_cost = cost_per_unit * volume
            objective.SetCoefficient(allocation_vars[row['opt_index']], total_cost)
        objective.SetMinimization()
        
        logger.info("🎯 Objective: Minimize total procurement cost")
        
        # Constraint 1: Each demand point allocations must sum to exactly 1.0 (100% allocated)
        demand_point_constraints = 0
        for demand_point in demand_points:
            demand_data = data[data['demand_point'] == demand_point]
            if len(demand_data) >= 1:
                # Allocations for this demand point must sum to exactly 1.0 (allows multi-supplier splits)
                demand_constraint = self.solver.Constraint(1.0, 1.0)
                for _, row in demand_data.iterrows():
                    demand_constraint.SetCoefficient(allocation_vars[row['opt_index']], 1.0)
                demand_point_constraints += 1
        
        logger.info(f"✓ Added {demand_point_constraints} demand point constraints (100% allocation, supports multi-supplier splits)")
        
        # Constraint 2: Supplier capacity constraints
        supplier_constraints_added = 0
        for supplier in suppliers:
            if supplier in supplier_constraints:
                constraint_info = supplier_constraints[supplier]
                supplier_data = data[data['Supplier'] == supplier]
                
                if len(supplier_data) > 0:
                    # Add supplier min/max capacity constraint
                    supplier_constraint = self.solver.Constraint(
                        constraint_info['min'], constraint_info['max']
                    )
                    
                    for _, row in supplier_data.iterrows():
                        volume = row['2024 Volume (lbs)']
                        supplier_constraint.SetCoefficient(allocation_vars[row['opt_index']], volume)
                    
                    supplier_constraints_added += 1
                    logger.info(f"  {supplier}: {constraint_info['min']:,.0f} - {constraint_info['max']:,.0f} lbs")
        
        logger.info(f"✓ Added {supplier_constraints_added} supplier capacity constraints")
        
        # Constraint 3: Company volume limit (total procurement = company limit exactly)
        if company_volume_limit:
            company_constraint = self.solver.Constraint(company_volume_limit, company_volume_limit)
            for _, row in data.iterrows():
                volume = row['2024 Volume (lbs)']
                company_constraint.SetCoefficient(allocation_vars[row['opt_index']], volume)
            
            logger.info(f"✓ Added company volume constraint: exactly {company_volume_limit:,.0f} lbs")
        
        
        # Solve
        logger.info("🎯 Solving enhanced optimization problem...")
        status = self.solver.Solve()
        
        return self._extract_results_with_company_constraint(
            data, allocation_vars, supplier_constraints, 
            company_volume_limit, status, []
        )
    
    def _solve_standard_optimization(self, data: pd.DataFrame,
                                   supplier_constraints: Dict) -> OptimizationResult:
        """Solve standard optimization without company constraint"""
        
        logger.info("Solving standard supply chain optimization")
        
        self.solver.Clear()
        
        # Similar to company constraint version but without company volume limit
        plants = data['Plant'].unique()
        suppliers = data['Supplier'].unique()
        
        allocation_vars = {}
        for _, row in data.iterrows():
            var_name = f"alloc_{row['opt_index']}"
            allocation_vars[row['opt_index']] = self.solver.NumVar(0.0, 1.0, var_name)
        
        # Objective: Minimize total cost
        objective = self.solver.Objective()
        for _, row in data.iterrows():
            cost_per_unit = row['DDP (USD)']
            volume = row['2024 Volume (lbs)']
            total_cost = cost_per_unit * volume
            objective.SetCoefficient(allocation_vars[row['opt_index']], total_cost)
        objective.SetMinimization()
        
        # Plant allocation constraints
        for plant in plants:
            plant_data = data[data['Plant'] == plant]
            if len(plant_data) > 0:
                plant_constraint = self.solver.Constraint(1.0, 1.0)
                for _, row in plant_data.iterrows():
                    plant_constraint.SetCoefficient(allocation_vars[row['opt_index']], 1.0)
        
        # Supplier capacity constraints
        for supplier in suppliers:
            if supplier in supplier_constraints:
                constraint = supplier_constraints[supplier]
                supplier_data = data[data['Supplier'] == supplier]
                
                if len(supplier_data) > 0:
                    supplier_constraint = self.solver.Constraint(
                        constraint['min'], constraint['max']
                    )
                    
                    for _, row in supplier_data.iterrows():
                        volume = row['2024 Volume (lbs)']
                        supplier_constraint.SetCoefficient(allocation_vars[row['opt_index']], volume)
        
        # Solve
        status = self.solver.Solve()
        
        return self._extract_results_standard(data, allocation_vars, supplier_constraints, status)
    
    def _extract_results_with_company_constraint(self, data: pd.DataFrame,
                                               allocation_vars: Dict,
                                               supplier_constraints: Dict,
                                               company_volume_limit: float,
                                               status: int,
                                               constraint_violations: List[str]) -> OptimizationResult:
        """Extract results for company-constrained optimization"""
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return self._create_failure_result(f"Solver failed with status: {status}")
        
        # Extract solution
        results_data = []
        total_optimized_cost = 0
        total_baseline_cost = 0
        company_volume_used = 0
        
        # First, calculate the true baseline (current state) costs per demand point
        baseline_costs_by_demand = {}
        for _, row in data.iterrows():
            demand_point = row['demand_point']
            if row['Baseline Allocated Volume'] > 0:  # This is the currently selected supplier
                baseline_costs_by_demand[demand_point] = row['Baseline Price Paid']
        
        for _, row in data.iterrows():
            allocation_percentage = allocation_vars[row['opt_index']].solution_value()
            
            # Calculate optimized values
            optimized_volume = row['2024 Volume (lbs)'] * allocation_percentage
            optimized_cost = optimized_volume * row['DDP (USD)']
            
            # Get the true baseline cost for this demand point
            demand_point = row['demand_point']
            baseline_cost = baseline_costs_by_demand.get(demand_point, 0)
            
            # Create result row
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if allocation_percentage > self.tolerance else ''
            result_row['Optimized Split'] = f"{allocation_percentage * 100:.2f}%"
            result_row['Allocation_Percentage'] = allocation_percentage
            result_row['Column_O_Savings'] = baseline_cost - optimized_cost
            
            results_data.append(result_row)
            total_optimized_cost += optimized_cost
            company_volume_used += optimized_volume
        
        # Calculate total baseline cost properly (once per demand point)
        total_baseline_cost = sum(baseline_costs_by_demand.values())
        
        results_df = pd.DataFrame(results_data)
        
        total_savings = total_baseline_cost - total_optimized_cost
        savings_percentage = (total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        
        logger.info(f"Company volume used: {company_volume_used:,.0f} / {company_volume_limit:,.0f} lbs")
        logger.info(f"Total savings: ${total_savings:,.2f} ({savings_percentage:.2f}%)")
        
        return OptimizationResult(
            success=True,
            solver_status="OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
            results_dataframe=results_df,
            total_savings=total_savings,
            savings_percentage=savings_percentage,
            baseline_total_cost=total_baseline_cost,
            optimized_total_cost=total_optimized_cost,
            execution_time=0,  # Set by caller
            company_volume_used=company_volume_used,
            company_volume_limit=company_volume_limit,
            constraint_violations=constraint_violations
        )
    
    def _extract_results_standard(self, data: pd.DataFrame,
                                allocation_vars: Dict,
                                supplier_constraints: Dict,
                                status: int) -> OptimizationResult:
        """Extract results for standard optimization"""
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return self._create_failure_result(f"Solver failed with status: {status}")
        
        # Similar extraction logic but without company volume tracking
        results_data = []
        total_optimized_cost = 0
        total_baseline_cost = 0
        
        for _, row in data.iterrows():
            allocation_percentage = allocation_vars[row['opt_index']].solution_value()
            
            optimized_volume = row['2024 Volume (lbs)'] * allocation_percentage
            optimized_cost = optimized_volume * row['DDP (USD)']
            baseline_cost = row['baseline_cost_per_location']
            
            result_row = row.copy()
            result_row['Optimized Volume'] = optimized_volume
            result_row['Optimized Price'] = optimized_cost
            result_row['Optimized Selection'] = 'X' if allocation_percentage > self.tolerance else ''
            result_row['Optimized Split'] = f"{allocation_percentage * 100:.2f}%"
            result_row['Allocation_Percentage'] = allocation_percentage
            result_row['Column_O_Savings'] = baseline_cost - optimized_cost
            
            results_data.append(result_row)
            total_optimized_cost += optimized_cost
            total_baseline_cost += baseline_cost
        
        results_df = pd.DataFrame(results_data)
        
        total_savings = total_baseline_cost - total_optimized_cost
        savings_percentage = (total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
        
        return OptimizationResult(
            success=True,
            solver_status="OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
            results_dataframe=results_df,
            total_savings=total_savings,
            savings_percentage=savings_percentage,
            baseline_total_cost=total_baseline_cost,
            optimized_total_cost=total_optimized_cost,
            execution_time=0,
            company_volume_used=0,
            company_volume_limit=0,
            constraint_violations=[]
        )
    
    def _create_failure_result(self, error_message: str) -> OptimizationResult:
        """Create a failure result"""
        return OptimizationResult(
            success=False,
            solver_status="FAILED",
            results_dataframe=pd.DataFrame(),
            total_savings=0,
            savings_percentage=0,
            baseline_total_cost=0,
            optimized_total_cost=0,
            execution_time=0,
            company_volume_used=0,
            company_volume_limit=0,
            constraint_violations=[error_message]
        )


def test_ortools_optimizer():
    """Test the OR-Tools optimizer with demo data"""
    try:
        # Load demo data
        demo_data_path = './data/samples/demo_data.csv'
        df = pd.read_csv(demo_data_path)
        
        # Define supplier constraints for demo_data.csv
        supplier_constraints = {
            'Aunt Baker': {
                'min': 190_000_000,
                'max': 208_800_000
            },
            'Aunt Bethany': {
                'min': 2_400_000_000,
                'max': 2_758_300_000
            },
            'Aunt Celine': {
                'min': 500_000_000,
                'max': 628_200_000
            },
            'Aunt Smith': {
                'min': 2_000_000_000,
                'max': 2_507_100_000
            }
        }
        
        # Test with company volume constraint
        company_volume_limit = 5_400_000_000
        
        optimizer = ORToolsOptimizer()
        result = optimizer.optimize(df, supplier_constraints, {}, company_volume_limit)
        
        if result.success:
            print(f"✅ OR-Tools optimization successful!")
            print(f"Total savings: ${result.total_savings:,.2f}")
            print(f"Savings rate: {result.savings_percentage:.2f}%")
            print(f"Company volume used: {result.company_volume_used:,.0f} / {result.company_volume_limit:,.0f} lbs")
            
            # Save results
            result.results_dataframe.to_csv('./data/processed/ortools_test_results.csv', index=False)
            print("Results saved to ortools_test_results.csv")
        else:
            print(f"❌ Optimization failed: {result.constraint_violations}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_ortools_optimizer()