"""
COMPLETE SUPPLY CHAIN OPTIMIZATION WITH COMPANY BUYING LIMIT
============================================================
Solves the constrained optimization problem where:
- Total company buying limit: 5.4B lbs
- Individual supplier min/max constraints
- Minimize total cost while respecting all constraints

Uses OR-Tools for maximum mathematical rigor and performance.
"""

from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

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
    optimal_allocations: Dict[str, float]
    total_cost_savings: float
    total_allocation: float
    savings_rate: float
    cuts_by_supplier: Dict[str, float]
    constraint_satisfaction: bool

class CompanyLimitOptimizer:
    """
    Optimizes supplier allocation with company-wide buying limit constraint.
    
    This is a classic constrained optimization problem:
    Minimize: Σ(allocation_i * price_i)
    Subject to: 
    - Σ(allocation_i) ≤ company_buying_limit
    - min_i ≤ allocation_i ≤ max_i for all suppliers i
    """
    
    def __init__(self, company_buying_limit: float):
        self.company_buying_limit = company_buying_limit
        self.solver = None
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
                    print(f"✓ Using {solver_name} for optimization")
                    break
            except:
                continue
        
        if not self.solver:
            raise Exception("❌ No suitable solver found")
    
    def optimize_allocation(self, supplier_constraints: Dict[str, SupplierConstraint]) -> OptimizationResult:
        """
        Solve the constrained optimization problem using OR-Tools.
        
        Args:
            supplier_constraints: Dictionary mapping supplier names to their constraints
            
        Returns:
            OptimizationResult with optimal allocations and savings
        """
        print("🚀 Starting OR-Tools optimization...")
        
        self.solver.Clear()
        
        # ===== DECISION VARIABLES =====
        allocation_vars = {}
        suppliers = list(supplier_constraints.keys())
        
        for supplier in suppliers:
            constraint = supplier_constraints[supplier]
            var_name = f"allocation_{supplier}"
            allocation_vars[supplier] = self.solver.NumVar(
                constraint.min_capacity,
                constraint.max_capacity,
                var_name
            )
        
        print(f"✓ Created {len(allocation_vars)} allocation variables")
        
        # ===== OBJECTIVE FUNCTION: Minimize Total Cost =====
        objective = self.solver.Objective()
        for supplier in suppliers:
            price = supplier_constraints[supplier].avg_price
            objective.SetCoefficient(allocation_vars[supplier], price)
        objective.SetMinimization()
        
        print("✓ Objective: Minimize total procurement cost")
        
        # ===== CONSTRAINT: Total allocation ≤ Company buying limit =====
        total_constraint = self.solver.Constraint(0, self.company_buying_limit)
        for supplier in suppliers:
            total_constraint.SetCoefficient(allocation_vars[supplier], 1.0)
        
        print(f"✓ Added company buying limit constraint: ≤ {self.company_buying_limit:,.0f}")
        
        # ===== SOLVE =====
        print("🎯 Solving optimization problem...")
        status = self.solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            return self._extract_results(allocation_vars, supplier_constraints, "OPTIMAL")
        elif status == pywraplp.Solver.FEASIBLE:
            return self._extract_results(allocation_vars, supplier_constraints, "FEASIBLE")
        else:
            raise Exception(f"❌ Optimization failed with status: {status}")
    
    def _extract_results(self, allocation_vars: Dict, 
                        supplier_constraints: Dict[str, SupplierConstraint],
                        status: str) -> OptimizationResult:
        """Extract optimization results"""
        
        optimal_allocations = {}
        cuts_by_supplier = {}
        total_current_cost = 0
        total_optimal_cost = 0
        total_allocation = 0
        
        print(f"\n📊 OPTIMIZATION RESULTS ({status}):")
        print("=" * 60)
        
        for supplier, var in allocation_vars.items():
            constraint = supplier_constraints[supplier]
            optimal_allocation = var.solution_value()
            current_allocation = constraint.current_allocation
            cut = current_allocation - optimal_allocation
            
            current_cost = current_allocation * constraint.avg_price
            optimal_cost = optimal_allocation * constraint.avg_price
            savings = current_cost - optimal_cost
            
            optimal_allocations[supplier] = optimal_allocation
            cuts_by_supplier[supplier] = cut
            total_current_cost += current_cost
            total_optimal_cost += optimal_cost
            total_allocation += optimal_allocation
            
            print(f"{supplier}:")
            print(f"  Current: {current_allocation:,.0f} lbs → Optimal: {optimal_allocation:,.0f} lbs")
            print(f"  Cut: {cut:,.0f} lbs | Savings: ${savings:,.0f}")
        
        total_savings = total_current_cost - total_optimal_cost
        savings_rate = (total_savings / total_current_cost) * 100
        
        print(f"\n💰 SUMMARY:")
        print(f"Total allocation: {total_allocation:,.0f} lbs")
        print(f"Buying limit compliance: {total_allocation <= self.company_buying_limit}")
        print(f"Total savings: ${total_savings:,.0f}")
        print(f"Savings rate: {savings_rate:.1f}%")
        
        # Verify constraints
        constraint_satisfaction = (total_allocation <= self.company_buying_limit and
                                 all(supplier_constraints[s].min_capacity <= optimal_allocations[s] <= supplier_constraints[s].max_capacity 
                                     for s in optimal_allocations))
        
        return OptimizationResult(
            optimal_allocations=optimal_allocations,
            total_cost_savings=total_savings,
            total_allocation=total_allocation,
            savings_rate=savings_rate,
            cuts_by_supplier=cuts_by_supplier,
            constraint_satisfaction=constraint_satisfaction
        )

def greedy_optimization_fallback(company_buying_limit: float, 
                                supplier_constraints: Dict[str, SupplierConstraint]) -> OptimizationResult:
    """
    Fallback greedy algorithm if OR-Tools is not available.
    
    Mathematical approach: Cut from most expensive suppliers first.
    """
    print("🔄 Using greedy optimization algorithm...")
    
    # Calculate total current allocation and required reduction
    total_current = sum(c.current_allocation for c in supplier_constraints.values())
    required_reduction = total_current - company_buying_limit
    
    print(f"Required reduction: {required_reduction:,.0f} lbs")
    
    if required_reduction <= 0:
        print("✓ No reduction needed")
        return OptimizationResult(
            optimal_allocations={s: c.current_allocation for s, c in supplier_constraints.items()},
            total_cost_savings=0,
            total_allocation=total_current,
            savings_rate=0,
            cuts_by_supplier={s: 0 for s in supplier_constraints},
            constraint_satisfaction=True
        )
    
    # Sort suppliers by price (most expensive first)
    suppliers_by_price = sorted(
        supplier_constraints.items(),
        key=lambda x: x[1].avg_price,
        reverse=True
    )
    
    print("\\n🎯 Cutting sequence (most expensive first):")
    
    optimal_allocations = {}
    cuts_by_supplier = {}
    remaining_to_cut = required_reduction
    
    for supplier_name, constraint in suppliers_by_price:
        flexibility = constraint.current_allocation - constraint.min_capacity
        
        if remaining_to_cut > 0 and flexibility > 0:
            cut = min(remaining_to_cut, flexibility)
            optimal_allocation = constraint.current_allocation - cut
            remaining_to_cut -= cut
            
            print(f"  {supplier_name}: Cut {cut:,.0f} lbs (${constraint.avg_price:.2f}/lb)")
        else:
            cut = 0
            optimal_allocation = constraint.current_allocation
        
        optimal_allocations[supplier_name] = optimal_allocation
        cuts_by_supplier[supplier_name] = cut
    
    # Calculate savings
    total_current_cost = sum(c.current_allocation * c.avg_price for c in supplier_constraints.values())
    total_optimal_cost = sum(optimal_allocations[s] * supplier_constraints[s].avg_price 
                           for s in optimal_allocations)
    total_savings = total_current_cost - total_optimal_cost
    total_allocation = sum(optimal_allocations.values())
    
    return OptimizationResult(
        optimal_allocations=optimal_allocations,
        total_cost_savings=total_savings,
        total_allocation=total_allocation,
        savings_rate=(total_savings / total_current_cost) * 100,
        cuts_by_supplier=cuts_by_supplier,
        constraint_satisfaction=True
    )

def main_optimization(company_buying_limit: float = 5_400_000_000) -> OptimizationResult:
    """
    Main optimization function with your specific data.
    
    Args:
        company_buying_limit: Total company buying limit in lbs
        
    Returns:
        OptimizationResult with complete solution
    """
    
    print("🚀 SUPPLY CHAIN OPTIMIZATION WITH COMPANY BUYING LIMIT")
    print("=" * 70)
    print(f"Company buying limit: {company_buying_limit:,.0f} lbs")
    
    # Your specific supplier constraints
    supplier_constraints = {
        'Aunt Smith': SupplierConstraint(
            min_capacity=2_000_000_000,
            max_capacity=2_507_100_000,
            current_allocation=2_507_100_000,
            avg_price=6.50  # Cheapest supplier
        ),
        'Aunt Bethany': SupplierConstraint(
            min_capacity=2_400_000_000,
            max_capacity=2_758_300_000,
            current_allocation=2_758_300_000,
            avg_price=8.80  # Most expensive
        ),
        'Aunt Baker': SupplierConstraint(
            min_capacity=190_000_000,
            max_capacity=208_800_000,
            current_allocation=208_800_000,
            avg_price=7.20
        ),
        'Aunt Celine': SupplierConstraint(
            min_capacity=500_000_000,
            max_capacity=628_200_000,
            current_allocation=628_200_000,
            avg_price=7.80
        )
    }
    
    print("\\n📋 Supplier constraints loaded:")
    for supplier, constraint in supplier_constraints.items():
        print(f"  {supplier}: {constraint.min_capacity:,.0f} - {constraint.max_capacity:,.0f} lbs @ ${constraint.avg_price:.2f}")
    
    # Try OR-Tools optimization first, fallback to greedy if needed
    try:
        optimizer = CompanyLimitOptimizer(company_buying_limit)
        result = optimizer.optimize_allocation(supplier_constraints)
        print("✅ OR-Tools optimization completed successfully")
    except Exception as e:
        print(f"⚠️ OR-Tools failed ({e}), using greedy algorithm")
        result = greedy_optimization_fallback(company_buying_limit, supplier_constraints)
    
    # Display final results
    print("\\n" + "=" * 70)
    print("🎯 FINAL OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print("\\n📊 OPTIMAL ALLOCATIONS:")
    for supplier, allocation in result.optimal_allocations.items():
        cut = result.cuts_by_supplier[supplier]
        current = supplier_constraints[supplier].current_allocation
        print(f"  {supplier}: {allocation:,.0f} lbs (cut: {cut:,.0f})")
    
    print(f"\\n💰 FINANCIAL IMPACT:")
    print(f"  Total cost savings: ${result.total_cost_savings:,.0f}")
    print(f"  Savings rate: {result.savings_rate:.1f}%")
    print(f"  Total allocation: {result.total_allocation:,.0f} lbs")
    print(f"  Constraint compliance: {'✅' if result.constraint_satisfaction else '❌'}")
    
    return result

# CSV Integration Function
def optimize_from_csv(csv_file_path: str, company_buying_limit: float = 5_400_000_000) -> OptimizationResult:
    """
    Load data from CSV and run optimization.
    
    Args:
        csv_file_path: Path to your demo_data.csv file
        company_buying_limit: Total company buying limit
        
    Returns:
        OptimizationResult with complete solution
    """
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_file_path)
        print(f"✓ Loaded {len(df)} rows from {csv_file_path}")
        
        # Extract supplier constraints from data
        baseline_data = df[df['Baseline Allocated Volume'] > 0]
        
        supplier_constraints = {}
        for supplier in baseline_data['Supplier'].unique():
            supplier_data = baseline_data[baseline_data['Supplier'] == supplier]
            
            total_allocation = supplier_data['Baseline Allocated Volume'].sum()
            avg_price = supplier_data['DDP (USD)'].mean()
            
            # Use data to infer constraints (you may need to adjust these)
            supplier_constraints[supplier] = SupplierConstraint(
                min_capacity=total_allocation * 0.8,  # Assume 20% flexibility down
                max_capacity=total_allocation,
                current_allocation=total_allocation,
                avg_price=avg_price
            )
        
        print("✓ Extracted supplier constraints from CSV data")
        
        # Run optimization
        optimizer = CompanyLimitOptimizer(company_buying_limit)
        return optimizer.optimize_allocation(supplier_constraints)
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        print("Using default constraints instead...")
        return main_optimization(company_buying_limit)

if __name__ == "__main__":
    # Run the optimization with your specific data
    result = main_optimization()
    
    # Optional: Export results to JSON
    results_dict = {
        'optimal_allocations': result.optimal_allocations,
        'total_savings': result.total_cost_savings,
        'savings_rate': result.savings_rate,
        'cuts_by_supplier': result.cuts_by_supplier
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\\n📄 Results exported to optimization_results.json")
    
    # To use with your CSV file:
    # result = optimize_from_csv('demo_data.csv', company_buying_limit=5_400_000_000)