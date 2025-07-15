"""
Optimization Interface Component for Supply Chain Optimization Platform
Handles optimization execution and progress display.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from config import settings


class OptimizationInterface:
    """Handles optimization execution interface."""
    
    def __init__(self, optimizer, ortools_optimizer, airflow_client):
        """Initialize optimization interface."""
        self.optimizer = optimizer
        self.ortools_optimizer = ortools_optimizer
        self.airflow_client = airflow_client
    
    def render_optimization_interface(self, on_optimization_complete: Optional[Callable] = None):
        """
        Render optimization execution interface.
        
        Args:
            on_optimization_complete: Callback when optimization completes
        """
        st.markdown("## Run Optimization")
        
        if st.button("Run Optimization", key="run_optimization"):
            with st.spinner("Running optimization... This may take a few minutes."):
                result = self._execute_optimization()
                
                if result and on_optimization_complete:
                    on_optimization_complete(result)
                
                if result and result.get('success'):
                    st.success("Optimization completed successfully!")
                    st.info("Optimization complete! Check the **Results** tab to view detailed results, download data, and generate AI summaries.")
                else:
                    st.error("Optimization failed. Please check your constraints and try again.")
    
    def _execute_optimization(self) -> Optional[Dict[str, Any]]:
        """Execute optimization and return results."""
        try:
            # Trigger Airflow DAG
            dag_run_id = self.airflow_client.trigger_optimization_dag(
                st.session_state.file_hash
            )
            
            if not dag_run_id:
                return {'success': False, 'error': 'Failed to start optimization process'}
            
            # Create history entry
            history_entry = self._create_history_entry(dag_run_id)
            
            # Choose optimizer based on file type
            if settings.is_demo_file(getattr(st.session_state, 'uploaded_file_name', '')):
                results = self._run_ortools_optimization(dag_run_id)
            else:
                results = self._run_standard_optimization(dag_run_id)
            
            # Process results
            if results:
                processed_results = self._process_optimization_results(results, dag_run_id)
                history_entry.update({
                    'status': 'success',
                    'total_savings': processed_results['total_savings'],
                    'execution_time': processed_results['execution_time']
                })
                return {**processed_results, 'success': True, 'history_entry': history_entry}
            else:
                history_entry.update({'status': 'failed', 'total_savings': 0, 'execution_time': 0})
                return {'success': False, 'history_entry': history_entry}
                
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_history_entry(self, dag_run_id: str) -> Dict[str, Any]:
        """Create optimization history entry."""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dag_run_id': dag_run_id,
            'file_hash': st.session_state.file_hash,
            'status': 'running',
            'file_name': getattr(st.session_state, 'uploaded_file_name', 'Unknown'),
            'total_savings': None,
            'execution_time': None
        }
    
    def _run_ortools_optimization(self, dag_run_id: str) -> Optional[Dict[str, Any]]:
        """Run Enhanced Pyomo optimization for demo files."""
        st.info("🚀 Using Enhanced Pyomo optimizer with advanced mathematical optimization...")
        
        # Use the main Pyomo optimizer for all optimization tasks
        results = self.optimizer.optimize(
            st.session_state.data,
            st.session_state.supplier_constraints,
            st.session_state.plant_constraints
        )
        
        # Return Pyomo result in standard format
        if results and results.get('success'):
            return results
        else:
            error_msg = results.get('error', 'Unknown error') if results else 'No results returned'
            st.error(f"Optimization failed: {error_msg}")
            return None
    
    def _run_standard_optimization(self, dag_run_id: str) -> Optional[Dict[str, Any]]:
        """Run standard Pyomo optimization."""
        st.info("🔧 Using standard Pyomo optimizer...")
        
        return self.optimizer.optimize(
            st.session_state.data,
            st.session_state.supplier_constraints,
            st.session_state.plant_constraints
        )
    
    def _process_optimization_results(self, results: Dict[str, Any], dag_run_id: str) -> Dict[str, Any]:
        """Process optimization results into standardized format."""
        results_df = results.get('results_dataframe', None)
        
        # Calculate metrics from results dataframe
        if results_df is not None and not results_df.empty:
            changed_plants = self._find_plants_with_changes(results_df)
            supplier_changes = self._calculate_supplier_changes(results_df)
            volume_reallocations = self._calculate_volume_reallocations(results_df)
            
            plants_optimized = len(changed_plants)
            suppliers_involved = len([s for s, data in supplier_changes.items() 
                                    if abs(data['optimized_volume'] - data['baseline_volume']) > 0])
            volume_reallocated = sum([abs(r['optimized_volume'] - r['baseline_volume']) 
                                    for r in volume_reallocations])
            supplier_switches = len(volume_reallocations)
        else:
            plants_optimized = 0
            suppliers_involved = 0
            volume_reallocated = 0
            supplier_switches = 0
        
        return {
            'total_savings': results.get('total_savings', 0),
            'savings_percentage': results.get('savings_percentage', 0),
            'baseline_cost': results.get('baseline_total_cost', 0),
            'optimized_cost': results.get('optimized_total_cost', 0),
            'optimized_data': results_df,
            'execution_time': results.get('execution_time', 0),
            'dag_run_id': dag_run_id,
            'detailed_results': results.get('detailed_results', []),
            'plants_optimized': plants_optimized,
            'suppliers_involved': suppliers_involved,
            'volume_redistributions': len(results.get('detailed_results', [])),
            'supplier_switches': supplier_switches,
            'volume_reallocated': volume_reallocated
        }
    
    def _find_plants_with_changes(self, results_df) -> Dict:
        """Find plants that had allocation changes."""
        changed_plants = {}
        
        results_df['demand_point'] = results_df['Plant'] + '_' + results_df['Product'] + '_' + results_df['Plant Location']
        
        for demand_point in results_df['demand_point'].unique():
            demand_data = results_df[results_df['demand_point'] == demand_point]
            changes_in_demand_point = []
            
            for _, row in demand_data.iterrows():
                baseline_vol = row.get('Baseline Allocated Volume', 0)
                optimized_vol = row.get('Optimized Volume', 0)
                
                if pd.isna(baseline_vol):
                    baseline_vol = 0
                if pd.isna(optimized_vol):
                    optimized_vol = 0
                
                if abs(baseline_vol - optimized_vol) > 0.01:
                    changes_in_demand_point.append(row)
            
            if changes_in_demand_point:
                plant_name = demand_data.iloc[0]['Plant']
                if plant_name not in changed_plants:
                    changed_plants[plant_name] = []
                
                baseline_rows = demand_data[demand_data['Baseline Allocated Volume'] > 0]
                optimized_rows = demand_data[demand_data['Optimized Volume'] > 0]
                
                changed_plants[plant_name].append({
                    'demand_point': demand_point,
                    'baseline_rows': baseline_rows,
                    'optimized_rows': optimized_rows,
                    'all_changes': changes_in_demand_point
                })
        
        return changed_plants
    
    def _calculate_supplier_changes(self, results_df) -> Dict:
        """Calculate volume and cost changes by supplier."""
        supplier_summary = {}
        
        for supplier in results_df['Supplier'].unique():
            supplier_data = results_df[results_df['Supplier'] == supplier]
            
            baseline_volume = supplier_data.get('Baseline Allocated Volume', pd.Series(0)).sum()
            optimized_volume = supplier_data.get('Optimized Volume', pd.Series(0)).sum()
            baseline_cost = supplier_data.get('Baseline Price Paid', pd.Series(0)).sum()
            optimized_cost = supplier_data.get('Optimized Price', pd.Series(0)).sum()
            
            if baseline_volume > 0 or optimized_volume > 0:
                supplier_summary[supplier] = {
                    'baseline_volume': baseline_volume,
                    'optimized_volume': optimized_volume,
                    'baseline_cost': baseline_cost,
                    'optimized_cost': optimized_cost
                }
        
        return supplier_summary
    
    def _calculate_volume_reallocations(self, results_df) -> list:
        """Calculate volume reallocations by supplier-plant combination."""
        reallocations = []
        
        for _, row in results_df.iterrows():
            baseline_volume = row.get('Baseline Allocated Volume', 0)
            optimized_volume = row.get('Optimized Volume', 0)
            
            if abs(baseline_volume - optimized_volume) > 0.01:
                baseline_cost = row.get('Baseline Price Paid', 0) if baseline_volume > 0 else 0
                optimized_cost = row.get('Optimized Price', 0) if optimized_volume > 0 else 0
                
                reallocations.append({
                    'plant': row['Plant'],
                    'location': row.get('Plant Location', ''),
                    'supplier': row['Supplier'],
                    'baseline_volume': baseline_volume,
                    'optimized_volume': optimized_volume,
                    'cost_change': optimized_cost - baseline_cost
                })
        
        return reallocations