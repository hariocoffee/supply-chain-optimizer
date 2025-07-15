"""
Constraints Configuration Component for Supply Chain Optimization Platform
Handles constraint detection, display, and user input interface.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Callable

from config import settings
from services import DataProcessor, ConstraintAnalysis


class ConstraintsComponent:
    """Handles constraints configuration interface."""
    
    def __init__(self, data_processor: DataProcessor):
        """Initialize constraints component."""
        self.data_processor = data_processor
    
    def render_constraints_interface(self, data: pd.DataFrame, filename: str = "",
                                   on_constraints_applied: Optional[Callable] = None):
        """
        Render complete constraints configuration interface.
        
        Args:
            data: DataFrame with optimization data
            filename: Original filename for constraint analysis
            on_constraints_applied: Callback when constraints are applied
        """
        st.markdown("## Automatically Detected Constraints")
        
        # Analyze constraints using data processor
        constraint_analysis = self.data_processor.analyze_constraints(data, filename)
        supplier_constraints = constraint_analysis.supplier_constraints
        
        if not supplier_constraints:
            st.warning("Unable to detect supplier constraints. Please check your data format.")
            return
        
        # Display constraint information
        self._render_constraint_summary(constraint_analysis)
        
        # Show auto-detected constraints table
        self._render_auto_detected_constraints(supplier_constraints)
        
        # User-editable constraints interface
        edited_constraints = self._render_editable_constraints(
            supplier_constraints, constraint_analysis
        )
        
        # Plant requirements section
        self._render_plant_requirements(data)
        
        # Company volume requirements (for demo files)
        company_volume_limit = self._render_company_volume_requirements(
            constraint_analysis, edited_constraints
        )
        
        # Apply constraints button
        if st.button("Apply Constraints", key="set_constraints"):
            if on_constraints_applied:
                on_constraints_applied(edited_constraints, company_volume_limit)
            st.success("Supplier constraints applied successfully!")
    
    def _render_constraint_summary(self, constraint_analysis: ConstraintAnalysis):
        """Render constraint analysis summary."""
        if constraint_analysis.file_type == 'demo_data':
            st.info("🎯 Demo data detected! Auto-filling predefined supplier constraints for OR-Tools optimization.")
        elif constraint_analysis.file_type == 'DEMO':
            st.info("📊 DEMO.csv detected! Using dynamic data-agnostic constraints from Column G (max volume) and Column H (total cost).")
        else:
            st.info("📋 Standard data format detected. Using baseline allocation for constraint calculation.")
    
    def _render_auto_detected_constraints(self, supplier_constraints: Dict[str, Dict[str, Any]]):
        """Render auto-detected constraints table."""
        st.markdown("### Supplier Constraints")
        st.markdown("**Auto-Detected Constraints (Reference):**")
        
        constraint_data = []
        for supplier, constraints in supplier_constraints.items():
            baseline_volume = constraints['baseline_volume']
            baseline_cost = constraints['baseline_cost']
            constraint_data.append({
                'Supplier': supplier,
                'Baseline Volume (lbs)': f"{baseline_volume:,.0f}",
                'Baseline Cost ($)': f"${baseline_cost:,.2f}",
                'Auto Min (lbs)': f"{constraints['min']:,.0f}",
                'Auto Max (lbs)': f"{constraints['display_max']:,.0f}"
            })
        
        constraints_df = pd.DataFrame(constraint_data)
        st.dataframe(constraints_df, use_container_width=True)
    
    def _render_editable_constraints(self, supplier_constraints: Dict[str, Dict[str, Any]],
                                   constraint_analysis: ConstraintAnalysis) -> Dict[str, Dict[str, Any]]:
        """Render editable constraints interface."""
        st.markdown("**Edit Supplier Constraints:**")
        st.info("Modify the minimum and maximum volume constraints for each supplier. Leave unchanged to use auto-detected values.")
        
        # Apply demo-specific constraints if applicable
        if constraint_analysis.file_type == 'DEMO':
            st.warning("⚠️ **FOR TESTING PURPOSES ONLY** - Auto-populated minimum constraints for DEMO.csv")
            self._apply_demo_constraints(supplier_constraints)
        
        edited_constraints = {}
        
        for supplier, constraints in supplier_constraints.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{supplier}**")
                st.caption(f"Baseline: {constraints['baseline_volume']:,.0f} lbs")
            
            with col2:
                min_val = st.number_input(
                    f"Min Volume (lbs)",
                    min_value=0,
                    value=int(constraints['min']),
                    step=1000,
                    key=f"min_{supplier}",
                    help=f"Minimum volume for {supplier}"
                )
            
            with col3:
                max_val = st.number_input(
                    f"Max Volume (lbs)",
                    min_value=min_val,
                    value=int(constraints['display_max']),
                    step=1000,
                    key=f"max_{supplier}",
                    help=f"Maximum volume for {supplier}"
                )
            
            # Use flexible optimization constraint unless user changed it
            optimization_max = constraints['max'] if max_val == int(constraints['display_max']) else max_val
            
            edited_constraints[supplier] = {
                'min': min_val,
                'max': optimization_max,
                'display_max': max_val,
                'baseline_volume': constraints['baseline_volume'],
                'baseline_cost': constraints['baseline_cost']
            }
        
        return edited_constraints
    
    def _apply_demo_constraints(self, supplier_constraints: Dict[str, Dict[str, Any]]):
        """Apply demo-specific constraints from settings."""
        for supplier, min_val in settings.constraints.demo_min_constraints.items():
            if supplier in supplier_constraints:
                supplier_constraints[supplier]['min'] = min_val
                if supplier in settings.constraints.demo_max_constraints:
                    supplier_constraints[supplier]['max'] = settings.constraints.demo_max_constraints[supplier]
                    supplier_constraints[supplier]['display_max'] = settings.constraints.demo_max_constraints[supplier]
    
    def _render_plant_requirements(self, data: pd.DataFrame):
        """Render plant requirements section."""
        st.markdown("### Plant Requirements")
        st.info("**Each plant-product-location combination MUST receive its full allocated volume as specified in Column C (2024 Volume).**")
        
        plant_requirements = data.groupby(['Plant', 'Product', 'Plant Location'])['2024 Volume (lbs)'].first().reset_index()
        plant_requirements['Required Volume (lbs)'] = plant_requirements['2024 Volume (lbs)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(plant_requirements[['Plant', 'Product', 'Plant Location', 'Required Volume (lbs)']], use_container_width=True)
    
    def _render_company_volume_requirements(self, constraint_analysis: ConstraintAnalysis,
                                          edited_constraints: Dict[str, Dict[str, Any]]) -> Optional[int]:
        """Render company volume requirements for demo files."""
        if not constraint_analysis.is_demo_file:
            return None
        
        st.markdown("### Company Volume Requirement")
        st.info("🏭 **Company Volume Constraint**: Specify the total volume your company can purchase from all suppliers.")
        
        # Calculate constraint bounds
        min_total = sum(constraints['min'] for constraints in edited_constraints.values())
        max_total = sum(constraints['max'] for constraints in edited_constraints.values())
        
        # Set default value based on file type
        if constraint_analysis.file_type == 'DEMO':
            default_value = settings.constraints.default_company_volume_demo
        else:
            default_value = settings.constraints.default_company_volume_standard
        
        col1, col2 = st.columns([2, 1])
        with col1:
            company_volume_limit = st.number_input(
                f"Company buying limit (lbs)",
                min_value=int(min_total),
                max_value=int(max_total),
                value=default_value,
                step=10_000_000,
                help=f"Must be between {min_total:,.0f} (sum of minimums) and {max_total:,.0f} (sum of maximums) lbs"
            )
        
        with col2:
            st.metric("Constraint Range", f"{min_total:,.0f} - {max_total:,.0f}")
        
        # Validation
        if company_volume_limit < min_total or company_volume_limit > max_total:
            st.error(f"⚠️ Company volume must be between {min_total:,.0f} and {max_total:,.0f} lbs")
            return None
        
        return company_volume_limit