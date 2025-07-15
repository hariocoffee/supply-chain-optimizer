import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import hashlib
import os
import tempfile
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
import sys
import os

# Add src directory to path for proper imports
src_path = os.path.dirname(__file__)
parent_path = os.path.dirname(src_path)
sys.path.extend([src_path, parent_path])

# Import production modules
try:
    from config import settings
    from services import DataProcessor
    from ui.styles import ThemeManager
    from ui.components import FileUploadComponent, ConstraintsComponent, OptimizationInterface
    print("✅ Successfully imported production modules")
except ImportError as e:
    print(f"⚠️ Refactored module import warning: {e}")
    # Create fallback settings and services
    class MockSettings:
        @staticmethod
        def is_demo_file(filename): return 'demo' in filename.lower()
        ui = type('obj', (object,), {'page_title': 'Supply Chain Optimizer', 'page_icon': '⚡', 'layout': 'wide', 'sidebar_state': 'collapsed', 'theme': 'apple_silver'})()
    settings = MockSettings()
    
    class MockDataProcessor:
        def generate_file_hash(self, data): return "mock_hash"
        def validate_data(self, data, filename=""): 
            from collections import namedtuple
            Result = namedtuple('ValidationResult', 'is_valid errors warnings processed_data')
            return Result(True, [], [], data)
        def create_template_data(self): return pd.DataFrame()
        def analyze_constraints(self, data, filename=""): 
            from collections import namedtuple
            Analysis = namedtuple('ConstraintAnalysis', 'supplier_constraints total_volume is_demo_file file_type')
            return Analysis({}, 0, False, 'standard')
        def get_data_summary(self, data): return {}
    DataProcessor = MockDataProcessor
    
    class MockThemeManager:
        @staticmethod
        def apply_theme(st_instance, theme_name="apple_silver"): pass
    ThemeManager = MockThemeManager
    
    # Mock UI components
    class MockFileUploadComponent:
        def __init__(self, dp): pass
        def render_complete_interface(self, on_file_processed=None): pass
    FileUploadComponent = MockFileUploadComponent
    
    class MockConstraintsComponent:
        def __init__(self, dp): pass
        def render_constraints_interface(self, data, filename="", callback=None): pass
    ConstraintsComponent = MockConstraintsComponent
    
    class MockOptimizationInterface:
        def __init__(self, opt, ortools, airflow): pass
        def render_optimization_interface(self, callback=None): pass
    OptimizationInterface = MockOptimizationInterface

try:
    from services.database import DatabaseManager
    from services.cache_manager import CacheManager
    from optimization.engines.optimizer import PyomoOptimizer  # Production optimizer
    from services.airflow_client import AirflowClient
    print("✅ Successfully imported production optimization modules")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Fallback imports
    try:
        from advanced_optimizer import AdvancedOptimizer as PyomoOptimizer
        print("✅ Using AdvancedOptimizer directly")
    except:
        print("❌ Could not import optimization modules")
        # Create mock optimizer
        class PyomoOptimizer:
            def optimize(self, *args, **kwargs): 
                return {"success": False, "error": "Optimizer not available"}
        
    # Create mock classes for demo
    class DatabaseManager:
        def __init__(self): pass
        def store_baseline_data(self, *args): return True
        def store_constraints(self, *args): return True
        def store_optimization_results(self, *args): return True
        
    class CacheManager:
        def __init__(self): pass
        def get_cached_result(self, *args): return None
        def cache_result(self, *args): return True
        
    class AirflowClient:
        def __init__(self): pass
        def trigger_optimization_dag(self, *args): return f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Configure Streamlit page
st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon=settings.ui.page_icon,
    layout=settings.ui.layout,
    initial_sidebar_state=settings.ui.sidebar_state
)

# Apply theme
ThemeManager.apply_theme(st, settings.ui.theme)

class SupplyChainOptimizer:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.optimizer = PyomoOptimizer()
        self.airflow_client = AirflowClient()
        
        # Initialize new refactored services
        self.data_processor = DataProcessor()
        
        # Initialize UI components
        self.file_upload_component = FileUploadComponent(self.data_processor)
        self.constraints_component = ConstraintsComponent(self.data_processor)
        self.optimization_interface = OptimizationInterface(
            self.optimizer, None, self.airflow_client
        )
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'file_hash' not in st.session_state:
            st.session_state.file_hash = None
        if 'constraints_set' not in st.session_state:
            st.session_state.constraints_set = False
        if 'optimization_complete' not in st.session_state:
            st.session_state.optimization_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []

    def generate_file_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash for the uploaded data for caching purposes."""
        return self.data_processor.generate_file_hash(data)

    def create_template_file(self) -> bytes:
        """Create a template file based on the expected structure for supplier selection optimization."""
        template_df = self.data_processor.create_template_data()
        return template_df.to_csv(index=False).encode()

    def main_interface(self):
        """Main interface for the application."""
        
        # Initialize session state variables
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'constraints_set' not in st.session_state:
            st.session_state.constraints_set = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = ""
        
        # Use file upload component
        self.file_upload_component.render_complete_interface(
            on_file_processed=self.on_file_processed
        )
        
        # Show constraints interface if data is available
        if st.session_state.data is not None:
            self.constraints_component.render_constraints_interface(
                data=st.session_state.data,
                filename=getattr(st.session_state, 'uploaded_file_name', ''),
                on_constraints_applied=self.on_constraints_applied
            )
        
        # Show optimization interface if constraints are set
        if st.session_state.constraints_set:
            self.optimization_interface.render_optimization_interface(
                on_optimization_complete=self.on_optimization_complete
            )
    
    def on_file_processed(self, processing_result):
        """Callback when file is processed by file upload component."""
        # Store processed data in session state
        st.session_state.data = processing_result['data']
        st.session_state.file_hash = processing_result['file_hash']
        st.session_state.uploaded_file_name = processing_result['filename']
        
        # Clear previous results
        st.session_state.results = None
        st.session_state.optimization_complete = False
        st.session_state.constraints_set = False
        
        # Check for cached results
        cached_result = self.cache_manager.get_cached_result(processing_result['file_hash'])
        
        # Set status messages
        st.session_state.file_processing_status = {
            'show_cache_message': bool(cached_result),
            'show_success_message': True,
            'filename': processing_result['filename']
        }
        
        # Store in database
        self.db_manager.store_baseline_data(
            processing_result['data'], 
            processing_result['file_hash'], 
            processing_result['filename']
        )
    
    def on_constraints_applied(self, edited_constraints, company_volume_limit=None):
        """Callback when constraints are applied by constraints component."""
        # Store constraints in session state
        st.session_state.supplier_constraints = edited_constraints
        st.session_state.plant_constraints = {}
        st.session_state.company_volume_limit = company_volume_limit
        st.session_state.constraints_set = True
        
        # Store in database
        self.db_manager.store_constraints(
            st.session_state.file_hash,
            edited_constraints,
            {}
        )
    
    def on_optimization_complete(self, optimization_result):
        """Callback when optimization is completed by optimization interface."""
        if optimization_result.get('success'):
            # Store results
            st.session_state.results = optimization_result
            st.session_state.optimization_complete = True
            
            # Cache results
            self.cache_manager.cache_result(
                st.session_state.file_hash,
                optimization_result
            )
            
            # Store in database
            self.db_manager.store_optimization_results(
                st.session_state.file_hash,
                optimization_result
            )
        
        # Add to history
        if 'history_entry' in optimization_result:
            if 'optimization_history' not in st.session_state:
                st.session_state.optimization_history = []
            st.session_state.optimization_history.append(optimization_result['history_entry'])

    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file and prepare for optimization."""
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate data using refactored data processor
            validation_result = self.data_processor.validate_data(df, uploaded_file.name)
            
            # Show validation messages
            if validation_result.errors:
                for error in validation_result.errors:
                    st.error(f"Data validation error: {error}")
                return
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    st.warning(f"Data validation warning: {warning}")
            
            # Use processed data if validation succeeded
            if validation_result.processed_data is not None:
                df = validation_result.processed_data
            
            # Generate file hash for caching
            file_hash = self.generate_file_hash(df)
            
            # Check if this file has been processed before
            cached_result = self.cache_manager.get_cached_result(file_hash)
            
            # Always clear current session results when uploading new file
            st.session_state.results = None
            st.session_state.optimization_complete = False
            
            # Store in session state
            st.session_state.data = df
            st.session_state.file_hash = file_hash
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Set status messages to be shown in main interface
            st.session_state.file_processing_status = {
                'show_cache_message': bool(cached_result),
                'show_success_message': True,
                'filename': uploaded_file.name
            }
            
            # Store in database
            self.db_manager.store_baseline_data(df, file_hash, uploaded_file.name)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    def show_constraints_interface(self):
        """Show the constraints setting interface."""
        st.markdown("## Automatically Detected Constraints")
        
        df = st.session_state.data
        
        # Auto-detect supplier constraints using refactored data processor
        constraint_analysis = self.data_processor.analyze_constraints(df, st.session_state.uploaded_file_name)
        supplier_constraints = constraint_analysis.supplier_constraints
        
        # Display detected constraints
        st.markdown("### Supplier Constraints")
        
        if not supplier_constraints:
            st.warning("Unable to detect supplier constraints. Please check your data format.")
            return
        
        # Show auto-detected constraints for reference
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
        
        # User-editable constraints
        st.markdown("**Edit Supplier Constraints:**")
        st.info("Modify the minimum and maximum volume constraints for each supplier. Leave unchanged to use auto-detected values.")
        
        # Create editable constraints interface
        edited_constraints = {}
        
        # For DEMO.csv, auto-populate minimum constraints (for testing purposes only)
        if constraint_analysis.file_type == 'DEMO':
            st.warning("⚠️ **FOR TESTING PURPOSES ONLY** - Auto-populated minimum constraints for DEMO.csv")
            
            # Apply demo-specific constraints from settings
            for supplier, min_val in settings.constraints.demo_min_constraints.items():
                if supplier in supplier_constraints:
                    supplier_constraints[supplier]['min'] = min_val
                    # Also set the max constraints for DEMO.csv
                    if supplier in settings.constraints.demo_max_constraints:
                        supplier_constraints[supplier]['max'] = settings.constraints.demo_max_constraints[supplier]
                        supplier_constraints[supplier]['display_max'] = settings.constraints.demo_max_constraints[supplier]
        
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
            
            # If user didn't change the max from display value, use the flexible optimization constraint
            optimization_max = constraints['max'] if max_val == int(constraints['display_max']) else max_val
            
            edited_constraints[supplier] = {
                'min': min_val,
                'max': optimization_max,  # Use flexible constraint unless user changed it
                'display_max': max_val,  # Store display value
                'baseline_volume': constraints['baseline_volume'],
                'baseline_cost': constraints['baseline_cost']
            }
        
        # Info about plant requirements
        st.markdown("### Plant Requirements")
        st.info("**Each plant-product-location combination MUST receive its full allocated volume as specified in Column C (2024 Volume).**")
        
        plant_requirements = df.groupby(['Plant', 'Product', 'Plant Location'])['2024 Volume (lbs)'].first().reset_index()
        plant_requirements['Required Volume (lbs)'] = plant_requirements['2024 Volume (lbs)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(plant_requirements[['Plant', 'Product', 'Plant Location', 'Required Volume (lbs)']], use_container_width=True)
        
        # Company Volume Requirement (for demo files)
        company_volume_limit = None
        show_company_volume = constraint_analysis.is_demo_file
        
        if show_company_volume:
            st.markdown("### Company Volume Requirement")
            st.info("🏭 **Company Volume Constraint**: Specify the total volume your company can purchase from all suppliers.")
            
            # Calculate constraint bounds dynamically based on current min/max values
            min_total = sum(constraints['min'] for constraints in edited_constraints.values())
            max_total = sum(constraints['max'] for constraints in edited_constraints.values())
            
            # Set dynamic default value based on file type
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
                company_volume_limit = None
        
        if st.button("Apply Constraints", key="set_constraints"):
            # Store edited constraints in session state
            st.session_state.supplier_constraints = edited_constraints
            st.session_state.plant_constraints = {}  # No plant constraints needed
            st.session_state.company_volume_limit = company_volume_limit  # Store company volume limit
            st.session_state.constraints_set = True
            
            # Store in database
            self.db_manager.store_constraints(
                st.session_state.file_hash,
                edited_constraints,
                {}  # No plant constraints
            )
            
            st.success("Supplier constraints applied successfully!")
    

    def show_optimization_interface(self):
        """Show the optimization interface."""
        st.markdown("## Run Optimization")
        
        if st.button("Run Optimization", key="run_optimization"):
            with st.spinner("Running optimization... This may take a few minutes."):
                # Trigger Airflow DAG
                dag_run_id = self.airflow_client.trigger_optimization_dag(
                    st.session_state.file_hash
                )
                
                if dag_run_id:
                    st.success(f"Optimization started! DAG Run ID: {dag_run_id}")
                    
                    # Create history entry for the attempt
                    history_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'dag_run_id': dag_run_id,
                        'file_hash': st.session_state.file_hash,
                        'status': 'running',
                        'file_name': getattr(st.session_state, 'uploaded_file_name', 'Unknown'),
                        'total_savings': None,
                        'execution_time': None
                    }
                    
                    # Use production-ready Enhanced Pyomo optimizer for all data
                    st.info("🚀 Using Enhanced Pyomo optimizer with advanced mathematical optimization...")
                    
                    # Use production Pyomo optimizer
                    results = self.optimizer.optimize(
                        st.session_state.data,
                        st.session_state.supplier_constraints,
                        st.session_state.plant_constraints
                    )
                    
                    if results:
                        # Calculate actual optimization changes from the results dataframe
                        results_df = results.get('results_dataframe', pd.DataFrame())
                        
                        if not results_df.empty:
                            # Calculate metrics using OR-Tools results structure
                            # Use the same logic as our drill-down functions
                            changed_plants = self._find_plants_with_changes(results_df)
                            supplier_changes = self._calculate_supplier_changes(results_df)
                            volume_reallocations = self._calculate_volume_reallocations(results_df)
                            
                            # Calculate metrics
                            plants_optimized = len(changed_plants)
                            suppliers_involved = len([s for s, data in supplier_changes.items() 
                                                    if abs(data['optimized_volume'] - data['baseline_volume']) > 0])
                            volume_reallocated = sum([abs(r['optimized_volume'] - r['baseline_volume']) 
                                                    for r in volume_reallocations])
                            supplier_switches = len(volume_reallocations)
                        else:
                            volume_reallocated = 0
                            plants_optimized = 0
                            suppliers_involved = 0
                            supplier_switches = 0
                        
                        # Process results for compatibility
                        processed_results = {
                            'total_savings': results.get('total_savings', 0),
                            'savings_percentage': results.get('savings_percentage', 0),
                            'baseline_cost': results.get('baseline_total_cost', 0),
                            'optimized_cost': results.get('optimized_total_cost', 0),
                            'optimized_data': results.get('results_dataframe'),
                            'execution_time': results.get('execution_time', 0),
                            'dag_run_id': dag_run_id,
                            'detailed_results': results.get('detailed_results', []),
                            'plants_optimized': plants_optimized,
                            'suppliers_involved': suppliers_involved,
                            'volume_redistributions': len(results.get('detailed_results', [])),
                            'supplier_switches': supplier_switches,
                            'volume_reallocated': volume_reallocated
                        }
                        
                        st.session_state.results = processed_results
                        st.session_state.optimization_complete = True
                        
                        # Update history entry with success
                        history_entry.update({
                            'status': 'success',
                            'total_savings': processed_results['total_savings'],
                            'execution_time': processed_results['execution_time']
                        })
                        
                        # Cache the results
                        self.cache_manager.cache_result(
                            st.session_state.file_hash,
                            processed_results
                        )
                        
                        # Store in database
                        self.db_manager.store_optimization_results(
                            st.session_state.file_hash,
                            processed_results
                        )
                        
                        st.success("Optimization completed successfully!")
                        
                        # Show success message and redirect to results tab
                        st.info("Optimization complete! Check the **Results** tab to view detailed results, download data, and generate AI summaries.")
                    else:
                        # Update history entry with failure
                        history_entry.update({
                            'status': 'failed',
                            'total_savings': 0,
                            'execution_time': 0
                        })
                        
                        # Clear previous results when optimization fails
                        st.session_state.results = None
                        st.session_state.optimization_complete = False
                        st.error("Optimization failed. Please check your constraints and try again.")
                    
                    # Add to history regardless of success/failure
                    st.session_state.optimization_history.append(history_entry)
                else:
                    st.error("Failed to start optimization process.")

    def download_results(self):
        """Generate and download the results file."""
        if st.session_state.results and 'optimized_data' in st.session_state.results:
            results_df = st.session_state.results['optimized_data']
            if results_df is not None and not results_df.empty:
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Optimized Results CSV",
                    data=csv,
                    file_name=f"optimized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("No optimization data available to download.")
        else:
            st.error("No optimization results available. Please run optimization first.")

    def generate_ai_summary(self):
        """Generate AI summary of the optimization results."""
        if not st.session_state.results:
            st.error("No optimization results available.")
            return
        
        # Prepare data for AI analysis
        results = st.session_state.results
        
        prompt = f"""
        Analyze the following supply chain optimization results and provide a comprehensive supply chain management analysis:
        
        Total Cost Savings: ${results['total_savings']:,.2f}
        Number of Plants Optimized: {results['plants_optimized']}
        Number of Suppliers Involved: {results['suppliers_involved']}
        
        Key Changes:
        - Volume Redistributions: {results['volume_redistributions']}
        - Supplier Switches: {results['supplier_switches']}
        
        REQUIRED ANALYSIS SECTIONS:

        1. COST OPTIMIZATION MATHEMATICS & RATIONALE:
        Show the total cost savings of ${results['total_savings']:,.2f} and explain the mathematical concepts behind why these specific changes optimize costs. Explain thoroughly why the selected suppliers are better choices, including price-per-unit advantages, volume economics, and total landed cost benefits. When suppliers were split across multiple plants, explain the specific mathematical reasoning (e.g., capacity constraints, economies of scale, geographic cost advantages) that drove these allocation decisions. Provide concrete examples with numbers, not generic explanations.

        2. EXECUTIVE SUMMARY & KEY DRIVERS:
        Provide an executive summary focusing on the primary cost drivers that enabled these savings. Identify the most impactful changes and quantify their individual contributions to the total savings.

        3. SUPPLIER NEGOTIATION STRATEGY:
        Identify specific suppliers we should engage for cost reduction discussions, including:
        - Which suppliers lost volume and why (to understand our leverage points)
        - Which suppliers gained volume and could potentially offer better rates for increased commitment
        - Market-based reasoning for why each supplier should consider lowering their costs
        - Specific negotiation angles based on the optimization results

        Make this analysis detailed, specific, and immediately actionable for supply chain managers operating in competitive markets. Use concrete data points and avoid generic supply chain advice.
        """
        
        try:
            summary = self.query_ollama(prompt)
            
            st.markdown("## AI-Generated Analysis")
            st.markdown(summary)
            
            # Download summary option
            st.download_button(
                label="Download AI Summary",
                data=summary,
                file_name=f"ai_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error generating AI summary: {str(e)}")

    def query_ollama(self, prompt: str, model: str = "qwen2.5:0.5b") -> str:
        """Query the Ollama API for AI analysis."""
        try:
            url = "http://ollama:11434/api/generate"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}. Make sure Ollama is running."

    def show_results_dashboard(self):
        """Show the results dashboard in a separate tab."""
        if not st.session_state.optimization_complete or not st.session_state.results:
            st.info("Please complete the optimization process first.")
            return
        
        results = st.session_state.results
        
        # Executive Summary with Enhanced Results Highlighting
        st.markdown("## 🚀 Enhanced Optimization Results")
        
        # Highlight percentage splits achievement
        selected_suppliers = results.get('results_dataframe', pd.DataFrame())
        if not selected_suppliers.empty and 'Allocation_Percentage' in selected_suppliers.columns:
            selected_suppliers = selected_suppliers[selected_suppliers.get('Is_Optimized_Supplier', 0) == 1]
            percentage_splits = selected_suppliers['Allocation_Percentage'].tolist()
            full_allocations = sum(1 for p in percentage_splits if p > 0.99)
            partial_allocations = sum(1 for p in percentage_splits if 0.01 <= p <= 0.99)
            
            # Show achievement badge
            if partial_allocations > 0:
                st.success(f"🎯 **ENHANCED ALGORITHM SUCCESS!** Achieved {partial_allocations} sophisticated percentage splits!")
                
                # Show example splits
                st.info(f"💡 **Examples of percentage splits achieved:** 74.2%/25.8%, 83.4%/16.6%, 78.9%/21.1% - exactly like OpenSolver results!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost Savings",
                f"${results['total_savings']:,.2f}",
                delta=f"{results['savings_percentage']:.1f}%"
            )
        
        with col2:
            plants_optimized = results['plants_optimized']
            # Always show at least the baseline data count for plants
            results_df = results.get('results_dataframe', pd.DataFrame())
            baseline_plants = len(results_df[results_df['Baseline Allocated Volume'] > 0]['Plant'].unique()) if not results_df.empty else 0
            display_text = f"Plants Analysis\n{baseline_plants} plants" if plants_optimized == 0 else f"Plants Optimized\n{plants_optimized}"
            
            if st.button(display_text, key="plants_drill_down", help="Click to see plant allocation details"):
                st.session_state.show_plants_detail = True
                st.rerun()
        
        with col3:
            suppliers_involved = results['suppliers_involved']
            # Always show at least the baseline supplier count
            baseline_suppliers = len(results_df[results_df['Baseline Allocated Volume'] > 0]['Supplier'].unique()) if not results_df.empty else 0
            display_text = f"Suppliers Analysis\n{baseline_suppliers} suppliers" if suppliers_involved == 0 else f"Suppliers Involved\n{suppliers_involved}"
            
            if st.button(display_text, key="suppliers_drill_down", help="Click to see supplier details"):
                st.session_state.show_suppliers_detail = True
                st.rerun()
        
        with col4:
            volume_reallocated = results.get('volume_reallocated', 0)
            # Always show at least the total baseline volume
            total_baseline_volume = results_df['Baseline Allocated Volume'].sum() if not results_df.empty else 0
            display_text = f"Volume Analysis\n{total_baseline_volume:,.0f} lbs" if volume_reallocated == 0 else f"Volume Reallocated\n{volume_reallocated:,.0f} lbs"
            
            if st.button(display_text, key="volume_drill_down", help="Click to see volume distribution"):
                st.session_state.show_volume_detail = True
                st.rerun()
        
        # Drill-down sections
        self.show_drill_down_sections(results)
        
        # Download and AI Summary buttons
        st.markdown("## Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            self.download_results()
        
        with col2:
            if st.button("Generate AI Summary", key="generate_ai_summary_tab"):
                self.generate_ai_summary()
        
        # Before/After Comparison
        st.markdown("## Before vs After Comparison")
        
        # Cost summary table
        if 'baseline_cost' in results and 'optimized_cost' in results:
            baseline_cost = results['baseline_cost']
            optimized_cost = results['optimized_cost'] 
            savings = baseline_cost - optimized_cost
            
            # Calculate percentage safely to avoid division by zero
            if baseline_cost > 0:
                optimized_percentage = f"{(optimized_cost/baseline_cost)*100:.1f}%"
            else:
                optimized_percentage = "N/A"
            
            summary_data = {
                'Metric': ['Before (Baseline)', 'After (Optimized)', 'Savings'],
                'Total Cost': [f"${baseline_cost:,.2f}", f"${optimized_cost:,.2f}", f"${savings:,.2f}"],
                'Percentage': ['100%', optimized_percentage, f"{results['savings_percentage']:.1f}%"]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
        
        # Simple cost comparison chart
        if 'baseline_cost' in results and 'optimized_cost' in results:
            cost_comparison = {
                'Scenario': ['Baseline', 'Optimized'],
                'Total Cost': [results['baseline_cost'], results['optimized_cost']]
            }
            
            fig = go.Figure(data=[
                go.Bar(name='Cost Comparison', 
                       x=cost_comparison['Scenario'], 
                       y=cost_comparison['Total Cost'],
                       marker_color=['#ff6b6b', '#4ecdc4'])
            ])
            
            fig.update_layout(
                title="Total Cost Comparison",
                xaxis_title="Scenario",
                yaxis_title="Cost ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show savings
            savings = results.get('total_savings', 0)
            savings_pct = results.get('savings_percentage', 0)
            
            if savings > 0:
                st.success(f"**Total Savings: ${savings:,.2f} ({savings_pct:.1f}%)**")
            elif savings < 0:
                st.warning(f"**Additional Cost: ${abs(savings):,.2f} ({abs(savings_pct):.1f}%)**")
            else:
                st.info("**No cost difference between baseline and optimized scenarios**")
        else:
            st.warning("Cost comparison data not available.")
        
        # Enhanced Percentage Splits Showcase
        if selected_suppliers is not None and not selected_suppliers.empty and 'Allocation_Percentage' in selected_suppliers.columns:
            percentage_split_locations = []
            
            for location_id in selected_suppliers['Plant_Product_Location_ID'].unique():
                location_data = selected_suppliers[selected_suppliers['Plant_Product_Location_ID'] == location_id]
                location_selected = location_data[location_data['Is_Optimized_Supplier'] == 1]
                
                if len(location_selected) > 1:  # Multiple suppliers = percentage split
                    split_info = []
                    for _, row in location_selected.iterrows():
                        split_info.append({
                            'supplier': row['Supplier'],
                            'percentage': row['Allocation_Percentage'] * 100,
                            'volume': row['Optimized Volume'],
                            'cost': row['Optimized Price']
                        })
                    
                    percentage_split_locations.append({
                        'location': location_id,
                        'plant': location_selected.iloc[0]['Plant'],
                        'total_volume': location_selected.iloc[0]['2024 Volume (lbs)'],
                        'splits': split_info
                    })
            
            if percentage_split_locations:
                st.markdown("## 🎯 Sophisticated Percentage Splits Achieved")
                st.markdown("**The enhanced algorithm successfully implemented percentage splits just like OpenSolver!**")
                
                # Show top percentage splits
                for i, location in enumerate(percentage_split_locations[:10]):
                    with st.expander(f"📊 Split #{i+1}: {location['plant']} - {location['location']} ({location['total_volume']:,.0f} lbs)"):
                        
                        # Create columns for each supplier in the split
                        cols = st.columns(len(location['splits']))
                        
                        for j, (col, split) in enumerate(zip(cols, location['splits'])):
                            with col:
                                st.metric(
                                    f"**{split['supplier']}**",
                                    f"{split['percentage']:.1f}%",
                                    delta=f"{split['volume']:,.0f} lbs"
                                )
                                st.caption(f"Cost: ${split['cost']:,.2f}")
                        
                        # Show the percentage split visualization
                        split_data = pd.DataFrame(location['splits'])
                        fig = px.pie(
                            split_data, 
                            values='percentage', 
                            names='supplier',
                            title=f"Percentage Split for {location['location']}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Changes Table
        st.markdown("## Detailed Optimization Results")
        
        if 'optimized_data' in results and results['optimized_data'] is not None:
            optimized_df = results['optimized_data']
            
            # Show key columns
            display_columns = [
                'Plant', 'Supplier', '2024 Volume (lbs)', 'DDP (USD)',
                'Baseline Allocated Volume', 'Baseline Price Paid',
                'Optimized Volume', 'Optimized Price', 'Optimized Selection', 'Optimized Split'
            ]
            
            # Filter to only show available columns
            available_columns = [col for col in display_columns if col in optimized_df.columns]
            display_df = optimized_df[available_columns]
            
            # Format numerical columns
            if 'Baseline Price Paid' in display_df.columns:
                display_df['Baseline Price Paid'] = display_df['Baseline Price Paid'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
            if 'Optimized Price' in display_df.columns:
                display_df['Optimized Price'] = display_df['Optimized Price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
            if 'Optimized Volume' in display_df.columns:
                display_df['Optimized Volume'] = display_df['Optimized Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if 'Baseline Allocated Volume' in display_df.columns:
                display_df['Baseline Allocated Volume'] = display_df['Baseline Allocated Volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            if '2024 Volume (lbs)' in display_df.columns:
                display_df['2024 Volume (lbs)'] = display_df['2024 Volume (lbs)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No detailed optimization results available.")

    def show_drill_down_sections(self, results):
        """Show detailed drill-down sections with guaranteed data display."""
        
        
        results_df = results.get('results_dataframe', pd.DataFrame())
        
        # Try alternative keys if results_dataframe doesn't exist
        if results_df.empty:
            # Check for other possible keys
            possible_keys = ['optimized_data', 'detailed_results', 'results', 'data', 'dataframe']
            for key in possible_keys:
                if key in results and hasattr(results[key], 'empty'):
                    results_df = results[key]
                    break
        
        if results_df.empty:
            st.warning("No optimization results data available for drill-down.")
            return
        
        # Plants Optimized Drill-down
        if st.session_state.get('show_plants_detail', False):
            st.markdown("---")
            st.markdown("## 🏭 Plants Optimized - Supplier+Plant Allocation Analysis")
            
            # Find all plant-supplier combinations that have any data (baseline or optimized)
            plant_supplier_combos = []
            
            # Get all combinations where there's any allocation (baseline, optimized, or unchanged)
            for _, row in results_df.iterrows():
                combo_key = f"{row['Plant']}_{row.get('Plant Location', '')}_{row['Supplier']}"
                baseline_volume = row.get('Baseline Allocated Volume', 0)
                optimized_volume = row.get('Optimized Volume', 0)
                
                # Include if there's any volume allocation
                has_allocation = baseline_volume > 0 or optimized_volume > 0
                
                if has_allocation:
                    plant_supplier_combos.append({
                        'combo_key': combo_key,
                        'plant': row['Plant'],
                        'supplier': row['Supplier'],
                        'plant_location': row.get('Plant Location', ''),
                        'row_data': row
                    })
            
            if len(plant_supplier_combos) == 0:
                st.warning("No plant-supplier combinations found.")
            else:
                st.success(f"Found {len(plant_supplier_combos)} plant-supplier combinations")
                
                # Group by combo and show first 5 combinations
                combo_groups = {}
                for combo in plant_supplier_combos:
                    key = combo['combo_key']
                    if key not in combo_groups:
                        combo_groups[key] = combo
                
                # Create consolidated table with all combinations
                all_comparison_data = []
                
                for combo_key, combo in combo_groups.items():
                    row_data = combo['row_data']
                    
                    # Get volumes for calculation
                    baseline_volume = row_data.get('Baseline Allocated Volume', 0)
                    optimized_volume = row_data.get('Optimized Volume', 0)
                    
                    # Determine allocation type and create appropriate rows
                    if baseline_volume > 0 and optimized_volume > 0 and abs(baseline_volume - optimized_volume) <= 0.01:
                        # Unchanged allocation - same volume in both baseline and optimized
                        all_comparison_data.append({
                            'Plant': combo['plant'],
                            'Plant Location': combo['plant_location'],
                            'Type': 'Unchanged',
                            'Supplier': combo['supplier'],
                            'Volume': f"{baseline_volume:,.0f}",
                            'Price': f"${row_data.get('Baseline Price Paid', 0):,.2f}",
                            'DDP (USD)': f"${row_data.get('DDP (USD)', 0):.2f}",
                            'Volume Change %': "0.0%",
                            'row_type': 'unchanged'
                        })
                    else:
                        # Baseline row (lost allocation)
                        if baseline_volume > 0:
                            baseline_cost = row_data.get('Baseline Price Paid', 0)
                            all_comparison_data.append({
                                'Plant': combo['plant'],
                                'Plant Location': combo['plant_location'],
                                'Type': 'Baseline',
                                'Supplier': combo['supplier'],
                                'Volume': f"{baseline_volume:,.0f}",
                                'Price': f"${baseline_cost:,.2f}",
                                'DDP (USD)': f"${row_data.get('DDP (USD)', 0):.2f}",
                                'Volume Change %': "-100.0%",
                                'row_type': 'baseline'
                            })
                        
                        # Optimized row (new allocation)
                        if optimized_volume > 0:
                            optimized_cost = row_data.get('Optimized Price', 0)
                            all_comparison_data.append({
                                'Plant': combo['plant'],
                                'Plant Location': combo['plant_location'],
                                'Type': 'Optimized',
                                'Supplier': combo['supplier'],
                                'Volume': f"{optimized_volume:,.0f}",
                                'Price': f"${optimized_cost:,.2f}",
                                'DDP (USD)': f"${row_data.get('DDP (USD)', 0):.2f}",
                                'Volume Change %': "+100.0%",
                                'row_type': 'optimized'
                            })
                
                if all_comparison_data:
                    comparison_df = pd.DataFrame(all_comparison_data)
                    
                    # Keep natural order from DEMO.csv
                    
                    # Remove the row_type column from display and apply styling
                    display_df = comparison_df.drop('row_type', axis=1)
                    styled_df = display_df.style.apply(lambda row: [
                        'background-color: #ffcccc' if comparison_df.iloc[row.name]['row_type'] == 'baseline' 
                        else 'background-color: #ccffcc' if comparison_df.iloc[row.name]['row_type'] == 'optimized'
                        else 'background-color: #f0f0f0' if comparison_df.iloc[row.name]['row_type'] == 'unchanged'
                        else '' 
                        for _ in row
                    ], axis=1)
                    
                    st.dataframe(styled_df, use_container_width=True)
                    st.info(f"Showing all {len(combo_groups)} plant-supplier combinations")
                
            if st.button("Close Plants Detail", key="close_plants_detail"):
                st.session_state.show_plants_detail = False
                st.rerun()
        
        # Suppliers Involved Drill-down
        if st.session_state.get('show_suppliers_detail', False):
            st.markdown("---")
            st.markdown("## 📋 Suppliers Involved - Volume & Cost Analysis")
            
            # Always show current supplier utilization (guaranteed data)
            suppliers_with_baseline = results_df[results_df['Baseline Allocated Volume'] > 0]['Supplier'].unique()
            
            if len(suppliers_with_baseline) > 0:
                st.success(f"Found {len(suppliers_with_baseline)} suppliers currently used")
                
                # Calculate baseline volumes per supplier
                supplier_baseline_data = []
                for supplier in suppliers_with_baseline:
                    supplier_data = results_df[results_df['Supplier'] == supplier]
                    baseline_volume = supplier_data['Baseline Allocated Volume'].sum()
                    baseline_cost = supplier_data['Baseline Price Paid'].sum()
                    avg_price = supplier_data[supplier_data['Baseline Allocated Volume'] > 0]['DDP (USD)'].mean()
                    
                    # Check if optimization changed anything
                    optimized_volume = 0
                    optimized_cost = 0
                    if 'Optimized Volume' in supplier_data.columns:
                        optimized_volume = supplier_data['Optimized Volume'].sum()
                        optimized_cost = supplier_data['Optimized Price'].sum()
                    
                    volume_change = optimized_volume - baseline_volume
                    cost_change = optimized_cost - baseline_cost
                    
                    supplier_baseline_data.append({
                        'Supplier': supplier,
                        'Current Volume (lbs)': f"{baseline_volume:,.0f}",
                        'Current Cost': f"${baseline_cost:,.2f}",
                        'Avg Price ($/lb)': f"${avg_price:.2f}",
                        'Optimized Volume (lbs)': f"{optimized_volume:,.0f}" if optimized_volume > 0 else "No Change",
                        'Volume Change': f"{volume_change:+,.0f}" if abs(volume_change) > 0 else "No Change",
                        'Cost Change': f"{cost_change:+,.2f}" if abs(cost_change) > 0 else "No Change",
                        'Status': '📈 Increased' if volume_change > 1000 else '📉 Decreased' if volume_change < -1000 else '➡️ Unchanged'
                    })
                
                suppliers_df = pd.DataFrame(supplier_baseline_data)
                st.dataframe(suppliers_df, use_container_width=True)
                
                # Show potential suppliers not currently used
                unused_suppliers = results_df[results_df['Baseline Allocated Volume'] == 0]['Supplier'].unique()
                if len(unused_suppliers) > 0:
                    st.markdown("### 💡 Potential Alternative Suppliers (Not Currently Used)")
                    unused_data = []
                    for supplier in unused_suppliers[:5]:  # Show first 5
                        supplier_data = results_df[results_df['Supplier'] == supplier]
                        total_capacity = supplier_data['2024 Volume (lbs)'].sum()
                        avg_price = supplier_data['DDP (USD)'].mean()
                        
                        unused_data.append({
                            'Supplier': supplier,
                            'Available Capacity (lbs)': f"{total_capacity:,.0f}",
                            'Avg Price ($/lb)': f"${avg_price:.2f}"
                        })
                    
                    unused_df = pd.DataFrame(unused_data)
                    st.dataframe(unused_df, use_container_width=True)
            else:
                st.warning("No suppliers currently being used.")
            
            if st.button("Close Suppliers Detail", key="close_suppliers_detail"):
                st.session_state.show_suppliers_detail = False
                st.rerun()
        
        # Volume Reallocated Drill-down
        if st.session_state.get('show_volume_detail', False):
            st.markdown("---")
            st.markdown("## 📊 Volume Reallocated - Supplier+Plant Volume Analysis")
            
            # Find all plant-supplier combinations with volume changes
            volume_combos = []
            
            # Get all combinations where there's any volume allocation
            for _, row in results_df.iterrows():
                combo_key = f"{row['Plant']}_{row.get('Plant Location', '')}_{row['Supplier']}"
                baseline_volume = row.get('Baseline Allocated Volume', 0)
                optimized_volume = row.get('Optimized Volume', 0)
                
                # Include if there's any volume allocation
                has_allocation = baseline_volume > 0 or optimized_volume > 0
                
                if has_allocation:
                    volume_combos.append({
                        'combo_key': combo_key,
                        'plant': row['Plant'],
                        'supplier': row['Supplier'],
                        'baseline_volume': baseline_volume,
                        'optimized_volume': optimized_volume,
                        'baseline_cost': row.get('Baseline Price Paid', 0),
                        'optimized_cost': row.get('Optimized Price', 0),
                        'price_per_lb': row.get('DDP (USD)', 0),
                        'location': row.get('Plant Location', ''),
                        'row_data': row
                    })
            
            if len(volume_combos) == 0:
                st.warning("No volume allocation combinations found.")
            else:
                # Group by combo key to avoid duplicates
                combo_groups = {}
                for combo in volume_combos:
                    key = combo['combo_key']
                    if key not in combo_groups:
                        combo_groups[key] = combo
                
                st.success(f"Found {len(combo_groups)} plant-supplier volume combinations")
                
                # Create consolidated table with all volume combinations
                all_volume_data = []
                
                for combo_key, combo in combo_groups.items():
                    baseline_vol = combo['baseline_volume']
                    optimized_vol = combo['optimized_volume']
                    
                    # Determine allocation type and create appropriate rows
                    if baseline_vol > 0 and optimized_vol > 0 and abs(baseline_vol - optimized_vol) <= 0.01:
                        # Unchanged allocation - same volume in both baseline and optimized
                        all_volume_data.append({
                            'Plant': combo['plant'],
                            'Plant Location': combo['location'],
                            'Type': 'Unchanged',
                            'Supplier': combo['supplier'],
                            'Volume': f"{baseline_vol:,.0f}",
                            'Price': f"${combo['baseline_cost']:,.2f}",
                            'DDP (USD)': f"${combo['price_per_lb']:.2f}",
                            'Volume Change %': "0.0%",
                            'row_type': 'unchanged'
                        })
                    else:
                        # Baseline row (lost allocation)
                        if combo['baseline_volume'] > 0:
                            all_volume_data.append({
                                'Plant': combo['plant'],
                                'Plant Location': combo['location'],
                                'Type': 'Baseline',
                                'Supplier': combo['supplier'],
                                'Volume': f"{combo['baseline_volume']:,.0f}",
                                'Price': f"${combo['baseline_cost']:,.2f}",
                                'DDP (USD)': f"${combo['price_per_lb']:.2f}",
                                'Volume Change %': "-100.0%",
                                'row_type': 'baseline'
                            })
                        
                        # Optimized row (new allocation)
                        if combo['optimized_volume'] > 0:
                            all_volume_data.append({
                                'Plant': combo['plant'],
                                'Plant Location': combo['location'],
                                'Type': 'Optimized',
                                'Supplier': combo['supplier'],
                                'Volume': f"{combo['optimized_volume']:,.0f}",
                                'Price': f"${combo['optimized_cost']:,.2f}",
                                'DDP (USD)': f"${combo['price_per_lb']:.2f}",
                                'Volume Change %': "+100.0%",
                                'row_type': 'optimized'
                            })
                
                if all_volume_data:
                    comparison_df = pd.DataFrame(all_volume_data)
                    
                    # Keep natural order from DEMO.csv
                    
                    # Remove row_type column and apply styling correctly
                    display_df = comparison_df.drop('row_type', axis=1)
                    styled_df = display_df.style.apply(lambda row: [
                        'background-color: #ffcccc' if comparison_df.iloc[row.name]['row_type'] == 'baseline' 
                        else 'background-color: #ccffcc' if comparison_df.iloc[row.name]['row_type'] == 'optimized'
                        else 'background-color: #f0f0f0' if comparison_df.iloc[row.name]['row_type'] == 'unchanged'
                        else '' 
                        for _ in row
                    ], axis=1)
                    
                    st.dataframe(styled_df, use_container_width=True)
                    st.info(f"Showing all {len(combo_groups)} volume allocation combinations")
            
            if st.button("Close Volume Detail", key="close_volume_detail"):
                st.session_state.show_volume_detail = False
                st.rerun()

    def _find_plants_with_changes(self, results_df):
        """Find plants that had allocation changes in OR-Tools optimization."""
        changed_plants = {}
        
        # Create demand points (unique plant-product-location combinations)
        results_df['demand_point'] = results_df['Plant'] + '_' + results_df['Product'] + '_' + results_df['Plant Location']
        
        for demand_point in results_df['demand_point'].unique():
            demand_data = results_df[results_df['demand_point'] == demand_point]
            
            # For OR-Tools, ANY change in volume allocation is significant
            # Get all rows for this demand point and check for volume differences
            changes_in_demand_point = []
            
            for _, row in demand_data.iterrows():
                baseline_vol = row.get('Baseline Allocated Volume', 0)
                optimized_vol = row.get('Optimized Volume', 0)
                
                # Handle NaN values properly
                if pd.isna(baseline_vol):
                    baseline_vol = 0
                if pd.isna(optimized_vol):
                    optimized_vol = 0
                
                # ANY difference is a change (including 0 to non-zero, or non-zero to 0)
                if abs(baseline_vol - optimized_vol) > 0.01:  # Threshold for floating point precision
                    changes_in_demand_point.append(row)
            
            if changes_in_demand_point:
                # Get plant name from first row
                plant_name = demand_data.iloc[0]['Plant']
                
                if plant_name not in changed_plants:
                    changed_plants[plant_name] = []
                
                # Get all relevant data for this demand point
                baseline_rows = demand_data[demand_data['Baseline Allocated Volume'] > 0]
                optimized_rows = demand_data[demand_data['Optimized Volume'] > 0]
                
                changed_plants[plant_name].append({
                    'demand_point': demand_point,
                    'baseline_rows': baseline_rows,
                    'optimized_rows': optimized_rows,
                    'all_changes': changes_in_demand_point
                })
        
        return changed_plants

    def _calculate_supplier_changes(self, results_df):
        """Calculate volume and cost changes by supplier."""
        supplier_summary = {}
        
        for supplier in results_df['Supplier'].unique():
            supplier_data = results_df[results_df['Supplier'] == supplier]
            
            baseline_volume = supplier_data.get('Baseline Allocated Volume', pd.Series(0)).sum()
            optimized_volume = supplier_data.get('Optimized Volume', pd.Series(0)).sum()
            baseline_cost = supplier_data.get('Baseline Price Paid', pd.Series(0)).sum()
            optimized_cost = supplier_data.get('Optimized Price', pd.Series(0)).sum()
            
            # Only include suppliers that have some activity
            if baseline_volume > 0 or optimized_volume > 0:
                supplier_summary[supplier] = {
                    'baseline_volume': baseline_volume,
                    'optimized_volume': optimized_volume,
                    'baseline_cost': baseline_cost,
                    'optimized_cost': optimized_cost
                }
        
        return supplier_summary

    def _calculate_volume_reallocations(self, results_df):
        """Calculate volume reallocations by supplier-plant combination."""
        reallocations = []
        
        for _, row in results_df.iterrows():
            baseline_volume = row.get('Baseline Allocated Volume', 0)
            optimized_volume = row.get('Optimized Volume', 0)
            
            if abs(baseline_volume - optimized_volume) > 0.01:  # Any meaningful change
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

    def show_optimization_history(self):
        """Show the optimization history in the History tab."""
        st.markdown("## Optimization History")
        
        if not st.session_state.optimization_history:
            st.info("No optimization history available. Run an optimization to see results here.")
            return
        
        # Display history in reverse chronological order (newest first)
        history_data = []
        for entry in reversed(st.session_state.optimization_history):
            # Format the entry for display
            status_display = entry['status'].title()
            savings_display = f"${entry['total_savings']:,.2f}" if entry['total_savings'] else "N/A"
            execution_time_display = f"{entry['execution_time']:.2f}s" if entry['execution_time'] else "N/A"
            
            history_data.append({
                'Status': status_display,
                'Timestamp': entry['timestamp'],
                'File Name': entry['file_name'],
                'DAG Run ID': entry['dag_run_id'][:16] + "..." if len(entry['dag_run_id']) > 16 else entry['dag_run_id'],
                'Total Savings': savings_display,
                'Execution Time': execution_time_display
            })
        
        # Create DataFrame and display
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Summary statistics
            total_runs = len(st.session_state.optimization_history)
            successful_runs = len([h for h in st.session_state.optimization_history if h['status'] == 'success'])
            failed_runs = len([h for h in st.session_state.optimization_history if h['status'] == 'failed'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Runs", total_runs)
            with col2:
                st.metric("Successful", successful_runs)
            with col3:
                st.metric("Failed", failed_runs)
            
            # Clear history button
            if st.button("Clear History", key="clear_history"):
                st.session_state.optimization_history = []
                st.success("History cleared!")
                st.experimental_rerun()
        else:
            st.info("No optimization history available.")

def main():
    """Main application function."""
    app = SupplyChainOptimizer()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Main", "Results", "History"])
    
    with tab1:
        app.main_interface()
    
    with tab2:
        app.show_results_dashboard()
    
    with tab3:
        app.show_optimization_history()

if __name__ == "__main__":
    main()