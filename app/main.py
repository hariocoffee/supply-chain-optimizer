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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from database import DatabaseManager
from cache_manager import CacheManager
from optimizer import PyomoOptimizer
from airflow_client import AirflowClient

# Configure Streamlit page
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Apple-like minimalistic design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .action-button {
        background: #007AFF;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        margin: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .action-button:hover {
        background: #0056CC;
        transform: translateY(-2px);
    }
    
    .constraint-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .savings-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .savings-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #007AFF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class SupplyChainOptimizer:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.optimizer = PyomoOptimizer()
        self.airflow_client = AirflowClient()
        
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
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()

    def create_template_file(self) -> bytes:
        """Create a template file based on the expected structure for supplier selection optimization."""
        template_data = {
            'Plant': ['Alpha', 'Beta', 'Charlie', 'Charlie', 'Delta', 'Delta'],
            'Product': ['Brownies', 'Brownies', 'Brownies', 'Brownies', 'Brownies', 'Brownies'],
            '2024 Volume (lbs)': [5900000, 15900000, 56800000, 56800000, 75500000, 75500000],
            'Supplier': ['Aunt Smith', 'Aunt Smith', 'Aunt Bethany', 'Aunt Smith', 'Aunt Bethany', 'Aunt Baker'],
            'Plant Location': ['Location1', 'Location2', 'Location3', 'Location3', 'Location4', 'Location4'],
            'DDP (USD)': [8.05, 6.91, 9.96, 9.28, 5.47, 7.29],
            'Baseline Allocated Volume': [5900000, 15900000, 56800000, 0, 75500000, 0],
            'Baseline Price Paid': [47495000, 109869000, 565728000, 0, 412985000, 0],
            'Selection': ['X', 'X', 'X', '', 'X', ''],
            'Split': ['100%', '100%', '100%', '0%', '100%', '0%'],
            'Optimized Volume': ['', '', '', '', '', ''],
            'Optimized Price': ['', '', '', '', '', ''],
            'Optimized Selection': ['', '', '', '', '', ''],
            'Optimized Split': ['', '', '', '', '', ''],
            'Cost Savings': ['', '', '', '', '', ''],
            'Plant_Product_Location_ID': ['Alpha_Location1_Brownies', 'Beta_Location2_Brownies', 'Charlie_Location3_Brownies', 'Charlie_Location3_Brownies', 'Delta_Location4_Brownies', 'Delta_Location4_Brownies'],
            'Supplier_Plant_Product_Location_ID': ['Alpha_Location1_Brownies_Aunt Smith', 'Beta_Location2_Brownies_Aunt Smith', 'Charlie_Location3_Brownies_Aunt Bethany', 'Charlie_Location3_Brownies_Aunt Smith', 'Delta_Location4_Brownies_Aunt Bethany', 'Delta_Location4_Brownies_Aunt Baker'],
            'Plant_Product_ID': ['Alpha_Brownies', 'Beta_Brownies', 'Charlie_Brownies', 'Charlie_Brownies', 'Delta_Brownies', 'Delta_Brownies'],
            'Supplier_Plant_Product_ID': ['Alpha_Brownies_Aunt Smith', 'Beta_Brownies_Aunt Smith', 'Charlie_Brownies_Aunt Bethany', 'Charlie_Brownies_Aunt Smith', 'Delta_Brownies_Aunt Bethany', 'Delta_Brownies_Aunt Baker'],
            'Plant_Product_Baseline_Cost': [47495000, 109869000, 565728000, 565728000, 412985000, 412985000],
            'Plant_Product_Optimized_Cost': ['', '', '', '', '', ''],
            'Plant_Product_Total_Savings': ['', '', '', '', '', ''],
            'Allocated_Cost_Savings': ['', '', '', '', '', ''],
            'Is_Baseline_Supplier': [1, 1, 1, 0, 1, 0],
            'Is_Optimized_Supplier': ['', '', '', '', '', ''],
            'Volume_Fraction': ['1.000', '1.000', '1.000', '0.000', '1.000', '0.000']
        }
        
        template_df = pd.DataFrame(template_data)
        return template_df.to_csv(index=False).encode()

    def main_interface(self):
        """Main interface for the application."""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⚡ Supply Chain Optimizer</h1>
            <p>Optimize your supply chain with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📥 Download Template", key="download_template"):
                template_file = self.create_template_file()
                st.download_button(
                    label="Download CSV Template",
                    data=template_file,
                    file_name="supply_chain_template.csv",
                    mime="text/csv"
                )
        
        with col2:
            uploaded_file = st.file_uploader(
                "📤 Upload File",
                type=['csv', 'xlsx', 'xls'],
                key="file_uploader"
            )
        
        with col3:
            if uploaded_file and st.button("➡️ Next", key="next_button"):
                self.process_uploaded_file(uploaded_file)
        
        # Process file and show constraints interface
        if st.session_state.data is not None:
            self.show_constraints_interface()
        
        # Show optimization button if constraints are set
        if st.session_state.constraints_set:
            self.show_optimization_interface()

    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded file and prepare for optimization."""
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Generate file hash for caching
            file_hash = self.generate_file_hash(df)
            
            # Check if this file has been processed before
            cached_result = self.cache_manager.get_cached_result(file_hash)
            if cached_result:
                st.info("💾 This file has been optimized before. Previous results are available in the History tab.")
            
            # Always clear current session results when uploading new file
            st.session_state.results = None
            st.session_state.optimization_complete = False
            
            # Create Plant_Product_ID if it doesn't exist
            if 'Plant_Product_ID' not in df.columns:
                df['Plant_Product_ID'] = df['Plant'] + '_' + df['Product']
            
            # Store in session state
            st.session_state.data = df
            st.session_state.file_hash = file_hash
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Store in database
            self.db_manager.store_baseline_data(df, file_hash, uploaded_file.name)
            
            st.success(f"✅ File '{uploaded_file.name}' processed successfully!")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Unique Plant-Product-Locations", df['Plant_Product_Location_ID'].nunique())
                with col4:
                    st.metric("Unique Suppliers", df['Supplier'].nunique())
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    def show_constraints_interface(self):
        """Show the constraints setting interface."""
        st.markdown("## 🎯 Automatically Detected Constraints")
        
        df = st.session_state.data
        
        # Auto-detect supplier constraints from baseline data
        supplier_constraints = self.calculate_supplier_constraints(df)
        
        # Display detected constraints
        st.markdown("### 📊 Supplier Constraints")
        
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
        st.markdown("**📝 Edit Supplier Constraints:**")
        st.info("💡 Modify the minimum and maximum volume constraints for each supplier. Leave unchanged to use auto-detected values.")
        
        # Create editable constraints interface
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
        st.markdown("### 🏭 Plant Requirements")
        st.info("**Each plant-product-location combination MUST receive its full allocated volume as specified in Column C (2024 Volume).**")
        
        plant_requirements = df.groupby(['Plant', 'Product', 'Plant Location'])['2024 Volume (lbs)'].first().reset_index()
        plant_requirements['Required Volume (lbs)'] = plant_requirements['2024 Volume (lbs)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(plant_requirements[['Plant', 'Product', 'Plant Location', 'Required Volume (lbs)']], use_container_width=True)
        
        if st.button("✅ Apply Constraints", key="set_constraints"):
            # Store edited constraints in session state
            st.session_state.supplier_constraints = edited_constraints
            st.session_state.plant_constraints = {}  # No plant constraints needed
            st.session_state.constraints_set = True
            
            # Store in database
            self.db_manager.store_constraints(
                st.session_state.file_hash,
                edited_constraints,
                {}  # No plant constraints
            )
            
            st.success("✅ Supplier constraints applied successfully!")
    
    def calculate_supplier_constraints(self, df):
        """Calculate supplier constraints from baseline data for supplier selection optimization."""
        try:
            supplier_constraints = {}
            
            # Validate required columns exist
            required_columns = ['Supplier', 'Baseline Allocated Volume', 'Baseline Price Paid', '2024 Volume (lbs)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns for constraint calculation: {missing_columns}")
                return {}
            
            # Group by supplier and sum baseline allocated volume and price
            supplier_data = df.groupby('Supplier').agg({
                'Baseline Allocated Volume': 'sum',
                'Baseline Price Paid': 'sum'
            }).reset_index()
            
            # For supplier selection optimization, the max should be based on realistic supplier capacity
            # Use baseline allocation as indicator of actual supplier capacity
            for _, row in supplier_data.iterrows():
                supplier = row['Supplier']
                baseline_volume = row['Baseline Allocated Volume']
                baseline_cost = row['Baseline Price Paid']
                
                # Set max capacity to allow optimization flexibility without being unrealistic
                total_volume = df['2024 Volume (lbs)'].sum()
                
                if baseline_volume > 0:
                    # For existing suppliers, allow them to take the full total volume if they're most cost-effective
                    # This is the maximum logical constraint - one supplier can't handle more than 100% of demand
                    max_capacity = total_volume
                else:
                    # For new suppliers (no baseline), allow them to compete for full volume
                    max_capacity = total_volume
                
                supplier_constraints[supplier] = {
                    'min': 0,  # Minimum can be 0 (supplier might not be used)
                    'max': max_capacity,  # Maximum for optimization (flexible)
                    'display_max': baseline_volume if baseline_volume > 0 else total_volume,  # Display value
                    'baseline_volume': baseline_volume,
                    'baseline_cost': baseline_cost
                }
            
            return supplier_constraints
            
        except Exception as e:
            st.error(f"Error calculating supplier constraints: {str(e)}")
            return {}

    def show_optimization_interface(self):
        """Show the optimization interface."""
        st.markdown("## 🚀 Run Optimization")
        
        if st.button("🔄 Run Optimization", key="run_optimization"):
            with st.spinner("Running optimization... This may take a few minutes."):
                # Trigger Airflow DAG
                dag_run_id = self.airflow_client.trigger_optimization_dag(
                    st.session_state.file_hash
                )
                
                if dag_run_id:
                    st.success(f"✅ Optimization started! DAG Run ID: {dag_run_id}")
                    
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
                    
                    # Run the actual optimization
                    results = self.optimizer.optimize(
                        st.session_state.data,
                        st.session_state.supplier_constraints,
                        st.session_state.plant_constraints
                    )
                    
                    if results:
                        # Calculate actual optimization changes from the results dataframe
                        results_df = results.get('results_dataframe', pd.DataFrame())
                        
                        if not results_df.empty:
                            # Find locations where baseline supplier changed
                            # Group by location and check if any baseline supplier lost allocation
                            location_switches = []
                            
                            for location_id in results_df['Plant_Product_Location_ID'].unique():
                                location_data = results_df[results_df['Plant_Product_Location_ID'] == location_id]
                                
                                # Find baseline supplier for this location
                                baseline_supplier = location_data[location_data['Is_Baseline_Supplier'] == 1]
                                
                                if not baseline_supplier.empty:
                                    baseline_row = baseline_supplier.iloc[0]
                                    # Check if baseline supplier got 0 optimized volume
                                    if baseline_row['Is_Optimized_Supplier'] == 0:
                                        location_switches.append({
                                            'location_id': location_id,
                                            'plant': baseline_row['Plant'],
                                            'volume': baseline_row['2024 Volume (lbs)'],
                                            'from_supplier': baseline_row['Supplier'],
                                            'to_supplier': location_data[location_data['Is_Optimized_Supplier'] == 1]['Supplier'].iloc[0] if any(location_data['Is_Optimized_Supplier'] == 1) else 'Unknown'
                                        })
                            
                            # Calculate metrics
                            volume_reallocated = sum([switch['volume'] for switch in location_switches])
                            plants_optimized = len(set([switch['plant'] for switch in location_switches]))
                            supplier_switches = len(location_switches)
                        else:
                            volume_reallocated = 0
                            plants_optimized = 0
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
                            'suppliers_involved': results.get('optimization_summary', {}).get('total_suppliers', 0),
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
                        
                        st.success("✅ Optimization completed successfully!")
                        
                        # Show success message and redirect to results tab
                        st.info("🎉 Optimization complete! Check the **Results** tab to view detailed results, download data, and generate AI summaries.")
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
                        st.error("❌ Optimization failed. Please check your constraints and try again.")
                    
                    # Add to history regardless of success/failure
                    st.session_state.optimization_history.append(history_entry)
                else:
                    st.error("❌ Failed to start optimization process.")

    def download_results(self):
        """Generate and download the results file."""
        if st.session_state.results and 'optimized_data' in st.session_state.results:
            results_df = st.session_state.results['optimized_data']
            if results_df is not None and not results_df.empty:
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Optimized Results CSV",
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
        Analyze the following supply chain optimization results and provide insights:
        
        Total Cost Savings: ${results['total_savings']:,.2f}
        Number of Plants Optimized: {results['plants_optimized']}
        Number of Suppliers Involved: {results['suppliers_involved']}
        
        Key Changes:
        - Volume Redistributions: {results['volume_redistributions']}
        - Supplier Switches: {results['supplier_switches']}
        
        Please provide:
        1. Executive Summary of the optimization
        2. Key drivers of cost savings
        3. Risk assessment of the recommended changes
        4. Implementation recommendations
        
        Make this analysis professional and actionable for supply chain managers.
        """
        
        try:
            summary = self.query_ollama(prompt)
            
            st.markdown("## 🤖 AI-Generated Analysis")
            st.markdown(summary)
            
            # Download summary option
            st.download_button(
                label="📥 Download AI Summary",
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
            st.info("🔄 Please complete the optimization process first.")
            return
        
        results = st.session_state.results
        
        # Executive Summary
        st.markdown("## 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cost Savings",
                f"${results['total_savings']:,.2f}",
                delta=f"{results['savings_percentage']:.1f}%"
            )
        
        with col2:
            plants_optimized = results['plants_optimized']
            if st.button(f"🏭 Plants Optimized\n{plants_optimized}", key="plants_drill_down", help="Click to see detailed changes"):
                st.session_state.show_plants_detail = True
        
        with col3:
            suppliers_involved = results['suppliers_involved']
            if st.button(f"🚚 Suppliers Involved\n{suppliers_involved}", key="suppliers_drill_down", help="Click to see supplier list"):
                st.session_state.show_suppliers_detail = True
        
        with col4:
            volume_reallocated = results.get('volume_reallocated', 0)
            if st.button(f"📦 Volume Reallocated\n{volume_reallocated:,.0f} lbs", key="volume_drill_down", help="Click to see volume changes"):
                st.session_state.show_volume_detail = True
        
        # Drill-down sections
        self.show_drill_down_sections(results)
        
        # Download and AI Summary buttons
        st.markdown("## 📊 Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            self.download_results()
        
        with col2:
            if st.button("🤖 Generate AI Summary", key="generate_ai_summary_tab"):
                self.generate_ai_summary()
        
        # Before/After Comparison
        st.markdown("## 📈 Before vs After Comparison")
        
        # Cost summary table
        if 'baseline_cost' in results and 'optimized_cost' in results:
            baseline_cost = results['baseline_cost']
            optimized_cost = results['optimized_cost'] 
            savings = baseline_cost - optimized_cost
            
            summary_data = {
                'Metric': ['Before (Baseline)', 'After (Optimized)', 'Savings'],
                'Total Cost': [f"${baseline_cost:,.2f}", f"${optimized_cost:,.2f}", f"${savings:,.2f}"],
                'Percentage': ['100%', f"{(optimized_cost/baseline_cost)*100:.1f}%", f"{results['savings_percentage']:.1f}%"]
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
                st.success(f"💰 **Total Savings: ${savings:,.2f} ({savings_pct:.1f}%)**")
            elif savings < 0:
                st.warning(f"⚠️ **Additional Cost: ${abs(savings):,.2f} ({abs(savings_pct):.1f}%)**")
            else:
                st.info("ℹ️ **No cost difference between baseline and optimized scenarios**")
        else:
            st.warning("Cost comparison data not available.")
        
        # Detailed Changes Table
        st.markdown("## 📋 Detailed Optimization Results")
        
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
        """Show detailed drill-down sections for interactive metrics."""
        results_df = results.get('optimized_data', pd.DataFrame())
        
        if results_df.empty:
            return
        
        # Plants Optimized Drill-down
        if st.session_state.get('show_plants_detail', False):
            st.markdown("---")
            st.markdown("## 🏭 Plants Optimized - Detailed Changes")
            
            # Find all location switches for detailed display
            location_switches = []
            for location_id in results_df['Plant_Product_Location_ID'].unique():
                location_data = results_df[results_df['Plant_Product_Location_ID'] == location_id]
                baseline_supplier = location_data[location_data['Is_Baseline_Supplier'] == 1]
                
                if not baseline_supplier.empty:
                    baseline_row = baseline_supplier.iloc[0]
                    if baseline_row['Is_Optimized_Supplier'] == 0:
                        optimized_supplier = location_data[location_data['Is_Optimized_Supplier'] == 1]
                        if not optimized_supplier.empty:
                            optimized_row = optimized_supplier.iloc[0]
                            location_switches.append((baseline_row, optimized_row))
            
            if location_switches:
                for i, (baseline_row, optimized_row) in enumerate(location_switches):
                    st.markdown(f"### Change #{i+1}: {baseline_row['Plant']} - {baseline_row['Plant Location']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔴 BEFORE (Baseline)**")
                        before_data = {
                            'Plant': baseline_row['Plant'],
                            'Location': baseline_row['Plant Location'],
                            'Supplier': baseline_row['Supplier'],
                            'Volume (lbs)': f"{baseline_row['2024 Volume (lbs)']:,.0f}",
                            'DDP Price': f"${baseline_row['DDP (USD)']:.2f}",
                            'Total Cost': f"${baseline_row['Baseline Price Paid']:,.2f}",
                            'Allocation': baseline_row['Split']
                        }
                        before_df = pd.DataFrame([before_data])
                        st.dataframe(before_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("**🟢 AFTER (Optimized)**")
                        after_data = {
                            'Plant': optimized_row['Plant'],
                            'Location': optimized_row['Plant Location'],
                            'Supplier': optimized_row['Supplier'],
                            'Volume (lbs)': f"{optimized_row['Optimized Volume']:,.0f}",
                            'DDP Price': f"${optimized_row['DDP (USD)']:.2f}",
                            'Total Cost': f"${optimized_row['Optimized Price']:,.2f}",
                            'Allocation': optimized_row['Optimized Split']
                        }
                        after_df = pd.DataFrame([after_data])
                        st.dataframe(after_df, use_container_width=True)
                    
                    # Show savings for this change
                    savings = baseline_row['Baseline Price Paid'] - optimized_row['Optimized Price']
                    savings_pct = (savings / baseline_row['Baseline Price Paid']) * 100 if baseline_row['Baseline Price Paid'] > 0 else 0
                    st.success(f"💰 **Savings for this change: ${savings:,.2f} ({savings_pct:.1f}%)**")
                    st.markdown("---")
                
                if st.button("❌ Close Plants Detail", key="close_plants_detail"):
                    st.session_state.show_plants_detail = False
                    st.rerun()
        
        # Suppliers Involved Drill-down
        if st.session_state.get('show_suppliers_detail', False):
            st.markdown("---")
            st.markdown("## 🚚 Suppliers Involved - Complete List")
            
            # Get all unique suppliers and standardize names
            all_suppliers = results_df['Supplier'].unique()
            supplier_data = []
            
            for supplier in all_suppliers:
                # Standardize by stripping spaces
                standardized_name = supplier.strip()
                supplier_info = results_df[results_df['Supplier'] == supplier]
                
                # Check if supplier has any optimized volume
                total_optimized_volume = supplier_info['Optimized Volume'].sum()
                total_baseline_volume = supplier_info['Baseline Allocated Volume'].sum()
                
                supplier_data.append({
                    'Original Name': f"'{supplier}'",
                    'Standardized Name': f"'{standardized_name}'",
                    'Baseline Volume (lbs)': f"{total_baseline_volume:,.0f}",
                    'Optimized Volume (lbs)': f"{total_optimized_volume:,.0f}",
                    'Volume Change': f"{total_optimized_volume - total_baseline_volume:+,.0f}",
                    'Status': 'Gained Volume' if total_optimized_volume > total_baseline_volume else 'Lost Volume' if total_optimized_volume < total_baseline_volume else 'No Change'
                })
            
            suppliers_df = pd.DataFrame(supplier_data)
            st.dataframe(suppliers_df, use_container_width=True)
            
            if st.button("❌ Close Suppliers Detail", key="close_suppliers_detail"):
                st.session_state.show_suppliers_detail = False
                st.rerun()
        
        # Volume Reallocated Drill-down
        if st.session_state.get('show_volume_detail', False):
            st.markdown("---")
            st.markdown("## 📦 Volume Reallocated - Detailed Changes")
            
            # Same logic as Plants Optimized but focused on volume
            location_switches = []
            for location_id in results_df['Plant_Product_Location_ID'].unique():
                location_data = results_df[results_df['Plant_Product_Location_ID'] == location_id]
                baseline_supplier = location_data[location_data['Is_Baseline_Supplier'] == 1]
                
                if not baseline_supplier.empty:
                    baseline_row = baseline_supplier.iloc[0]
                    if baseline_row['Is_Optimized_Supplier'] == 0:
                        optimized_supplier = location_data[location_data['Is_Optimized_Supplier'] == 1]
                        if not optimized_supplier.empty:
                            optimized_row = optimized_supplier.iloc[0]
                            location_switches.append((baseline_row, optimized_row))
            
            if location_switches:
                # Summary table first
                st.markdown("### 📊 Volume Reallocation Summary")
                summary_data = []
                total_volume = 0
                
                for baseline_row, optimized_row in location_switches:
                    volume = baseline_row['2024 Volume (lbs)']
                    total_volume += volume
                    summary_data.append({
                        'Plant': baseline_row['Plant'],
                        'Location': baseline_row['Plant Location'],
                        'Volume Reallocated (lbs)': f"{volume:,.0f}",
                        'From Supplier': baseline_row['Supplier'],
                        'To Supplier': optimized_row['Supplier'],
                        'Cost Reduction': f"${baseline_row['Baseline Price Paid'] - optimized_row['Optimized Price']:,.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                st.info(f"**Total Volume Reallocated: {total_volume:,.0f} lbs across {len(location_switches)} locations**")
                
                # Detailed before/after (same as Plants Optimized)
                st.markdown("### 🔍 Detailed Before/After Comparison")
                
                for i, (baseline_row, optimized_row) in enumerate(location_switches):
                    with st.expander(f"📦 Volume Change #{i+1}: {baseline_row['Plant']} - {baseline_row['2024 Volume (lbs)']:,.0f} lbs"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🔴 BEFORE (Baseline)**")
                            before_data = {
                                'Supplier': baseline_row['Supplier'],
                                'Volume (lbs)': f"{baseline_row['2024 Volume (lbs)']:,.0f}",
                                'DDP Price': f"${baseline_row['DDP (USD)']:.2f}",
                                'Total Cost': f"${baseline_row['Baseline Price Paid']:,.2f}"
                            }
                            before_df = pd.DataFrame([before_data])
                            st.dataframe(before_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("**🟢 AFTER (Optimized)**")
                            after_data = {
                                'Supplier': optimized_row['Supplier'],
                                'Volume (lbs)': f"{optimized_row['Optimized Volume']:,.0f}",
                                'DDP Price': f"${optimized_row['DDP (USD)']:.2f}",
                                'Total Cost': f"${optimized_row['Optimized Price']:,.2f}"
                            }
                            after_df = pd.DataFrame([after_data])
                            st.dataframe(after_df, use_container_width=True)
                
                if st.button("❌ Close Volume Detail", key="close_volume_detail"):
                    st.session_state.show_volume_detail = False
                    st.rerun()

    def show_optimization_history(self):
        """Show the optimization history in the History tab."""
        st.markdown("## 📋 Optimization History")
        
        if not st.session_state.optimization_history:
            st.info("🔄 No optimization history available. Run an optimization to see results here.")
            return
        
        # Display history in reverse chronological order (newest first)
        history_data = []
        for entry in reversed(st.session_state.optimization_history):
            # Format the entry for display
            status_emoji = "✅" if entry['status'] == 'success' else "❌" if entry['status'] == 'failed' else "🔄"
            savings_display = f"${entry['total_savings']:,.2f}" if entry['total_savings'] else "N/A"
            execution_time_display = f"{entry['execution_time']:.2f}s" if entry['execution_time'] else "N/A"
            
            history_data.append({
                'Status': f"{status_emoji} {entry['status'].title()}",
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
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.optimization_history = []
                st.success("History cleared!")
                st.experimental_rerun()
        else:
            st.info("No optimization history available.")

def main():
    """Main application function."""
    app = SupplyChainOptimizer()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Main", "📊 Results", "📋 History"])
    
    with tab1:
        app.main_interface()
    
    with tab2:
        app.show_results_dashboard()
    
    with tab3:
        app.show_optimization_history()

if __name__ == "__main__":
    main()