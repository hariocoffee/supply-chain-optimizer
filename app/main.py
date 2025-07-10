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

    def generate_file_hash(self, data: pd.DataFrame) -> str:
        """Generate a hash for the uploaded data for caching purposes."""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()

    def create_template_file(self) -> bytes:
        """Create a template file based on the expected structure."""
        template_data = {
            'Plant': ['Alpha', 'Beta', 'Charlie'],
            'Product': ['Brownies', 'Brownies', 'Brownies'],
            '2024 Volume (lbs)': [5900000, 15900000, 56800000],
            'Supplier': ['Aunt Smith', 'Aunt Smith', 'Aunt Bethany'],
            'Plant Location': ['Location1', 'Location2', 'Location3'],
            'DDP (USD)': [8.05, 6.91, 9.28],
            'Baseline Allocated Volume': [5900000, 15900000, 56800000],
            'Baseline Price Paid': [47495000, 109869000, 527104000],
            'Selection': ['X', 'X', 'X'],
            'Split': ['100%', '100%', '100%'],
            'Optimized Volume': ['', '', ''],
            'Optimized Price': ['', '', ''],
            'Optimized Selection': ['', '', ''],
            'Optimized Split': ['', '', ''],
            'Cost Savings': ['', '', '']
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
                st.success("✅ Similar data found in cache! Loading previous results...")
                st.session_state.data = df
                st.session_state.file_hash = file_hash
                st.session_state.results = cached_result
                st.session_state.optimization_complete = True
                return
            
            # Store in session state
            st.session_state.data = df
            st.session_state.file_hash = file_hash
            
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
                    st.metric("Unique Plants", df['Plant'].nunique())
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
        st.markdown("### 📊 Supplier Constraints (Auto-Detected)")
        
        constraint_data = []
        for supplier, constraints in supplier_constraints.items():
            baseline_volume = constraints['baseline_volume']
            baseline_cost = constraints['baseline_cost']
            constraint_data.append({
                'Supplier': supplier,
                'Baseline Volume (lbs)': f"{baseline_volume:,.0f}",
                'Baseline Cost ($)': f"${baseline_cost:,.2f}",
                'Max Volume Constraint (lbs)': f"{constraints['max']:,.0f}",
                'Min Volume Constraint (lbs)': f"{constraints['min']:,.0f}"
            })
        
        constraints_df = pd.DataFrame(constraint_data)
        st.dataframe(constraints_df, use_container_width=True)
        
        # Info about plant requirements
        st.markdown("### 🏭 Plant Requirements")
        st.info("**Each plant MUST receive its full allocated volume as specified in Column C (2024 Volume).**")
        
        plant_requirements = df.groupby('Plant')['2024 Volume (lbs)'].first().reset_index()
        plant_requirements['Required Volume (lbs)'] = plant_requirements['2024 Volume (lbs)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(plant_requirements[['Plant', 'Required Volume (lbs)']], use_container_width=True)
        
        if st.button("✅ Use Auto-Detected Constraints", key="set_constraints"):
            # Store constraints in session state
            st.session_state.supplier_constraints = supplier_constraints
            st.session_state.plant_constraints = {}  # No plant constraints needed
            st.session_state.constraints_set = True
            
            # Store in database
            self.db_manager.store_constraints(
                st.session_state.file_hash,
                supplier_constraints,
                {}  # No plant constraints
            )
            
            st.success("✅ Auto-detected constraints applied successfully!")
    
    def calculate_supplier_constraints(self, df):
        """Calculate supplier constraints from baseline data."""
        supplier_constraints = {}
        
        # Group by supplier and sum baseline allocated volume and price
        supplier_data = df.groupby('Supplier').agg({
            'Baseline Allocated Volume': 'sum',
            'Baseline Price Paid': 'sum'
        }).reset_index()
        
        for _, row in supplier_data.iterrows():
            supplier = row['Supplier']
            baseline_volume = row['Baseline Allocated Volume']
            baseline_cost = row['Baseline Price Paid']
            
            supplier_constraints[supplier] = {
                'min': 0,  # Minimum can be 0 (supplier might not be used)
                'max': baseline_volume,  # Maximum is current baseline allocation
                'baseline_volume': baseline_volume,
                'baseline_cost': baseline_cost
            }
        
        return supplier_constraints

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
                    
                    # Run the actual optimization
                    results = self.optimizer.optimize(
                        st.session_state.data,
                        st.session_state.supplier_constraints,
                        st.session_state.plant_constraints
                    )
                    
                    if results:
                        # Calculate volume reallocated
                        baseline_total_volume = st.session_state.data['Baseline Allocated Volume'].sum()
                        optimized_total_volume = results.get('results_dataframe', pd.DataFrame()).get('Optimized Volume', pd.Series()).sum() if 'results_dataframe' in results else 0
                        volume_reallocated = abs(optimized_total_volume - baseline_total_volume)
                        
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
                            'plants_optimized': results.get('optimization_summary', {}).get('total_plants', 0),
                            'suppliers_involved': results.get('optimization_summary', {}).get('total_suppliers', 0),
                            'volume_redistributions': len(results.get('detailed_results', [])),
                            'supplier_switches': len([r for r in results.get('detailed_results', []) if r.get('selection_flag')]),
                            'volume_reallocated': volume_reallocated
                        }
                        
                        st.session_state.results = processed_results
                        st.session_state.optimization_complete = True
                        
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
                        st.error("❌ Optimization failed. Please check your constraints.")
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
            st.metric(
                "Plants Optimized",
                results['plants_optimized']
            )
        
        with col3:
            st.metric(
                "Suppliers Involved",
                results['suppliers_involved']
            )
        
        with col4:
            volume_reallocated = results.get('volume_reallocated', 0)
            st.metric(
                "Volume Reallocated",
                f"{volume_reallocated:,.0f} lbs"
            )
        
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

def main():
    """Main application function."""
    app = SupplyChainOptimizer()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏠 Main", "📊 Results"])
    
    with tab1:
        app.main_interface()
    
    with tab2:
        app.show_results_dashboard()

if __name__ == "__main__":
    main()