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
        st.markdown("## 🎯 Set Constraints")
        
        df = st.session_state.data
        
        # Get unique suppliers
        suppliers = df['Supplier'].unique()
        
        st.markdown("### Supplier Volume Constraints")
        
        supplier_constraints = {}
        
        for supplier in suppliers:
            with st.container():
                st.markdown(f"**{supplier}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_vol = st.number_input(
                        f"Min Volume (lbs)",
                        min_value=0,
                        key=f"min_{supplier}",
                        format="%d"
                    )
                
                with col2:
                    max_vol = st.number_input(
                        f"Max Volume (lbs)",
                        min_value=min_vol,
                        key=f"max_{supplier}",
                        format="%d"
                    )
                
                supplier_constraints[supplier] = {'min': min_vol, 'max': max_vol}
        
        # Plant constraints
        st.markdown("### Plant Maximum Volume Constraints")
        plants = df['Plant'].unique()
        plant_constraints = {}
        
        for plant in plants:
            plant_max = st.number_input(
                f"Max Volume for {plant} (lbs)",
                min_value=0,
                key=f"plant_max_{plant}",
                format="%d"
            )
            plant_constraints[plant] = plant_max
        
        if st.button("✅ Set Constraints", key="set_constraints"):
            # Store constraints in session state
            st.session_state.supplier_constraints = supplier_constraints
            st.session_state.plant_constraints = plant_constraints
            st.session_state.constraints_set = True
            
            # Store in database
            self.db_manager.store_constraints(
                st.session_state.file_hash,
                supplier_constraints,
                plant_constraints
            )
            
            st.success("✅ Constraints set successfully!")

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
                        st.session_state.results = results
                        st.session_state.optimization_complete = True
                        
                        # Cache the results
                        self.cache_manager.cache_result(
                            st.session_state.file_hash,
                            results
                        )
                        
                        # Store in database
                        self.db_manager.store_optimization_results(
                            st.session_state.file_hash,
                            results
                        )
                        
                        st.success("✅ Optimization completed successfully!")
                        
                        # Show download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📥 Download Results", key="download_results"):
                                self.download_results()
                        
                        with col2:
                            if st.button("🤖 Generate AI Summary", key="generate_summary"):
                                self.generate_ai_summary()
                    else:
                        st.error("❌ Optimization failed. Please check your constraints.")
                else:
                    st.error("❌ Failed to start optimization process.")

    def download_results(self):
        """Generate and download the results file."""
        if st.session_state.results:
            results_df = st.session_state.results['optimized_data']
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="Download Optimized Results",
                data=csv,
                file_name=f"optimized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

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
            st.metric(
                "Volume Reallocated",
                f"{results['volume_reallocated']:,.0f} lbs"
            )
        
        # Before/After Comparison
        st.markdown("## 📈 Before vs After Comparison")
        
        comparison_data = results['comparison_data']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cost Comparison", "Volume Distribution", "Supplier Utilization", "Savings by Plant"),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Cost comparison
        fig.add_trace(
            go.Bar(name="Baseline", x=["Total Cost"], y=[results['baseline_cost']]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Optimized", x=["Total Cost"], y=[results['optimized_cost']]),
            row=1, col=1
        )
        
        # Volume distribution pie chart
        fig.add_trace(
            go.Pie(labels=list(results['supplier_volumes'].keys()),
                   values=list(results['supplier_volumes'].values()),
                   name="Volume Distribution"),
            row=1, col=2
        )
        
        # Supplier utilization
        suppliers = list(results['supplier_utilization'].keys())
        utilization = list(results['supplier_utilization'].values())
        
        fig.add_trace(
            go.Bar(name="Utilization %", x=suppliers, y=utilization),
            row=2, col=1
        )
        
        # Savings by plant
        plants = list(results['savings_by_plant'].keys())
        savings = list(results['savings_by_plant'].values())
        
        fig.add_trace(
            go.Bar(name="Savings", x=plants, y=savings),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Changes Table
        st.markdown("## 📋 Detailed Changes")
        
        changes_df = results['changes_summary']
        
        # Highlight savings in green
        def highlight_savings(val):
            if isinstance(val, (int, float)) and val > 0:
                return 'background-color: #d4edda; color: #155724;'
            return ''
        
        styled_df = changes_df.style.applymap(highlight_savings, subset=['Cost Savings'])
        st.dataframe(styled_df, use_container_width=True)

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