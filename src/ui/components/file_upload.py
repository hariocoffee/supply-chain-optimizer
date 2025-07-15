"""
File Upload Component for Supply Chain Optimization Platform
Handles file uploading, validation, and processing interface.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from config import settings
from services import DataProcessor, ValidationResult


class FileUploadComponent:
    """Handles file upload interface and processing."""
    
    def __init__(self, data_processor: DataProcessor):
        """Initialize file upload component."""
        self.data_processor = data_processor
    
    def render_header(self):
        """Render the main header section."""
        st.markdown("""
        <div class="main-header">
            <h1 class="main-title">Supply Chain Optimizer</h1>
            <p class="main-subtitle">Optimize your supply chain with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_template_download(self):
        """Render template download section."""
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Download Template", key="download_template", use_container_width=True):
                template_file = self._create_template_file()
                st.download_button(
                    label="Download CSV Template",
                    data=template_file,
                    file_name="supply_chain_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    def render_file_uploader(self, on_file_processed: Optional[Callable] = None):
        """
        Render file upload interface.
        
        Args:
            on_file_processed: Callback function to call when file is processed
        """
        st.markdown("### Upload Data File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=settings.files.allowed_extensions,
            key="file_uploader",
            help="Upload your supply chain data file (CSV or Excel format)"
        )
        
        if uploaded_file:
            if st.button("Continue", key="next_button"):
                result = self.process_file(uploaded_file)
                if result and on_file_processed:
                    on_file_processed(result)
                st.rerun()
    
    def process_file(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """
        Process uploaded file and return processing result.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with processing results or None if failed
        """
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate data using data processor
            validation_result = self.data_processor.validate_data(df, uploaded_file.name)
            
            # Show validation messages
            if validation_result.errors:
                for error in validation_result.errors:
                    st.error(f"Data validation error: {error}")
                return None
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    st.warning(f"Data validation warning: {warning}")
            
            # Use processed data if validation succeeded
            if validation_result.processed_data is not None:
                df = validation_result.processed_data
            
            # Generate file hash for caching
            file_hash = self.data_processor.generate_file_hash(df)
            
            # Create processing result
            result = {
                'data': df,
                'file_hash': file_hash,
                'filename': uploaded_file.name,
                'validation_result': validation_result,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    def render_processing_status(self, show_cache_message: bool = False, 
                               show_success_message: bool = False, 
                               filename: str = ""):
        """Render file processing status messages."""
        if show_cache_message:
            st.info("This file has been optimized before. Previous results are available in the History tab.")
        
        if show_success_message and filename:
            st.success(f"File '{filename}' processed successfully!")
    
    def render_data_preview(self, data: pd.DataFrame):
        """Render data preview section with metrics."""
        if data is None or data.empty:
            return
        
        with st.expander("Data Preview", expanded=True):
            st.dataframe(data.head(10))
            
            # Get data summary using data processor
            summary = self.data_processor.get_data_summary(data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", summary.get('total_rows', 0))
            with col2:
                st.metric("Columns", summary.get('total_columns', 0))
            with col3:
                st.metric("Unique Plant-Product-Locations", summary.get('unique_locations', 0))
            with col4:
                st.metric("Unique Suppliers", summary.get('unique_suppliers', 0))
    
    def _create_template_file(self) -> bytes:
        """Create template file for download."""
        template_df = self.data_processor.create_template_data()
        return template_df.to_csv(index=False).encode()
    
    def render_complete_interface(self, on_file_processed: Optional[Callable] = None):
        """Render complete file upload interface."""
        self.render_header()
        self.render_template_download()
        self.render_file_uploader(on_file_processed)
        
        # Show processing status if available in session state
        if hasattr(st.session_state, 'file_processing_status'):
            status = st.session_state.file_processing_status
            self.render_processing_status(
                show_cache_message=status.get('show_cache_message', False),
                show_success_message=status.get('show_success_message', False),
                filename=status.get('filename', '')
            )
        
        # Show data preview if data is available
        if hasattr(st.session_state, 'data') and st.session_state.data is not None:
            self.render_data_preview(st.session_state.data)