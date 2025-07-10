import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Data Summarizer",
    page_icon="📊",
    layout="wide"
)

def query_ollama(prompt, model="qwen2.5:0.5b"): 
    """
    Send a request to Ollama API to get LLM response
    """
    try:
        url = "http://ollama:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=1200)  # 20 minutes
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}. Make sure Ollama is running."

def process_dataframe(df):
    """
    Process dataframe and create a summary for the LLM
    """
    # Basic info about the dataset
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict()  # Reduced to 5 rows for shorter prompt
    }
    
    # Create a more concise summary for the LLM
    summary_text = f"""
    Dataset Summary:
    - Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
    - Columns: {', '.join(info['columns'])}
    - Missing Values: {info['missing_values']}
    
    Sample Data (first 5 rows):
    {df.head(5).to_string()}
    """
    
    return summary_text

def main():
    st.title("📊 Data Summarizer with Local LLM")
    st.markdown("Upload CSV or Excel files to get AI-powered summaries using your local Qwen2.5 model")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel files
                df = pd.read_excel(uploaded_file)
            
            # Display basic file info
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Show basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Generate summary button
            if st.button("🤖 Generate AI Summary", type="primary"):
                with st.spinner("Generating summary with Qwen2.5... This may take a few minutes..."):
                    # Process dataframe
                    data_summary = process_dataframe(df)
                    
                    # Create shorter, more focused prompt for LLM
                    prompt = f"""
                    Analyze this dataset and provide a summary including:
                    1. Data overview and structure
                    2. Key insights about the data
                    3. Data quality assessment
                    4. Notable patterns or trends
                    5. Suggestions for further analysis
                    
                    Dataset Information:
                    {data_summary}
                    
                    Provide a clear, concise analysis that would be useful for understanding this dataset.
                    """
                    
                    # Get LLM response
                    summary = query_ollama(prompt)
                    
                    # Display summary
                    st.subheader("🎯 AI-Generated Summary")
                    st.markdown(summary)
                    
                    # Option to download summary
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name=f"summary_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
            
            # Show detailed data info
            with st.expander("🔍 Detailed Data Information"):
                st.subheader("Data Types")
                st.write(df.dtypes)
                
                st.subheader("Missing Values")
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    st.write(missing_data[missing_data > 0])
                else:
                    st.write("No missing values found!")
                
                st.subheader("Statistical Summary")
                st.write(df.describe())
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file is a valid CSV or Excel file.")
    
    else:
        st.info("👆 Please upload a CSV or Excel file to get started.")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. **Upload your data file** (CSV or Excel)
        2. **Review the data preview** to ensure it loaded correctly
        3. **Click "Generate AI Summary"** to get insights from your local Qwen2.5 model
        4. **Download the summary** if needed
        
        ### Requirements:
        - Ollama must be running with Qwen2.5 model
        - Supported file formats: CSV, XLSX, XLS
        - Processing may take a few minutes for larger datasets
        """)

if __name__ == "__main__":
    main()