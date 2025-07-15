"""
UI Themes and Styling for Supply Chain Optimization Platform
Contains Apple-inspired design system and other theme options.
"""

from typing import Dict, Any


class AppleSilverTheme:
    """Apple-inspired silver gradient theme."""
    
    @staticmethod
    def get_css() -> str:
        """Get the complete CSS for Apple Silver theme."""
        return """
        <style>
            /* Apple-style silver gradient background */
            .stApp {
                background: linear-gradient(180deg, #ffffff 0%, #e8e8ed 100%) !important;
                min-height: 100vh !important;
            }
            
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(180deg, #ffffff 0%, #e8e8ed 100%) !important;
                min-height: 100vh !important;
            }
            
            [data-testid="stAppViewContainer"] > .main {
                background: linear-gradient(180deg, #ffffff 0%, #e8e8ed 100%) !important;
            }
            
            .main .block-container {
                background: transparent !important;
                padding-top: 2rem !important;
            }
            
            /* Hide Streamlit branding and menu */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Override Streamlit's default dark mode */
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff !important;
            }
            
            [data-testid="stHeader"] {
                background-color: #ffffff !important;
            }
            
            [data-testid="stSidebar"] {
                background-color: #fbfbfd !important;
            }
            
            /* Typography - Apple style */
            .main-title {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-weight: 600;
                font-size: 48px;
                line-height: 1.1;
                letter-spacing: -0.02em;
                color: #1d1d1f;
                text-align: center;
                margin-bottom: 16px;
            }
            
            .main-subtitle {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-weight: 400;
                font-size: 21px;
                line-height: 1.4;
                color: #86868b;
                text-align: center;
                margin-bottom: 48px;
            }
            
            /* Main header section with silver gradient */
            .main-header {
                text-align: center;
                padding: 80px 0 20px 0;
                background: linear-gradient(180deg, #ffffff 0%, #f5f5f7 100%);
                margin-bottom: 20px;
            }
            
            /* Button styles - Apple design */
            .stButton > button {
                background: #0071e3 !important;
                color: white !important;
                border: none !important;
                border-radius: 980px !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 400 !important;
                font-size: 17px !important;
                line-height: 1.2 !important;
                padding: 12px 22px !important;
                min-height: 44px !important;
                transition: all 0.15s ease !important;
                text-transform: none !important;
                letter-spacing: 0 !important;
                margin: 0 auto !important;
                display: block !important;
                width: fit-content !important;
            }
            
            .stButton > button:hover {
                background: #0077ed !important;
                transform: none !important;
                box-shadow: none !important;
            }
            
            .stButton > button:active {
                background: #006edb !important;
            }
            
            /* File uploader styling - minimalistic */
            .stFileUploader {
                border: 1px solid #d2d2d7 !important;
                border-radius: 12px !important;
                background: #ffffff !important;
                padding: 24px !important;
                text-align: center !important;
            }
            
            .stFileUploader:hover {
                border-color: #0071e3 !important;
                background: #fbfbfd !important;
            }
            
            .stFileUploader > div {
                border: none !important;
                background: transparent !important;
            }
            
            .stFileUploader label {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-size: 17px !important;
                font-weight: 400 !important;
                color: #1d1d1f !important;
            }
            
            .stFileUploader small {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-size: 13px !important;
                color: #86868b !important;
            }
            
            /* Metric cards with subtle gradient */
            .metric-card {
                background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
                border: 1px solid #f5f5f7;
                border-radius: 18px;
                padding: 24px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            }
            
            /* Data frame styling */
            .stDataFrame {
                border: 1px solid #f5f5f7;
                border-radius: 12px;
                overflow: hidden;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: #f5f5f7;
                border-radius: 12px;
                padding: 4px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 44px;
                background: transparent;
                border-radius: 8px;
                padding: 0 16px;
                color: #1d1d1f;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-weight: 400;
                font-size: 17px;
                border: none;
            }
            
            .stTabs [aria-selected="true"] {
                background: #ffffff;
                color: #1d1d1f;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            /* Section headers */
            h2 {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 600 !important;
                font-size: 32px !important;
                line-height: 1.125 !important;
                letter-spacing: -0.01em !important;
                color: #1d1d1f !important;
                margin-top: 48px !important;
                margin-bottom: 24px !important;
            }
            
            h3 {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 600 !important;
                font-size: 24px !important;
                line-height: 1.17 !important;
                color: #1d1d1f !important;
                margin-top: 32px !important;
                margin-bottom: 16px !important;
            }
            
            /* Text elements */
            .stMarkdown p {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-size: 17px !important;
                line-height: 1.47 !important;
                color: #1d1d1f !important;
            }
            
            /* Info boxes */
            .stInfo {
                background: #f5f5f7 !important;
                border: 1px solid #d2d2d7 !important;
                border-radius: 12px !important;
                color: #1d1d1f !important;
            }
            
            .stSuccess {
                background: #d4edda !important;
                border: 1px solid #c3e6cb !important;
                border-radius: 12px !important;
                color: #155724 !important;
            }
            
            .stWarning {
                background: #fff3cd !important;
                border: 1px solid #ffeaa7 !important;
                border-radius: 12px !important;
                color: #856404 !important;
            }
            
            .stError {
                background: #f8d7da !important;
                border: 1px solid #f5c6cb !important;
                border-radius: 12px !important;
                color: #721c24 !important;
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 400 !important;
                font-size: 17px !important;
                color: #1d1d1f !important;
                background: #fbfbfd !important;
                border: 1px solid #f5f5f7 !important;
                border-radius: 12px !important;
            }
            
            /* Remove borders and box shadows for cleaner look */
            .stContainer {
                border: none !important;
                box-shadow: none !important;
            }
            
            /* Clean up sidebar with silver gradient */
            .stSidebar {
                background: linear-gradient(180deg, #f5f5f7 0%, #e8e8ed 100%);
                border-right: 1px solid #d2d2d7;
            }
            
            /* Input field styling */
            .stNumberInput input {
                border: 1px solid #d2d2d7 !important;
                border-radius: 8px !important;
                background: #ffffff !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-size: 17px !important;
            }
            
            /* Download button styling */
            .stDownloadButton > button {
                background: #ffffff !important;
                color: #0071e3 !important;
                border: 1px solid #0071e3 !important;
                border-radius: 980px !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
                font-weight: 400 !important;
                font-size: 17px !important;
                padding: 12px 22px !important;
                min-height: 44px !important;
            }
            
            .stDownloadButton > button:hover {
                background: #f5f5f7 !important;
                color: #0077ed !important;
                border-color: #0077ed !important;
            }
        </style>
        """


class ThemeManager:
    """Manages UI themes and styling."""
    
    themes = {
        'apple_silver': AppleSilverTheme,
    }
    
    @classmethod
    def get_theme_css(cls, theme_name: str = 'apple_silver') -> str:
        """Get CSS for specified theme."""
        theme_class = cls.themes.get(theme_name, AppleSilverTheme)
        return theme_class.get_css()
    
    @classmethod
    def apply_theme(cls, st_instance, theme_name: str = 'apple_silver'):
        """Apply theme to Streamlit app."""
        css = cls.get_theme_css(theme_name)
        st_instance.markdown(css, unsafe_allow_html=True)