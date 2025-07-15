"""
Configuration Management for Supply Chain Optimization Platform
Centralizes all configuration settings, environment variables, and constants.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "data/processed/optimization.db"
    
    @property
    def full_path(self) -> str:
        """Get full database path."""
        return os.path.abspath(self.path)


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    directory: str = "cache"
    ttl_hours: int = 24
    
    @property
    def ttl_seconds(self) -> int:
        """Get TTL in seconds."""
        return self.ttl_hours * 60 * 60


@dataclass
class OptimizationConfig:
    """Optimization engine configuration."""
    default_solver: str = "cbc"
    available_solvers: list = None
    timeout_seconds: int = 300
    optimality_gap: float = 0.01
    threads: int = 4
    
    def __post_init__(self):
        if self.available_solvers is None:
            self.available_solvers = ['cbc', 'glpk', 'ipopt', 'cplex', 'gurobi']


@dataclass
class AIConfig:
    """AI service configuration."""
    ollama_url: str = "http://ollama:11434/api/generate"
    default_model: str = "qwen2.5:0.5b"
    timeout_seconds: int = 300
    max_retries: int = 3


@dataclass
class UIConfig:
    """UI configuration settings."""
    page_title: str = "Supply Chain Optimizer"
    page_icon: str = "⚡"
    layout: str = "wide"
    sidebar_state: str = "collapsed"
    theme: str = "apple_silver"


@dataclass
class FileConfig:
    """File handling configuration."""
    max_file_size_mb: int = 100
    allowed_extensions: list = None
    upload_directory: str = "data/uploads"
    output_directory: str = "data/processed"
    samples_directory: str = "data/samples"
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['csv', 'xlsx', 'xls']


@dataclass
class ConstraintConfig:
    """Default constraint configuration."""
    demo_min_constraints: Dict[str, int] = None
    demo_max_constraints: Dict[str, int] = None
    default_company_volume_demo: int = 5_797_280_000
    default_company_volume_standard: int = 5_400_000_000
    
    def __post_init__(self):
        if self.demo_min_constraints is None:
            self.demo_min_constraints = {
                'Aunt Baker': 190_000_000,
                'Aunt Bethany': 2_400_000_000, 
                'Aunt Celine': 500_000_000,
                'Aunt Smith': 2_000_000_000
            }
        
        if self.demo_max_constraints is None:
            self.demo_max_constraints = {
                'Aunt Baker': 230_000_000,
                'Aunt Bethany': 3_000_000_000,
                'Aunt Celine': 700_000_000,
                'Aunt Smith': 2_400_000_000
            }


class Settings:
    """Central configuration manager."""
    
    def __init__(self):
        """Initialize settings with environment variable overrides."""
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.optimization = OptimizationConfig()
        self.ai = AIConfig()
        self.ui = UIConfig()
        self.files = FileConfig()
        self.constraints = ConstraintConfig()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Database overrides
        if os.getenv('DB_PATH'):
            self.database.path = os.getenv('DB_PATH')
        
        # Cache overrides
        if os.getenv('CACHE_DIR'):
            self.cache.directory = os.getenv('CACHE_DIR')
        if os.getenv('CACHE_TTL_HOURS'):
            self.cache.ttl_hours = int(os.getenv('CACHE_TTL_HOURS'))
        
        # Optimization overrides
        if os.getenv('DEFAULT_SOLVER'):
            self.optimization.default_solver = os.getenv('DEFAULT_SOLVER')
        if os.getenv('OPTIMIZATION_TIMEOUT'):
            self.optimization.timeout_seconds = int(os.getenv('OPTIMIZATION_TIMEOUT'))
        
        # AI overrides
        if os.getenv('OLLAMA_URL'):
            self.ai.ollama_url = os.getenv('OLLAMA_URL')
        if os.getenv('AI_MODEL'):
            self.ai.default_model = os.getenv('AI_MODEL')
        
        # File overrides
        if os.getenv('MAX_FILE_SIZE_MB'):
            self.files.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB'))
        if os.getenv('UPLOAD_DIR'):
            self.files.upload_directory = os.getenv('UPLOAD_DIR')
    
    def get_data_columns(self) -> Dict[str, list]:
        """Get required data column definitions."""
        return {
            'required_baseline': [
                'Plant_Product_Location_ID', 'Supplier', '2024 Volume (lbs)', 
                'DDP (USD)', 'Is_Baseline_Supplier'
            ],
            'required_optimization': [
                'Plant', 'Product', 'Plant Location', 'Supplier',
                'Baseline Allocated Volume', 'Baseline Price Paid'
            ],
            'template_columns': [
                'Plant', 'Product', '2024 Volume (lbs)', 'Supplier', 
                'Plant Location', 'DDP (USD)', 'Baseline Allocated Volume',
                'Baseline Price Paid', 'Selection', 'Split'
            ]
        }
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get standardized file paths."""
        base_dir = Path(__file__).parent.parent.parent
        return {
            'base_dir': str(base_dir),
            'data_dir': str(base_dir / 'data'),
            'cache_dir': str(base_dir / self.cache.directory),
            'upload_dir': str(base_dir / self.files.upload_directory),
            'output_dir': str(base_dir / self.files.output_directory),
            'template_dir': str(base_dir / 'templates'),
            'docs_dir': str(base_dir / 'docs'),
            'archive_dir': str(base_dir / 'archive')
        }
    
    def is_demo_file(self, filename: str) -> bool:
        """Check if file is a demo file."""
        if not filename:
            return False
        return any(demo_name in filename.lower() for demo_name in ['demo_data', 'demo.csv'])
    
    def get_solver_config(self, solver_name: str) -> Dict[str, Any]:
        """Get solver-specific configuration."""
        configs = {
            'cbc': {
                'seconds': self.optimization.timeout_seconds,
                'ratio': self.optimization.optimality_gap,
                'threads': self.optimization.threads,
                'cuts': 'on',
                'heuristics': 'on'
            },
            'glpk': {
                'tmlim': self.optimization.timeout_seconds,
                'mipgap': self.optimization.optimality_gap
            },
            'ipopt': {
                'max_iter': 3000,
                'tol': 1e-6
            }
        }
        return configs.get(solver_name, {})


# Global settings instance
settings = Settings()