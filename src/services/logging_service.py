"""
Comprehensive Logging Service for Supply Chain Optimization Platform
Provides structured logging with multiple handlers and formatters.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from config import settings


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'file_hash'):
            log_entry['file_hash'] = record.file_hash
        if hasattr(record, 'optimization_id'):
            log_entry['optimization_id'] = record.optimization_id
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create colored message
        formatted_message = (
            f"{color}[{timestamp}] {record.levelname:8s}{reset} "
            f"{record.name:20s} | {record.getMessage()}"
        )
        
        return formatted_message


class LoggingService:
    """Comprehensive logging service."""
    
    def __init__(self):
        """Initialize logging service."""
        self.log_dir = Path(settings.get_file_paths()['base_dir']) / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
        
        # Create application loggers
        self._create_application_loggers()
    
    def _configure_root_logger(self):
        """Configure root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        all_logs_file = self.log_dir / 'application.log'
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
        
        # Error logs file
        error_logs_file = self.log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_logs_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    def _create_application_loggers(self):
        """Create specific loggers for different application components."""
        
        # Optimization logger
        optimization_logger = logging.getLogger('optimization')
        optimization_file = self.log_dir / 'optimization.log'
        optimization_handler = logging.handlers.RotatingFileHandler(
            optimization_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        optimization_handler.setFormatter(StructuredFormatter())
        optimization_logger.addHandler(optimization_handler)
        
        # Data processing logger
        data_logger = logging.getLogger('data_processing')
        data_file = self.log_dir / 'data_processing.log'
        data_handler = logging.handlers.RotatingFileHandler(
            data_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        data_handler.setFormatter(StructuredFormatter())
        data_logger.addHandler(data_handler)
        
        # UI interaction logger
        ui_logger = logging.getLogger('ui_interactions')
        ui_file = self.log_dir / 'ui_interactions.log'
        ui_handler = logging.handlers.RotatingFileHandler(
            ui_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        ui_handler.setFormatter(StructuredFormatter())
        ui_logger.addHandler(ui_handler)
        
        # Performance logger
        perf_logger = logging.getLogger('performance')
        perf_file = self.log_dir / 'performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        perf_handler.setFormatter(StructuredFormatter())
        perf_logger.addHandler(perf_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        return logging.getLogger(name)
    
    def log_optimization_start(self, optimization_id: str, file_hash: str, 
                              engine: str, data_size: int):
        """Log optimization start."""
        logger = self.get_logger('optimization')
        logger.info(
            "Optimization started",
            extra={
                'optimization_id': optimization_id,
                'file_hash': file_hash,
                'engine': engine,
                'data_size': data_size
            }
        )
    
    def log_optimization_complete(self, optimization_id: str, file_hash: str,
                                 success: bool, execution_time: float,
                                 total_savings: float = None):
        """Log optimization completion."""
        logger = self.get_logger('optimization')
        
        if success:
            logger.info(
                "Optimization completed successfully",
                extra={
                    'optimization_id': optimization_id,
                    'file_hash': file_hash,
                    'execution_time': execution_time,
                    'total_savings': total_savings
                }
            )
        else:
            logger.error(
                "Optimization failed",
                extra={
                    'optimization_id': optimization_id,
                    'file_hash': file_hash,
                    'execution_time': execution_time
                }
            )
    
    def log_file_upload(self, filename: str, file_size: int, file_hash: str):
        """Log file upload."""
        logger = self.get_logger('data_processing')
        logger.info(
            "File uploaded successfully",
            extra={
                'filename': filename,
                'file_size': file_size,
                'file_hash': file_hash
            }
        )
    
    def log_data_validation(self, file_hash: str, is_valid: bool, 
                           errors: list = None, warnings: list = None):
        """Log data validation results."""
        logger = self.get_logger('data_processing')
        
        if is_valid:
            logger.info(
                "Data validation passed",
                extra={
                    'file_hash': file_hash,
                    'warnings_count': len(warnings) if warnings else 0
                }
            )
        else:
            logger.error(
                "Data validation failed",
                extra={
                    'file_hash': file_hash,
                    'errors': errors,
                    'warnings': warnings
                }
            )
    
    def log_ui_interaction(self, action: str, component: str, 
                          session_data: Dict[str, Any] = None):
        """Log UI interaction."""
        logger = self.get_logger('ui_interactions')
        logger.info(
            f"UI interaction: {action}",
            extra={
                'action': action,
                'component': component,
                'session_data': session_data
            }
        )
    
    def log_performance(self, operation: str, duration: float, 
                       details: Dict[str, Any] = None):
        """Log performance metrics."""
        logger = self.get_logger('performance')
        logger.info(
            f"Performance: {operation}",
            extra={
                'operation': operation,
                'duration': duration,
                'details': details
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None,
                  logger_name: str = 'application'):
        """Log error with context."""
        logger = self.get_logger(logger_name)
        logger.error(
            f"Error occurred: {str(error)}",
            exc_info=True,
            extra=context or {}
        )
    
    def create_context_logger(self, base_logger_name: str, 
                             context: Dict[str, Any]) -> logging.LoggerAdapter:
        """Create a logger adapter with context."""
        logger = self.get_logger(base_logger_name)
        return logging.LoggerAdapter(logger, context)


# Global logging service instance
logging_service = LoggingService()


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return logging_service.get_logger(name)


def get_context_logger(name: str, context: Dict[str, Any]) -> logging.LoggerAdapter:
    """Get a context logger."""
    return logging_service.create_context_logger(name, context)