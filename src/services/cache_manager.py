"""
Cache Manager for Supply Chain Optimization
Handles caching of optimization results for improved performance
"""

import pickle
import os
import time
import logging
from typing import Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of optimization results."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        self.cache_ttl = 24 * 60 * 60  # 24 hours in seconds
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file_path(self, file_hash: str) -> str:
        """Get the cache file path for a given file hash."""
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")
    
    def cache_result(self, file_hash: str, results: Dict[str, Any]) -> bool:
        """Cache optimization results."""
        try:
            cache_file = self._get_cache_file_path(file_hash)
            
            cache_data = {
                'timestamp': time.time(),
                'results': results
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Results cached for hash: {file_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching results: {str(e)}")
            return False
    
    def get_cached_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached optimization results."""
        try:
            cache_file = self._get_cache_file_path(file_hash)
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is still valid
            if time.time() - cache_data['timestamp'] > self.cache_ttl:
                logger.info(f"Cache expired for hash: {file_hash}")
                os.remove(cache_file)
                return None
            
            logger.info(f"Cache hit for hash: {file_hash}")
            return cache_data['results']
            
        except Exception as e:
            logger.error(f"Error retrieving cached results: {str(e)}")
            return None
    
    def clear_cache(self) -> bool:
        """Clear all cached results."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            
            return {
                'total_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'total_files': 0, 'total_size_bytes': 0, 'total_size_mb': 0}