import os
import json
import redis
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            # Fallback to in-memory cache
            self.redis_client = None
            self.memory_cache = {}
    
    def _generate_cache_key(self, data: Any, prefix: str = "opt") -> str:
        """Generate a cache key from data."""
        if isinstance(data, pd.DataFrame):
            # For DataFrames, use column names, shape, and sample data
            key_data = {
                'columns': list(data.columns),
                'shape': data.shape,
                'sample': data.head(3).to_dict('records'),
                'dtypes': data.dtypes.to_dict()
            }
        elif isinstance(data, dict):
            key_data = data
        else:
            key_data = str(data)
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"{prefix}:{hashlib.sha256(key_string.encode()).hexdigest()}"
    
    def get_cached_result(self, file_hash: str) -> Optional[Dict]:
        """Get cached optimization result."""
        cache_key = f"optimization:{file_hash}"
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result = pickle.loads(cached_data)
                    logger.info(f"Cache hit for key: {cache_key}")
                    return result
            else:
                # Fallback to memory cache
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if entry['expiry'] > datetime.utcnow():
                        logger.info(f"Memory cache hit for key: {cache_key}")
                        return entry['data']
                    else:
                        del self.memory_cache[cache_key]
            
            logger.info(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {str(e)}")
            return None
    
    def cache_result(self, file_hash: str, result: Dict, ttl_hours: int = 24) -> bool:
        """Cache optimization result."""
        cache_key = f"optimization:{file_hash}"
        
        try:
            if self.redis_client:
                # Use Redis
                serialized_result = pickle.dumps(result)
                ttl_seconds = ttl_hours * 3600
                self.redis_client.setex(cache_key, ttl_seconds, serialized_result)
                logger.info(f"Cached result with key: {cache_key}")
            else:
                # Fallback to memory cache
                expiry = datetime.utcnow() + timedelta(hours=ttl_hours)
                self.memory_cache[cache_key] = {
                    'data': result,
                    'expiry': expiry
                }
                logger.info(f"Cached result in memory with key: {cache_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
            return False
    
    def get_similar_cached_results(self, data: pd.DataFrame, similarity_threshold: float = 0.8) -> List[Dict]:
        """Find similar cached results based on data characteristics."""
        try:
            current_features = self._extract_data_features(data)
            similar_results = []
            
            if self.redis_client:
                # Get all optimization cache keys
                keys = self.redis_client.keys("optimization:*")
                
                for key in keys:
                    try:
                        cached_data = self.redis_client.get(key)
                        if cached_data:
                            result = pickle.loads(cached_data)
                            if 'data_features' in result:
                                similarity = self._calculate_similarity(current_features, result['data_features'])
                                if similarity >= similarity_threshold:
                                    similar_results.append({
                                        'key': key.decode(),
                                        'similarity': similarity,
                                        'result': result
                                    })
                    except Exception as e:
                        logger.warning(f"Error processing cached key {key}: {str(e)}")
            
            # Sort by similarity
            similar_results.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_results[:5]  # Return top 5 similar results
            
        except Exception as e:
            logger.error(f"Error finding similar cached results: {str(e)}")
            return []
    
    def _extract_data_features(self, data: pd.DataFrame) -> Dict:
        """Extract key features from data for similarity comparison."""
        try:
            features = {
                'row_count': len(data),
                'unique_plants': data['Plant'].nunique(),
                'unique_suppliers': data['Supplier'].nunique(),
                'total_volume': data['2024 Volume (lbs)'].sum(),
                'avg_volume': data['2024 Volume (lbs)'].mean(),
                'price_range': {
                    'min': data['DDP (USD)'].min(),
                    'max': data['DDP (USD)'].max(),
                    'avg': data['DDP (USD)'].mean()
                },
                'supplier_distribution': data['Supplier'].value_counts().to_dict(),
                'plant_distribution': data['Plant'].value_counts().to_dict()
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting data features: {str(e)}")
            return {}
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets."""
        try:
            total_score = 0
            max_score = 0
            
            # Compare numeric features
            numeric_features = ['row_count', 'unique_plants', 'unique_suppliers', 'total_volume', 'avg_volume']
            for feature in numeric_features:
                if feature in features1 and feature in features2:
                    val1, val2 = features1[feature], features2[feature]
                    if val1 != 0 and val2 != 0:
                        similarity = 1 - abs(val1 - val2) / max(val1, val2)
                        total_score += max(0, similarity)
                    elif val1 == val2 == 0:
                        total_score += 1
                    max_score += 1
            
            # Compare price ranges
            if 'price_range' in features1 and 'price_range' in features2:
                price1, price2 = features1['price_range'], features2['price_range']
                for key in ['min', 'max', 'avg']:
                    if key in price1 and key in price2:
                        val1, val2 = price1[key], price2[key]
                        if val1 != 0 and val2 != 0:
                            similarity = 1 - abs(val1 - val2) / max(val1, val2)
                            total_score += max(0, similarity)
                        elif val1 == val2 == 0:
                            total_score += 1
                        max_score += 1
            
            # Compare distributions (simplified)
            for dist_key in ['supplier_distribution', 'plant_distribution']:
                if dist_key in features1 and dist_key in features2:
                    dist1, dist2 = features1[dist_key], features2[dist_key]
                    common_keys = set(dist1.keys()) & set(dist2.keys())
                    if common_keys:
                        similarity = len(common_keys) / max(len(dist1), len(dist2))
                        total_score += similarity
                    max_score += 1
            
            return total_score / max_score if max_score > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0
    
    def cache_intermediate_result(self, key: str, data: Any, ttl_minutes: int = 60) -> bool:
        """Cache intermediate processing results."""
        cache_key = f"intermediate:{key}"
        
        try:
            if self.redis_client:
                serialized_data = pickle.dumps(data)
                ttl_seconds = ttl_minutes * 60
                self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
            else:
                expiry = datetime.utcnow() + timedelta(minutes=ttl_minutes)
                self.memory_cache[cache_key] = {
                    'data': data,
                    'expiry': expiry
                }
            
            logger.info(f"Cached intermediate result with key: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching intermediate result: {str(e)}")
            return False
    
    def get_intermediate_result(self, key: str) -> Optional[Any]:
        """Get cached intermediate result."""
        cache_key = f"intermediate:{key}"
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pickle.loads(cached_data)
            else:
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if entry['expiry'] > datetime.utcnow():
                        return entry['data']
                    else:
                        del self.memory_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting intermediate result: {str(e)}")
            return None
    
    def clear_cache(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries matching pattern: {pattern}")
            else:
                # Clear memory cache
                keys_to_delete = [k for k in self.memory_cache.keys() if pattern == "*" or pattern in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                logger.info(f"Cleared {len(keys_to_delete)} memory cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'memory_usage': info.get('used_memory_human', 'N/A'),
                    'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)),
                    'cache_type': 'Redis'
                }
            else:
                return {
                    'total_keys': len(self.memory_cache),
                    'memory_usage': 'N/A',
                    'hit_rate': 'N/A',
                    'cache_type': 'Memory'
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': str(e)}