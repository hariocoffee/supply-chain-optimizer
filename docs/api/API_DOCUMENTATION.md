# Supply Chain Optimization API Documentation

## Overview
RESTful API for supply chain optimization with real-time processing capabilities.

## Base URL
```
http://localhost:8501/api/v1
```

## Authentication
Currently using session-based authentication. Future versions will implement JWT tokens.

## Core Endpoints

### Optimization Engine

#### POST /optimize
Execute supply chain optimization algorithm.

**Request:**
```json
{
  "data": "base64_encoded_csv",
  "constraints": {
    "max_suppliers_per_plant": 3,
    "min_volume_threshold": 1000
  },
  "optimization_type": "cost_minimization"
}
```

**Response:**
```json
{
  "optimization_id": "opt_123456",
  "status": "completed",
  "total_savings": 2673320400.00,
  "plants_optimized": 27,
  "suppliers_involved": 156,
  "execution_time_ms": 1247,
  "results_url": "/api/v1/results/opt_123456"
}
```

#### GET /results/{optimization_id}
Retrieve optimization results.

**Response:**
```json
{
  "optimization_id": "opt_123456",
  "results": {
    "baseline_cost": 45629415800.00,
    "optimized_cost": 42956095400.00,
    "total_savings": 2673320400.00,
    "savings_percentage": 5.86,
    "allocations": [...]
  },
  "metadata": {
    "created_at": "2025-07-12T00:00:00Z",
    "execution_time": 1247,
    "algorithm": "or_tools_linear_programming"
  }
}
```

### AI Analysis

#### POST /ai/analyze
Generate AI-powered insights from optimization results.

**Request:**
```json
{
  "optimization_id": "opt_123456",
  "analysis_type": "executive_summary"
}
```

**Response:**
```json
{
  "analysis_id": "ai_789012",
  "summary": "Executive summary text...",
  "key_insights": [
    "Primary cost driver: Supplier consolidation",
    "Risk factor: Geographic concentration"
  ],
  "recommendations": [
    "Negotiate volume discounts with Aunt Smith",
    "Diversify supplier base in Location7"
  ]
}
```

### Data Management

#### POST /data/upload
Upload supply chain data for optimization.

**Request:** Multipart form data with CSV file

**Response:**
```json
{
  "upload_id": "upload_345678",
  "status": "processed",
  "records_count": 1547,
  "validation_results": {
    "errors": 0,
    "warnings": 2,
    "data_quality_score": 0.98
  }
}
```

#### GET /data/validate/{upload_id}
Validate uploaded data quality.

**Response:**
```json
{
  "validation_id": "val_456789",
  "status": "completed",
  "results": {
    "total_records": 1547,
    "valid_records": 1545,
    "error_records": 2,
    "warnings": [
      "Missing plant location for 2 records"
    ]
  }
}
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "OPTIMIZATION_FAILED",
    "message": "Optimization algorithm failed to converge",
    "details": {
      "reason": "Infeasible constraints",
      "suggestions": ["Review capacity constraints", "Check demand requirements"]
    }
  },
  "request_id": "req_123456",
  "timestamp": "2025-07-12T00:00:00Z"
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Rate Limiting
- 100 requests per minute per IP
- 10 optimization requests per hour per user
- Burst limit: 20 requests in 10 seconds

## WebSocket Events

### Real-time Optimization Updates
```javascript
ws://localhost:8501/ws/optimization/{optimization_id}

// Events
{
  "event": "optimization_progress",
  "data": {
    "progress_percentage": 45,
    "current_step": "constraint_processing",
    "estimated_completion": "2025-07-12T00:02:30Z"
  }
}
```

## SDK Examples

### Python
```python
import requests

# Execute optimization
response = requests.post(
    "http://localhost:8501/api/v1/optimize",
    json={
        "data": base64_data,
        "constraints": {"max_suppliers_per_plant": 3}
    }
)

# Get results
optimization_id = response.json()["optimization_id"]
results = requests.get(f"http://localhost:8501/api/v1/results/{optimization_id}")
```

### JavaScript
```javascript
// Execute optimization
const response = await fetch('/api/v1/optimize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    data: base64Data,
    constraints: {max_suppliers_per_plant: 3}
  })
});

const result = await response.json();
```