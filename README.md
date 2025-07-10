# Supply Chain Optimizer

A comprehensive AI-powered supply chain optimization platform that helps businesses minimize procurement costs while maintaining operational constraints.

## Features
- **File Upload & Processing**: Support for CSV and Excel files with automatic data validation
- **Auto-Constraint Detection**: Intelligent detection of supplier and plant constraints from baseline data
- **Mathematical Optimization**: Pyomo-based optimization engine using CBC solver
- **AI-Powered Analysis**: Ollama integration for generating actionable insights and summaries
- **Interactive Dashboard**: Real-time visualization of optimization results with before/after comparisons
- **Workflow Management**: Apache Airflow integration for complex optimization pipelines
- **Data Persistence**: PostgreSQL database with Redis caching for improved performance
- **Results Export**: Download optimized results in CSV format

## Architecture
- **Frontend**: Streamlit with custom CSS for modern UI
- **Backend**: Python with FastAPI components
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis for performance optimization
- **Optimization**: Pyomo with CBC solver
- **AI**: Ollama for natural language analysis
- **Orchestration**: Apache Airflow for workflow management
- **Containerization**: Docker Compose for easy deployment

## Services
- **Streamlit App** (Port 8501): Main web interface
- **PostgreSQL** (Port 5432): Primary database
- **Redis** (Port 6379): Caching layer
- **Ollama** (Port 11434): AI inference engine
- **Airflow Webserver** (Port 8080): Workflow management UI
- **Airflow Scheduler**: Background task processor

## Setup
1. Copy `.env.example` to `.env` and configure environment variables
2. Run `docker-compose up -d` to start all services
3. Access the application at http://localhost:8501
4. Access Airflow UI at http://localhost:8080 (admin/admin)

## Usage
1. **Upload Data**: Use the template or upload your own CSV/Excel file
2. **Review Constraints**: Auto-detected supplier and plant constraints are displayed
3. **Run Optimization**: Click to start the optimization process
4. **View Results**: Analyze savings, download results, and generate AI summaries

## Core Components
- `app/main.py`: Main Streamlit application
- `core/optimizer.py`: Pyomo-based optimization engine
- `core/database.py`: Database management and ORM
- `core/cache_manager.py`: Redis caching layer
- `core/airflow_client.py`: Airflow integration
- `airflow/dags/`: Workflow definitions
- `database/init.sql`: Database schema

## Requirements
- Docker & Docker Compose
- Python 3.9+
- 4GB+ RAM (for Ollama AI models)
