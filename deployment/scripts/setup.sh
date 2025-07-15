#!/bin/bash
set -e

echo "🚀 Setting up Supply Chain Optimization Platform..."

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Setup environment
setup_environment() {
    echo "🔧 Setting up environment..."
    
    if [ ! -f .env ]; then
        echo "📝 Creating .env file..."
        cat > .env << EOF
# Database Configuration
POSTGRES_DB=supply_chain_optimization
POSTGRES_USER=sc_admin
POSTGRES_PASSWORD=$(openssl rand -hex 16)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -hex 16)

# Ollama Configuration
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5:0.5b

# Application Configuration
SECRET_KEY=$(openssl rand -hex 32)
LOG_LEVEL=INFO
ENVIRONMENT=development

# Cache Configuration
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
EOF
        echo "✅ Environment file created"
    else
        echo "✅ Environment file already exists"
    fi
}

# Create necessary directories
create_directories() {
    echo "📁 Creating directories..."
    
    mkdir -p data/{input,output}
    mkdir -p logs
    mkdir -p cache
    mkdir -p uploads
    mkdir -p tests/{unit,integration,demo}
    mkdir -p docs/{api,architecture,deployment}
    mkdir -p scripts
    mkdir -p config
    
    echo "✅ Directories created"
}

# Build and start services
start_services() {
    echo "🐳 Building and starting Docker services..."
    
    # Build the application
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        echo "✅ Services are running"
    else
        echo "❌ Some services failed to start"
        docker-compose logs
        exit 1
    fi
}

# Initialize database
initialize_database() {
    echo "🗄️ Initializing database..."
    
    # Wait for PostgreSQL to be ready
    docker-compose exec -T postgres sh -c 'until pg_isready -U $POSTGRES_USER; do sleep 1; done'
    
    # Run initialization script if exists
    if [ -f database/init.sql ]; then
        docker-compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB < database/init.sql
        echo "✅ Database initialized"
    else
        echo "ℹ️ No database initialization script found"
    fi
}

# Setup Ollama model
setup_ollama() {
    echo "🤖 Setting up Ollama AI model..."
    
    # Wait for Ollama to be ready
    sleep 10
    
    # Pull the model
    docker-compose exec ollama ollama pull qwen2.5:0.5b
    
    echo "✅ Ollama model ready"
}

# Verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    # Check application health
    if curl -f http://localhost:8501/health &> /dev/null; then
        echo "✅ Application is responding"
    else
        echo "⚠️ Application health check failed"
    fi
    
    # Check database connection
    if docker-compose exec -T postgres pg_isready -U $POSTGRES_USER &> /dev/null; then
        echo "✅ Database is ready"
    else
        echo "⚠️ Database connection failed"
    fi
    
    # Check Redis connection
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        echo "✅ Redis is ready"
    else
        echo "⚠️ Redis connection failed"
    fi
}

# Display success message
display_success() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📊 Your Supply Chain Optimization Platform is ready at:"
    echo "   🌐 Web Application: http://localhost:8501"
    echo "   📚 API Documentation: http://localhost:8501/docs"
    echo ""
    echo "🔧 Useful commands:"
    echo "   📋 View logs: docker-compose logs -f"
    echo "   🔄 Restart services: docker-compose restart"
    echo "   🛑 Stop services: docker-compose down"
    echo "   🧹 Clean up: docker-compose down -v --remove-orphans"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Upload your supply chain data via the web interface"
    echo "   2. Configure optimization parameters"
    echo "   3. Run optimization and review AI-generated insights"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_environment
    create_directories
    start_services
    initialize_database
    setup_ollama
    verify_installation
    display_success
}

# Error handling
trap 'echo "❌ Setup failed. Check the logs for details."; exit 1' ERR

# Run main function
main "$@"