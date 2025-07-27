#!/bin/bash

# TruthSeeQ Development Setup Script
# This script helps you manage the Docker development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to start the development environment
start_dev() {
    print_status "Starting TruthSeeQ development environment..."
    
    # Start only database services (postgres and redis)
    docker-compose up -d postgres redis
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are healthy
    if docker-compose ps | grep -q "healthy"; then
        print_success "Database services are ready!"
        print_status "PostgreSQL: localhost:5432"
        print_status "Redis: localhost:6379"
        print_status ""
        print_status "You can now run your backend locally with:"
        print_status "cd backend && python -m uvicorn app.main:app --reload"
        print_status ""
        print_status "Or run the full stack with:"
        print_status "docker-compose --profile backend up"
    else
        print_warning "Services may still be starting up. Check with: docker-compose ps"
    fi
}

# Function to start the full stack (including backend)
start_full() {
    print_status "Starting full TruthSeeQ stack..."
    docker-compose --profile backend up -d
    print_success "Full stack started!"
    print_status "API: http://localhost:8000"
    print_status "Docs: http://localhost:8000/docs"
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to restart services
restart() {
    print_status "Restarting services..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to view logs
logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, volumes, and data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to run database migrations
migrate() {
    print_status "Running database migrations..."
    cd backend
    alembic upgrade head
    print_success "Migrations completed"
}

# Function to show status
status() {
    print_status "Service Status:"
    docker-compose ps
    echo
    print_status "Database Connection Info:"
    print_status "Host: localhost"
    print_status "Port: 5432"
    print_status "Database: truthseeq"
    print_status "Username: truthseeq"
    print_status "Password: dev_password"
}

# Function to show help
show_help() {
    echo "TruthSeeQ Development Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start database services only (postgres + redis)"
    echo "  full        Start full stack including backend API"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs [service] Show logs (all services or specific service)"
    echo "  status      Show service status and connection info"
    echo "  migrate     Run database migrations"
    echo "  cleanup     Remove all containers and volumes (DESTRUCTIVE)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start databases, run backend locally"
    echo "  $0 full           # Start everything in Docker"
    echo "  $0 logs postgres  # Show PostgreSQL logs"
}

# Main script logic
case "${1:-help}" in
    start)
        check_docker
        start_dev
        ;;
    full)
        check_docker
        start_full
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    migrate)
        migrate
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac 