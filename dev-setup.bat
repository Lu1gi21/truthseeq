@echo off
setlocal enabledelayedexpansion

REM TruthSeeQ Development Setup Script for Windows
REM This script helps you manage the Docker development environment

set "SCRIPT_NAME=%~n0"

REM Function to print colored output (Windows compatible)
:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Function to check if Docker is running
:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker Desktop and try again."
    exit /b 1
)
call :print_success "Docker is running"
goto :eof

REM Function to start the development environment
:start_dev
call :print_status "Starting TruthSeeQ development environment..."
docker-compose up -d postgres redis
if errorlevel 1 (
    call :print_error "Failed to start services"
    exit /b 1
)

call :print_status "Waiting for services to be ready..."
timeout /t 10 /nobreak >nul

call :print_success "Database services are ready!"
call :print_status "PostgreSQL: localhost:5432"
call :print_status "Redis: localhost:6379"
echo.
call :print_status "You can now run your backend locally with:"
call :print_status "cd backend && python -m uvicorn app.main:app --reload"
echo.
call :print_status "Or run the full stack with:"
call :print_status "docker-compose --profile backend up"
goto :eof

REM Function to start the full stack
:start_full
call :print_status "Starting full TruthSeeQ stack..."
docker-compose --profile backend up -d
if errorlevel 1 (
    call :print_error "Failed to start full stack"
    exit /b 1
)
call :print_success "Full stack started!"
call :print_status "API: http://localhost:8000"
call :print_status "Docs: http://localhost:8000/docs"
goto :eof

REM Function to stop all services
:stop_all
call :print_status "Stopping all services..."
docker-compose down
call :print_success "All services stopped"
goto :eof

REM Function to restart services
:restart
call :print_status "Restarting services..."
docker-compose restart
call :print_success "Services restarted"
goto :eof

REM Function to view logs
:logs
if "%~1"=="" (
    docker-compose logs -f
) else (
    docker-compose logs -f %~1
)
goto :eof

REM Function to clean up everything
:cleanup
call :print_warning "This will remove all containers, volumes, and data!"
set /p "confirm=Are you sure? (y/N): "
if /i "!confirm!"=="y" (
    call :print_status "Cleaning up..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    call :print_success "Cleanup completed"
) else (
    call :print_status "Cleanup cancelled"
)
goto :eof

REM Function to run database migrations
:migrate
call :print_status "Running database migrations..."
cd backend
alembic upgrade head
if errorlevel 1 (
    call :print_error "Migrations failed"
    exit /b 1
)
call :print_success "Migrations completed"
cd ..
goto :eof

REM Function to show status
:status
call :print_status "Service Status:"
docker-compose ps
echo.
call :print_status "Database Connection Info:"
call :print_status "Host: localhost"
call :print_status "Port: 5432"
call :print_status "Database: truthseeq"
call :print_status "Username: truthseeq"
call :print_status "Password: dev_password"
goto :eof

REM Function to show help
:show_help
echo TruthSeeQ Development Setup Script for Windows
echo.
echo Usage: %SCRIPT_NAME% [COMMAND]
echo.
echo Commands:
echo   start       Start database services only (postgres + redis)
echo   full        Start full stack including backend API
echo   stop        Stop all services
echo   restart     Restart all services
echo   logs [service] Show logs (all services or specific service)
echo   status      Show service status and connection info
echo   migrate     Run database migrations
echo   cleanup     Remove all containers and volumes (DESTRUCTIVE)
echo   help        Show this help message
echo.
echo Examples:
echo   %SCRIPT_NAME% start          # Start databases, run backend locally
echo   %SCRIPT_NAME% full           # Start everything in Docker
echo   %SCRIPT_NAME% logs postgres  # Show PostgreSQL logs
goto :eof

REM Main script logic
if "%~1"=="" goto show_help
if "%~1"=="help" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help

if "%~1"=="start" (
    call check_docker
    call start_dev
    goto :eof
)

if "%~1"=="full" (
    call check_docker
    call start_full
    goto :eof
)

if "%~1"=="stop" (
    call stop_all
    goto :eof
)

if "%~1"=="restart" (
    call restart
    goto :eof
)

if "%~1"=="logs" (
    call logs %~2
    goto :eof
)

if "%~1"=="status" (
    call status
    goto :eof
)

if "%~1"=="migrate" (
    call migrate
    goto :eof
)

if "%~1"=="cleanup" (
    call cleanup
    goto :eof
)

call :print_error "Unknown command: %~1"
echo.
call show_help
exit /b 1 