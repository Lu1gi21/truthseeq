# TruthSeeQ Docker Development Setup

This setup provides a simple Docker environment for testing your TruthSeeQ backend without needing to install PostgreSQL or Redis locally.

## Quick Start

### Option 1: Start Databases Only (Recommended for Development)

1. **Start the database services:**
   ```bash
   # On Linux/Mac:
   ./dev-setup.sh start
   
   # On Windows:
   dev-setup.bat start
   ```

2. **Run your backend locally:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```

### Option 2: Start Full Stack (Everything in Docker)

```bash
# On Linux/Mac:
./dev-setup.sh full

# On Windows:
dev-setup.bat full
```

This starts everything including the backend API at http://localhost:8000

## What's Included

- **PostgreSQL 15**: Database for TruthSeeQ
  - Host: `localhost`
  - Port: `5432`
  - Database: `truthseeq`
  - Username: `truthseeq`
  - Password: `dev_password`

- **Redis 7**: Caching and rate limiting
  - Host: `localhost`
  - Port: `6379`

- **Backend API** (optional): FastAPI application
  - URL: `http://localhost:8000`
  - Docs: `http://localhost:8000/docs`

## Available Commands

### Using the Setup Scripts

```bash
# Start database services only
./dev-setup.sh start          # Linux/Mac
dev-setup.bat start           # Windows

# Start full stack
./dev-setup.sh full           # Linux/Mac
dev-setup.bat full            # Windows

# Stop all services
./dev-setup.sh stop           # Linux/Mac
dev-setup.bat stop            # Windows

# View logs
./dev-setup.sh logs           # All services
./dev-setup.sh logs postgres  # PostgreSQL only
dev-setup.bat logs            # Windows

# Check status
./dev-setup.sh status         # Linux/Mac
dev-setup.bat status          # Windows

# Run migrations
./dev-setup.sh migrate        # Linux/Mac
dev-setup.bat migrate         # Windows

# Clean up everything (DESTRUCTIVE)
./dev-setup.sh cleanup        # Linux/Mac
dev-setup.bat cleanup         # Windows
```

### Using Docker Compose Directly

```bash
# Start only databases
docker-compose up -d postgres redis

# Start everything
docker-compose --profile backend up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Remove everything including volumes
docker-compose down -v
```

## Database Migrations

After starting the databases, you can run migrations:

```bash
cd backend
alembic upgrade head
```

## Testing Your Backend

1. **Start the databases:**
   ```bash
   ./dev-setup.sh start
   ```

2. **Run your tests:**
   ```bash
   cd backend
   python test_workflow_simple.py
   ```

3. **Test the API:**
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```

4. **Visit the API docs:**
   - http://localhost:8000/docs

## Troubleshooting

### Docker Not Running
Make sure Docker Desktop is installed and running.

### Port Conflicts
If ports 5432, 6379, or 8000 are already in use, you can modify the `docker-compose.yml` file to use different ports.

### Database Connection Issues
1. Check if services are running: `docker-compose ps`
2. Check logs: `docker-compose logs postgres`
3. Wait a few seconds for services to fully start

### Permission Issues (Linux/Mac)
Make the setup script executable:
```bash
chmod +x dev-setup.sh
```

## Environment Variables

The Docker setup uses these default environment variables:

```bash
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=truthseeq
POSTGRES_PASSWORD=dev_password
POSTGRES_DB=truthseeq
REDIS_HOST=localhost
REDIS_PORT=6379
ENVIRONMENT=development
DEBUG=true
```

## Data Persistence

- PostgreSQL data is stored in a Docker volume (`postgres_data`)
- Redis data is stored in a Docker volume (`redis_data`)
- Data persists between container restarts
- Use `./dev-setup.sh cleanup` to remove all data

## Production vs Development

This setup is for **development only**. For production:

1. Use proper secrets management
2. Configure proper passwords
3. Set up SSL/TLS
4. Use production-grade PostgreSQL and Redis
5. Configure proper backup strategies
6. Set up monitoring and logging

## Next Steps

1. Start the databases: `./dev-setup.sh start`
2. Run your backend locally
3. Test your workflows
4. Set up API keys for AI services
5. Configure your environment variables 