# TruthSeeQ Backend Documentation

This directory contains comprehensive documentation for all major dependencies used in the TruthSeeQ backend application.

## 📚 Documentation Index

### Core Framework & Web Development
- [FastAPI Guide](./fastapi-guide.md) - High-performance async web framework
- [Uvicorn & Deployment](./uvicorn-deployment.md) - ASGI server configuration and deployment

### Database & Data Management
- [SQLAlchemy & Alembic](./database-guide.md) - ORM and database migrations
- [PostgreSQL Setup](./postgresql-setup.md) - Database configuration and best practices

### AI/ML & LangChain Ecosystem
- [LangGraph & LangChain](./langchain-guide.md) - AI workflow orchestration
- [OpenAI Integration](./openai-guide.md) - AI model integration and best practices

### Caching & Background Tasks
- [Redis Configuration](./redis-guide.md) - Caching and message broker setup
- [Celery Tasks](./celery-guide.md) - Background task processing

### Web Scraping & Content Processing
- [Web Scraping Stack](./scraping-guide.md) - BeautifulSoup, requests, and content extraction
- [Content Processing](./content-processing.md) - Text processing and analysis

### Testing & Development
- [Testing Framework](./testing-guide.md) - pytest, coverage, and testing best practices
- [Code Quality](./code-quality.md) - Linting, formatting, and type checking

### Security & Authentication
- [Security Implementation](./security-guide.md) - Authentication, authorization, and security best practices

### Monitoring & Logging
- [Observability Setup](./monitoring-guide.md) - Logging, metrics, and monitoring

## 🚀 Quick Start

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 Environment Setup

### Required Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/truthseeq
DATABASE_URL_SYNC=postgresql://user:password@localhost/truthseeq

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# AI Service Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Development
DEBUG=False
ENVIRONMENT=development
```

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration management
│   ├── api/
│   │   ├── routes/            # API endpoints
│   │   └── dependencies.py    # FastAPI dependencies
│   ├── core/
│   │   ├── exceptions.py      # Custom exceptions
│   │   ├── logging.py         # Logging configuration
│   │   └── rate_limiting.py   # Rate limiting logic
│   ├── database/
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── database.py        # Database connection
│   │   └── migrations/        # Alembic migrations
│   ├── services/
│   │   ├── ai_service.py      # LangGraph/LangChain integration
│   │   ├── scraper_service.py # Web scraping service
│   │   └── fact_checker.py    # Fact-checking logic
│   └── schemas/
│       └── *.py               # Pydantic models
├── langgraph/
│   ├── workflows/             # LangGraph workflows
│   ├── nodes/                 # Individual processing nodes
│   └── tools/                 # Custom tools and utilities
├── tests/
│   ├── test_api/             # API endpoint tests
│   ├── test_services/        # Service layer tests
│   └── test_langgraph/       # LangGraph workflow tests
├── docs/                     # This documentation
├── requirements.txt          # Python dependencies
└── alembic.ini              # Database migration configuration
```

## 🔧 Development Commands

### Database Operations
```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api/test_content.py
```

### Code Quality
```bash
# Format code
black app/ tests/

# Lint code
ruff check app/ tests/

# Type checking
mypy app/
```

### Development Server
```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start with specific workers (production)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Redis Documentation](https://redis.io/documentation)
- [Celery Documentation](https://docs.celeryproject.org/)

## 🤝 Contributing

When adding new dependencies:

1. Add them to `requirements.txt` with pinned versions
2. Create or update relevant documentation in this `docs/` folder
3. Include usage examples and best practices
4. Update this README.md index as needed

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 