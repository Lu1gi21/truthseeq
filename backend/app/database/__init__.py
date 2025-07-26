"""
Database module for TruthSeeQ backend.

This module provides database connectivity, models, and utilities for the TruthSeeQ
fact-checking platform.

Exports:
- Database connection and session management
- All database models and enums
- Database initialization utilities
"""

from .database import (
    Base,
    DatabaseManager,
    db_manager,
    init_database,
    get_db,
    create_tables,
    drop_tables,
)

from .models import (
    # Enums
    ContentStatus,
    FactCheckVerdict,
    SourceType,
    WorkflowStatus,
    
    # Rate Limiting Models
    UserSession,
    RateLimitLog,
    
    # Content Models
    ContentItem,
    ContentMetadata,
    
    # Fact-Checking Models
    FactCheckResult,
    FactCheckSource,
    
    # Social Feed Models
    FeedPost,
    FeedLike,
    
    # AI Analysis Models
    AIAnalysisResult,
    AIWorkflowExecution,
)

__all__ = [
    # Database Management
    "Base",
    "DatabaseManager",
    "db_manager", 
    "init_database",
    "get_db",
    "create_tables",
    "drop_tables",
    
    # Enums
    "ContentStatus",
    "FactCheckVerdict",
    "SourceType", 
    "WorkflowStatus",
    
    # Rate Limiting Models
    "UserSession",
    "RateLimitLog",
    
    # Content Models
    "ContentItem",
    "ContentMetadata",
    
    # Fact-Checking Models
    "FactCheckResult",
    "FactCheckSource",
    
    # Social Feed Models
    "FeedPost",
    "FeedLike",
    
    # AI Analysis Models
    "AIAnalysisResult",
    "AIWorkflowExecution",
]
