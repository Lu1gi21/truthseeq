"""
Database models for TruthSeeQ platform.

This module contains all SQLAlchemy models for the TruthSeeQ fact-checking platform,
implementing the database schema defined in PLAN.MD step 1.3.

Models include:
- Rate limiting and user tracking
- Content management 
- Fact-checking results
- Social feed functionality
- AI analysis storage
"""

import enum
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


# ========================================
# Enums for controlled vocabularies
# ========================================

class ContentStatus(str, enum.Enum):
    """Content processing status enumeration."""
    PENDING = "pending"           # Newly scraped, awaiting processing
    PROCESSING = "processing"     # Currently being analyzed
    COMPLETED = "completed"       # Analysis completed successfully  
    FAILED = "failed"            # Processing failed with errors
    ARCHIVED = "archived"        # Content archived/no longer active


class FactCheckVerdict(str, enum.Enum):
    """Fact-checking verdict enumeration."""
    TRUE = "true"                # Content verified as accurate
    MOSTLY_TRUE = "mostly_true"  # Content mostly accurate with minor issues
    MIXED = "mixed"              # Content has both true and false elements
    MOSTLY_FALSE = "mostly_false" # Content mostly inaccurate
    FALSE = "false"              # Content verified as false/misleading
    UNVERIFIABLE = "unverifiable" # Cannot be verified with available sources
    OPINION = "opinion"          # Content is opinion-based, not factual


class SourceType(str, enum.Enum):
    """Source type enumeration for fact-checking sources."""
    NEWS_ARTICLE = "news_article"     # Traditional news source
    ACADEMIC_PAPER = "academic_paper" # Peer-reviewed academic source
    GOVERNMENT = "government"         # Official government source
    FACT_CHECKER = "fact_checker"     # Professional fact-checking organization
    EXPERT_OPINION = "expert_opinion" # Expert or professional opinion
    SOCIAL_MEDIA = "social_media"     # Social media post or content
    BLOG = "blog"                    # Blog post or opinion piece
    OTHER = "other"                  # Other source types


class WorkflowStatus(str, enum.Enum):
    """AI workflow execution status."""
    PENDING = "pending"       # Workflow queued for execution
    RUNNING = "running"       # Workflow currently executing
    COMPLETED = "completed"   # Workflow completed successfully
    FAILED = "failed"        # Workflow failed with errors
    CANCELLED = "cancelled"   # Workflow cancelled by user/system


# ========================================
# Rate Limiting and User Tracking Models
# ========================================

class UserSession(Base):
    """
    User session tracking for rate limiting and analytics.
    
    Tracks anonymous user sessions based on IP address, user agent,
    and session identifiers for rate limiting and usage analytics.
    """
    __tablename__ = "user_sessions"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Session identification
    session_id: Mapped[UUID] = mapped_column(
        Uuid, 
        default=uuid4, 
        unique=True, 
        index=True,
        doc="Unique session identifier for tracking user activity"
    )
    
    # Client information for identification and rate limiting
    ip_address: Mapped[str] = mapped_column(
        String(45),  # IPv6 addresses can be up to 45 characters
        index=True,
        doc="Client IP address for rate limiting and analytics"
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Client user agent string for device identification"
    )
    
    # Activity tracking
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Session creation timestamp"
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        doc="Last activity timestamp for session cleanup"
    )
    request_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Total number of requests made in this session"
    )
    
    # Relationships
    rate_limit_logs: Mapped[List["RateLimitLog"]] = relationship(
        "RateLimitLog",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    fact_check_results: Mapped[List["FactCheckResult"]] = relationship(
        "FactCheckResult",
        back_populates="session"
    )
    feed_posts: Mapped[List["FeedPost"]] = relationship(
        "FeedPost",
        back_populates="session"
    )
    feed_likes: Mapped[List["FeedLike"]] = relationship(
        "FeedLike",
        back_populates="session"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_user_sessions_ip_created", "ip_address", "created_at"),
        Index("ix_user_sessions_last_activity", "last_activity"),
    )


class RateLimitLog(Base):
    """
    Rate limiting log entries for monitoring and enforcement.
    
    Tracks API requests per session/endpoint for implementing
    sliding window rate limiting and usage analytics.
    """
    __tablename__ = "rate_limit_logs"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Session reference
    session_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("user_sessions.session_id", ondelete="CASCADE"),
        index=True,
        doc="Reference to user session"
    )
    
    # Request details
    endpoint: Mapped[str] = mapped_column(
        String(255),
        index=True,
        doc="API endpoint that was accessed"
    )
    request_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        doc="Number of requests in this time window"
    )
    
    # Time window for rate limiting
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        doc="Start of the rate limiting time window"
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        doc="End of the rate limiting time window"
    )
    
    # Relationships
    session: Mapped["UserSession"] = relationship(
        "UserSession",
        back_populates="rate_limit_logs"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_rate_limit_session_endpoint", "session_id", "endpoint"),
        Index("ix_rate_limit_window", "window_start", "window_end"),
        Index("ix_rate_limit_endpoint_window", "endpoint", "window_start"),
    )


# ========================================
# Content Management Models
# ========================================

class ContentItem(Base):
    """
    Scraped content items for fact-checking analysis.
    
    Stores web content that has been scraped for fact-checking,
    including source information and processing status.
    """
    __tablename__ = "content_items"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Content identification
    url: Mapped[str] = mapped_column(
        Text,
        unique=True,
        index=True,
        doc="Original URL of the scraped content"
    )
    title: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Title of the content (article headline, page title, etc.)"
    )
    
    # Content data
    content: Mapped[str] = mapped_column(
        Text,
        doc="Main textual content extracted from the page"
    )
    source_domain: Mapped[str] = mapped_column(
        String(255),
        index=True,
        doc="Domain name of the content source"
    )
    
    # Processing information
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Timestamp when content was scraped"
    )
    status: Mapped[ContentStatus] = mapped_column(
        default=ContentStatus.PENDING,
        index=True,
        doc="Current processing status of the content"
    )
    
    # Relationships
    metadata: Mapped[List["ContentMetadata"]] = relationship(
        "ContentMetadata",
        back_populates="content_item",
        cascade="all, delete-orphan"
    )
    fact_check_results: Mapped[List["FactCheckResult"]] = relationship(
        "FactCheckResult",
        back_populates="content_item"
    )
    feed_posts: Mapped[List["FeedPost"]] = relationship(
        "FeedPost",
        back_populates="content_item"
    )
    ai_analysis_results: Mapped[List["AIAnalysisResult"]] = relationship(
        "AIAnalysisResult",
        back_populates="content_item"
    )
    ai_workflow_executions: Mapped[List["AIWorkflowExecution"]] = relationship(
        "AIWorkflowExecution",
        back_populates="content_item"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_content_domain_status", "source_domain", "status"),
        Index("ix_content_scraped_status", "scraped_at", "status"),
    )


class ContentMetadata(Base):
    """
    Additional metadata for content items.
    
    Flexible key-value storage for content-specific metadata
    like author, publish date, tags, etc.
    """
    __tablename__ = "content_metadata"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Content reference
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_items.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the content item"
    )
    
    # Metadata key-value pairs
    metadata_type: Mapped[str] = mapped_column(
        String(100),
        index=True,
        doc="Type/category of metadata (author, publish_date, tags, etc.)"
    )
    metadata_value: Mapped[str] = mapped_column(
        Text,
        doc="Value of the metadata field"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Timestamp when metadata was created"
    )
    
    # Relationships
    content_item: Mapped["ContentItem"] = relationship(
        "ContentItem",
        back_populates="metadata"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_content_metadata_type", "content_id", "metadata_type"),
        UniqueConstraint("content_id", "metadata_type", name="uq_content_metadata_type"),
    )


# ========================================
# Fact-Checking Results Models
# ========================================

class FactCheckResult(Base):
    """
    Results of fact-checking analysis for content items.
    
    Stores the AI-generated fact-checking results including
    confidence scores, verdicts, and reasoning.
    """
    __tablename__ = "fact_check_results"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # References
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_items.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the analyzed content"
    )
    session_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("user_sessions.session_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Session that requested this fact-check"
    )
    
    # Analysis results
    confidence_score: Mapped[float] = mapped_column(
        Float,
        doc="Confidence score (0.0 to 1.0) in the fact-check result"
    )
    verdict: Mapped[FactCheckVerdict] = mapped_column(
        index=True,
        doc="Fact-checking verdict for the content"
    )
    reasoning: Mapped[str] = mapped_column(
        Text,
        doc="Detailed explanation of the fact-checking analysis"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Timestamp when fact-check was completed"
    )
    
    # Relationships
    content_item: Mapped["ContentItem"] = relationship(
        "ContentItem",
        back_populates="fact_check_results"
    )
    session: Mapped[Optional["UserSession"]] = relationship(
        "UserSession",
        back_populates="fact_check_results"
    )
    sources: Mapped[List["FactCheckSource"]] = relationship(
        "FactCheckSource",
        back_populates="fact_check_result",
        cascade="all, delete-orphan"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_fact_check_verdict_confidence", "verdict", "confidence_score"),
        Index("ix_fact_check_created", "created_at"),
    )


class FactCheckSource(Base):
    """
    Sources used in fact-checking analysis.
    
    Tracks the external sources consulted during fact-checking
    to provide transparency and verification paths.
    """
    __tablename__ = "fact_check_sources"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Fact-check reference
    fact_check_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("fact_check_results.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the fact-check result"
    )
    
    # Source information
    source_url: Mapped[str] = mapped_column(
        Text,
        doc="URL of the source used for verification"
    )
    source_type: Mapped[SourceType] = mapped_column(
        index=True,
        doc="Type/category of the source"
    )
    relevance_score: Mapped[float] = mapped_column(
        Float,
        doc="Relevance score (0.0 to 1.0) of this source to the fact-check"
    )
    
    # Relationships
    fact_check_result: Mapped["FactCheckResult"] = relationship(
        "FactCheckResult",
        back_populates="sources"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_fact_check_source_type", "fact_check_id", "source_type"),
        Index("ix_fact_check_source_relevance", "relevance_score"),
    )


# ========================================
# Social Feed Models
# ========================================

class FeedPost(Base):
    """
    Social feed posts sharing fact-checked content.
    
    Represents posts in the social feed where users can share
    and discover fact-checked content.
    """
    __tablename__ = "feed_posts"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # References
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_items.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the fact-checked content"
    )
    session_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("user_sessions.session_id", ondelete="CASCADE"),
        index=True,
        doc="Session that created this post"
    )
    
    # Post content
    title: Mapped[str] = mapped_column(
        String(500),
        doc="Title/headline for the social feed post"
    )
    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Brief summary of the content for the feed"
    )
    verdict: Mapped[FactCheckVerdict] = mapped_column(
        index=True,
        doc="Fact-checking verdict displayed in the feed"
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        doc="Confidence score displayed in the feed"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        doc="Timestamp when post was created"
    )
    
    # Relationships
    content_item: Mapped["ContentItem"] = relationship(
        "ContentItem",
        back_populates="feed_posts"
    )
    session: Mapped["UserSession"] = relationship(
        "UserSession",
        back_populates="feed_posts"
    )
    likes: Mapped[List["FeedLike"]] = relationship(
        "FeedLike",
        back_populates="post",
        cascade="all, delete-orphan"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_feed_post_verdict_created", "verdict", "created_at"),
        Index("ix_feed_post_confidence", "confidence"),
    )


class FeedLike(Base):
    """
    User likes/reactions to feed posts.
    
    Tracks user engagement with social feed posts for
    personalization and content ranking.
    """
    __tablename__ = "feed_likes"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # References
    post_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("feed_posts.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the liked post"
    )
    session_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("user_sessions.session_id", ondelete="CASCADE"),
        index=True,
        doc="Session that liked the post"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Timestamp when like was created"
    )
    
    # Relationships
    post: Mapped["FeedPost"] = relationship(
        "FeedPost",
        back_populates="likes"
    )
    session: Mapped["UserSession"] = relationship(
        "UserSession",
        back_populates="feed_likes"
    )
    
    # Database optimizations
    __table_args__ = (
        UniqueConstraint("post_id", "session_id", name="uq_feed_like_post_session"),
        Index("ix_feed_like_created", "created_at"),
    )


# ========================================
# AI Analysis Models
# ========================================

class AIAnalysisResult(Base):
    """
    AI analysis results for content items.
    
    Stores detailed AI analysis results including sentiment analysis,
    bias detection, and other content analysis metrics.
    """
    __tablename__ = "ai_analysis_results"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Content reference
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_items.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the analyzed content"
    )
    
    # Analysis details
    analysis_type: Mapped[str] = mapped_column(
        String(100),
        index=True,
        doc="Type of analysis performed (sentiment, bias, credibility, etc.)"
    )
    result_data: Mapped[dict] = mapped_column(
        JSONB,
        doc="Structured analysis results as JSON"
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        doc="Confidence score (0.0 to 1.0) in the analysis result"
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        doc="Timestamp when analysis was completed"
    )
    
    # Relationships
    content_item: Mapped["ContentItem"] = relationship(
        "ContentItem",
        back_populates="ai_analysis_results"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_ai_analysis_type_confidence", "analysis_type", "confidence"),
        Index("ix_ai_analysis_content_type", "content_id", "analysis_type"),
    )


class AIWorkflowExecution(Base):
    """
    AI workflow execution tracking and monitoring.
    
    Tracks the execution of LangGraph workflows for monitoring
    performance, debugging, and analytics.
    """
    __tablename__ = "ai_workflow_executions"
    
    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Content reference
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content_items.id", ondelete="CASCADE"),
        index=True,
        doc="Reference to the content being processed"
    )
    
    # Workflow details
    workflow_type: Mapped[str] = mapped_column(
        String(100),
        index=True,
        doc="Type of workflow executed (fact_check, content_analysis, etc.)"
    )
    status: Mapped[WorkflowStatus] = mapped_column(
        default=WorkflowStatus.PENDING,
        index=True,
        doc="Current status of the workflow execution"
    )
    execution_time: Mapped[Optional[float]] = mapped_column(
        Float,
        doc="Execution time in seconds (null if not completed)"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        doc="Timestamp when workflow was started"
    )
    
    # Relationships
    content_item: Mapped["ContentItem"] = relationship(
        "ContentItem",
        back_populates="ai_workflow_executions"
    )
    
    # Database optimizations
    __table_args__ = (
        Index("ix_workflow_type_status", "workflow_type", "status"),
        Index("ix_workflow_execution_time", "execution_time"),
        Index("ix_workflow_created_status", "created_at", "status"),
    )


# ========================================
# Export all models
# ========================================

__all__ = [
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
