"""
Content-related Pydantic schemas for TruthSeeQ platform.

This module contains Pydantic models for content validation, API requests/responses,
and data transfer objects related to content scraping and analysis.

Schemas include:
- Content creation and retrieval models
- Scraping request and result models  
- Search and discovery models
- Content quality and validation models
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, validator
from pydantic.types import constr, confloat, conint

from ..database.models import ContentStatus, FactCheckVerdict


# ========================================
# Base Content Models
# ========================================

class ContentMetadataCreate(BaseModel):
    """Schema for creating content metadata."""
    
    metadata_type: constr(min_length=1, max_length=100) = Field(
        ..., 
        description="Type of metadata (author, publish_date, tags, etc.)",
        example="author"
    )
    metadata_value: str = Field(
        ..., 
        description="Value of the metadata field",
        example="John Doe"
    )


class ContentMetadataResponse(ContentMetadataCreate):
    """Schema for content metadata response."""
    
    id: int = Field(..., description="Unique metadata ID")
    content_id: int = Field(..., description="Associated content ID")
    created_at: datetime = Field(..., description="Metadata creation timestamp")
    
    class Config:
        from_attributes = True


class ContentItemCreate(BaseModel):
    """Schema for creating a content item."""
    
    url: HttpUrl = Field(
        ..., 
        description="URL of the content to scrape",
        example="https://example.com/article"
    )
    title: Optional[str] = Field(
        None, 
        max_length=1000,
        description="Optional title override",
        example="Breaking News: Important Update"
    )
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format and scheme."""
        if not str(v).startswith(('http://', 'https://')):
            raise ValueError('URL must use http or https scheme')
        return v


class ContentItemResponse(BaseModel):
    """Schema for content item response."""
    
    id: int = Field(..., description="Unique content ID")
    url: str = Field(..., description="Original URL")
    title: Optional[str] = Field(None, description="Content title")
    content: str = Field(..., description="Extracted content")
    source_domain: str = Field(..., description="Source domain")
    status: ContentStatus = Field(..., description="Processing status")
    scraped_at: datetime = Field(..., description="Scraping timestamp")
    metadata: List[ContentMetadataResponse] = Field(
        default_factory=list, 
        description="Associated metadata"
    )
    
    class Config:
        from_attributes = True


class ContentItemUpdate(BaseModel):
    """Schema for updating content items."""
    
    title: Optional[str] = Field(None, max_length=1000)
    content: Optional[str] = Field(None)
    status: Optional[ContentStatus] = Field(None)


# ========================================
# Scraping Request/Response Models
# ========================================

class ScrapingRequest(BaseModel):
    """Schema for scraping requests."""
    
    urls: List[HttpUrl] = Field(
        ..., 
        min_items=1,
        max_items=100,
        description="URLs to scrape (max 100)",
        example=["https://example.com/article1", "https://example.com/article2"]
    )
    priority: Optional[conint(ge=1, le=10)] = Field(
        5, 
        description="Scraping priority (1=lowest, 10=highest)"
    )
    force_rescrape: bool = Field(
        False, 
        description="Force re-scraping of existing content"
    )
    include_metadata: bool = Field(
        True, 
        description="Extract and store metadata"
    )
    
    @validator('urls')
    def validate_urls(cls, v):
        """Validate URLs in batch request."""
        unique_urls = set(str(url) for url in v)
        if len(unique_urls) != len(v):
            raise ValueError('Duplicate URLs are not allowed')
        return v


class ScrapingResult(BaseModel):
    """Schema for individual scraping result."""
    
    url: str = Field(..., description="Scraped URL")
    success: bool = Field(..., description="Whether scraping succeeded")
    content: Optional[str] = Field(None, description="Extracted content")
    title: Optional[str] = Field(None, description="Extracted title")
    metadata: Optional[Dict[str, str]] = Field(
        None, 
        description="Extracted metadata"
    )
    error_message: Optional[str] = Field(
        None, 
        description="Error message if scraping failed"
    )
    method_used: Optional[str] = Field(
        None, 
        description="Scraping method used"
    )
    response_time: Optional[confloat(ge=0)] = Field(
        None, 
        description="Response time in seconds"
    )
    quality_score: Optional[confloat(ge=0, le=1)] = Field(
        None, 
        description="Content quality score (0-1)"
    )


class ScrapingResponse(BaseModel):
    """Schema for scraping batch response."""
    
    job_id: UUID = Field(..., description="Unique job identifier")
    total_urls: int = Field(..., description="Total URLs requested")
    successful: int = Field(..., description="Successfully scraped URLs")
    failed: int = Field(..., description="Failed URLs")
    results: List[ScrapingResult] = Field(..., description="Scraping results")
    processing_time: confloat(ge=0) = Field(
        ..., 
        description="Total processing time in seconds"
    )


# ========================================
# Search and Discovery Models
# ========================================

class SearchRequest(BaseModel):
    """Schema for content search requests."""
    
    query: constr(min_length=1, max_length=1000) = Field(
        ..., 
        description="Search query",
        example="climate change latest research"
    )
    max_results: conint(ge=1, le=100) = Field(
        10, 
        description="Maximum number of results"
    )
    search_type: Optional[str] = Field(
        "web", 
        description="Type of search (web, news, images, videos)"
    )
    country: Optional[str] = Field(
        None, 
        description="Country code for localized results"
    )
    language: Optional[str] = Field(
        None, 
        description="Language code for results"
    )
    freshness: Optional[str] = Field(
        None, 
        description="Content freshness filter (pd=past day, pw=past week, pm=past month, py=past year)"
    )
    safe_search: Optional[str] = Field(
        "moderate", 
        description="Safe search setting (strict, moderate, off)"
    )


class SearchResult(BaseModel):
    """Schema for individual search result."""
    
    title: str = Field(..., description="Result title")
    url: HttpUrl = Field(..., description="Result URL")
    description: Optional[str] = Field(None, description="Result description/snippet")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    source_domain: str = Field(..., description="Source domain")
    relevance_score: Optional[confloat(ge=0, le=1)] = Field(
        None, 
        description="Relevance score (0-1)"
    )


class SearchResponse(BaseModel):
    """Schema for search response."""
    
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total results found")
    results: List[SearchResult] = Field(..., description="Search results")
    search_time: confloat(ge=0) = Field(..., description="Search time in seconds")
    suggestions: List[str] = Field(
        default_factory=list, 
        description="Query suggestions"
    )


# ========================================
# Content Quality and Validation Models
# ========================================

class ContentQualityMetrics(BaseModel):
    """Schema for content quality assessment."""
    
    length_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Content length adequacy score"
    )
    readability_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Content readability score"
    )
    structure_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Content structure quality score"
    )
    metadata_completeness: confloat(ge=0, le=1) = Field(
        ..., 
        description="Metadata completeness score"
    )
    source_credibility: Optional[confloat(ge=0, le=1)] = Field(
        None, 
        description="Source credibility score"
    )
    overall_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Overall quality score"
    )


class ContentValidationRequest(BaseModel):
    """Schema for content validation requests."""
    
    content_id: int = Field(..., description="Content ID to validate")
    validation_type: str = Field(
        ..., 
        description="Type of validation (quality, authenticity, relevance)"
    )
    parameters: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        None, 
        description="Validation parameters"
    )


class ContentValidationResponse(BaseModel):
    """Schema for content validation results."""
    
    content_id: int = Field(..., description="Validated content ID")
    validation_type: str = Field(..., description="Type of validation performed")
    is_valid: bool = Field(..., description="Whether content passed validation")
    confidence: confloat(ge=0, le=1) = Field(
        ..., 
        description="Confidence in validation result"
    )
    quality_metrics: Optional[ContentQualityMetrics] = Field(
        None, 
        description="Detailed quality metrics"
    )
    issues: List[str] = Field(
        default_factory=list, 
        description="Identified issues"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Improvement recommendations"
    )


# ========================================
# Deduplication and Analysis Models
# ========================================

class ContentSimilarity(BaseModel):
    """Schema for content similarity comparison."""
    
    content_id_1: int = Field(..., description="First content ID")
    content_id_2: int = Field(..., description="Second content ID")
    similarity_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Similarity score (0=different, 1=identical)"
    )
    similarity_type: str = Field(
        ..., 
        description="Type of similarity (text, semantic, structural)"
    )
    details: Optional[Dict[str, Union[str, float]]] = Field(
        None, 
        description="Detailed similarity metrics"
    )


class DeduplicationRequest(BaseModel):
    """Schema for content deduplication requests."""
    
    content_ids: List[int] = Field(
        ..., 
        min_items=2,
        description="Content IDs to check for duplicates"
    )
    similarity_threshold: confloat(ge=0, le=1) = Field(
        0.8, 
        description="Similarity threshold for considering content duplicate"
    )
    deduplication_strategy: str = Field(
        "keep_first", 
        description="Strategy for handling duplicates (keep_first, keep_best, merge)"
    )


class DeduplicationResponse(BaseModel):
    """Schema for deduplication results."""
    
    total_content: int = Field(..., description="Total content items processed")
    duplicates_found: int = Field(..., description="Number of duplicates found")
    duplicates_removed: int = Field(..., description="Number of duplicates removed")
    duplicate_groups: List[List[int]] = Field(
        ..., 
        description="Groups of duplicate content IDs"
    )
    similarities: List[ContentSimilarity] = Field(
        ..., 
        description="Detailed similarity comparisons"
    )


# ========================================
# AI Analysis and Fact-Checking Models
# ========================================

class FactCheckRequest(BaseModel):
    """Schema for fact-checking requests."""
    
    content_id: int = Field(..., description="Content ID to fact-check")
    model_name: Optional[str] = Field(
        None, 
        description="AI model to use for analysis"
    )
    force_refresh: bool = Field(
        False, 
        description="Force refresh of cached results"
    )
    analysis_depth: str = Field(
        "standard", 
        description="Analysis depth (quick, standard, comprehensive)"
    )


class FactCheckResponse(BaseModel):
    """Schema for fact-checking responses."""
    
    content_id: int = Field(..., description="Content ID that was analyzed")
    confidence_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Confidence score in the fact-check result"
    )
    verdict: str = Field(..., description="Fact-checking verdict")
    reasoning: str = Field(..., description="Detailed reasoning for the verdict")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Sources used in fact-checking"
    )
    execution_time: confloat(ge=0) = Field(
        ..., 
        description="Analysis execution time in seconds"
    )
    workflow_execution_id: Optional[int] = Field(
        None, 
        description="ID of the workflow execution"
    )


class FactCheckResult(BaseModel):
    """Schema for individual fact-check result."""
    
    id: int = Field(..., description="Fact-check result ID")
    content_id: int = Field(..., description="Content ID that was analyzed")
    confidence_score: confloat(ge=0, le=1) = Field(
        ..., 
        description="Confidence score in the fact-check result"
    )
    verdict: str = Field(..., description="Fact-checking verdict")
    reasoning: str = Field(..., description="Detailed reasoning for the verdict")
    created_at: datetime = Field(..., description="When the fact-check was performed")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Sources used in fact-checking"
    )


class BatchVerificationRequest(BaseModel):
    """Schema for batch verification requests."""
    
    content_ids: List[int] = Field(
        ..., 
        min_items=1,
        max_items=50,
        description="Content IDs to verify (max 50)"
    )
    model_name: Optional[str] = Field(
        None, 
        description="AI model to use for analysis"
    )
    force_refresh: bool = Field(
        False, 
        description="Force refresh of cached results"
    )
    priority: str = Field(
        "normal", 
        description="Verification priority (low, normal, high, urgent)"
    )


class BatchVerificationResponse(BaseModel):
    """Schema for batch verification responses."""
    
    results: List[FactCheckResponse] = Field(
        ..., 
        description="Individual verification results"
    )
    failed_items: List[int] = Field(
        default_factory=list, 
        description="Content IDs that failed verification"
    )
    summary: Dict[str, Any] = Field(
        ..., 
        description="Summary statistics of verification results"
    )
    total_processed: int = Field(..., description="Total items processed")
    successful_count: int = Field(..., description="Number of successful verifications")


class VerificationHistoryRequest(BaseModel):
    """Schema for verification history requests."""
    
    content_id: Optional[int] = Field(
        None, 
        description="Optional content ID filter"
    )
    start_date: Optional[datetime] = Field(
        None, 
        description="Start date for history filter"
    )
    end_date: Optional[datetime] = Field(
        None, 
        description="End date for history filter"
    )
    limit: Optional[int] = Field(
        50, 
        description="Maximum number of results to return"
    )
    offset: Optional[int] = Field(
        0, 
        description="Number of results to skip"
    )


class VerificationHistoryResponse(BaseModel):
    """Schema for verification history responses."""
    
    history: List[FactCheckResult] = Field(
        ..., 
        description="Verification history items"
    )
    summary: Dict[str, Any] = Field(
        ..., 
        description="Summary statistics of history"
    )
    total_count: int = Field(..., description="Total number of history items")


class ContentAnalysisRequest(BaseModel):
    """Schema for content analysis requests."""
    
    content_id: int = Field(..., description="Content ID to analyze")
    analysis_types: List[str] = Field(
        default_factory=lambda: ["sentiment", "bias", "credibility"],
        description="Types of analysis to perform"
    )
    model_name: Optional[str] = Field(
        None, 
        description="AI model to use for analysis"
    )
    force_refresh: bool = Field(
        False, 
        description="Force refresh of cached results"
    )


class ContentAnalysisResponse(BaseModel):
    """Schema for content analysis responses."""
    
    content_id: int = Field(..., description="Content ID that was analyzed")
    analysis_results: Dict[str, Any] = Field(
        ..., 
        description="Analysis results for each requested type"
    )
    confidence: confloat(ge=0, le=1) = Field(
        ..., 
        description="Overall confidence in analysis results"
    )
    analysis_id: Optional[int] = Field(
        None, 
        description="ID of the stored analysis result"
    )


class AIAnalysisRequest(BaseModel):
    """Schema for general AI analysis requests."""
    
    content_id: int = Field(..., description="Content ID to analyze")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Analysis-specific parameters"
    )
    model_name: Optional[str] = Field(
        None, 
        description="AI model to use"
    )


class AIAnalysisResponse(BaseModel):
    """Schema for general AI analysis responses."""
    
    content_id: int = Field(..., description="Content ID that was analyzed")
    analysis_type: str = Field(..., description="Type of analysis performed")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    confidence: confloat(ge=0, le=1) = Field(..., description="Confidence score")
    execution_time: confloat(ge=0) = Field(..., description="Execution time")


class WorkflowExecutionRequest(BaseModel):
    """Schema for workflow execution requests."""
    
    workflow_type: str = Field(..., description="Type of workflow to execute")
    content_id: int = Field(..., description="Content ID for workflow")
    parameters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Workflow-specific parameters"
    )


class WorkflowExecutionResponse(BaseModel):
    """Schema for workflow execution responses."""
    
    execution_id: int = Field(..., description="Workflow execution ID")
    workflow_type: str = Field(..., description="Type of workflow executed")
    status: str = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result")
    execution_time: Optional[confloat(ge=0)] = Field(None, description="Execution time")


# ========================================
# Job Status and Monitoring Models
# ========================================

class JobStatus(BaseModel):
    """Schema for background job status."""
    
    job_id: UUID = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of job (scraping, validation, deduplication)")
    status: str = Field(
        ..., 
        description="Current job status (pending, running, completed, failed)"
    )
    progress: confloat(ge=0, le=1) = Field(
        ..., 
        description="Job progress (0=started, 1=completed)"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    result_summary: Optional[Dict[str, Union[str, int, float]]] = Field(
        None, 
        description="Summary of job results"
    )


# ========================================
# Export schemas
# ========================================

__all__ = [
    # Content models
    "ContentMetadataCreate",
    "ContentMetadataResponse", 
    "ContentItemCreate",
    "ContentItemResponse",
    "ContentItemUpdate",
    
    # Scraping models
    "ScrapingRequest",
    "ScrapingResult",
    "ScrapingResponse",
    
    # Search models
    "SearchRequest",
    "SearchResult", 
    "SearchResponse",
    
    # Quality models
    "ContentQualityMetrics",
    "ContentValidationRequest",
    "ContentValidationResponse",
    
    # Deduplication models
    "ContentSimilarity",
    "DeduplicationRequest",
    "DeduplicationResponse",
    
    # AI Analysis models
    "FactCheckRequest",
    "FactCheckResponse",
    "FactCheckResult",
    "BatchVerificationRequest",
    "BatchVerificationResponse",
    "VerificationHistoryRequest",
    "VerificationHistoryResponse",
    "ContentAnalysisRequest",
    "ContentAnalysisResponse",
    "AIAnalysisRequest",
    "AIAnalysisResponse",
    "WorkflowExecutionRequest",
    "WorkflowExecutionResponse",
    
    # Job monitoring
    "JobStatus",
]
