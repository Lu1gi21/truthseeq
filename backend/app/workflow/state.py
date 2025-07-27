"""
Workflow State Management

This module defines the state structures for TruthSeeQ workflows using TypedDict
for type safety and clear data contracts between workflow nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class VerdictType(str, Enum):
    """Enumeration of possible fact-checking verdicts."""
    TRUE = "true"
    MOSTLY_TRUE = "mostly_true"
    PARTIALLY_TRUE = "partially_true"
    UNVERIFIABLE = "unverifiable"
    MOSTLY_FALSE = "mostly_false"
    FALSE = "false"
    MISLEADING = "misleading"


class ConfidenceLevel(str, Enum):
    """Enumeration of confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class SourceType(str, Enum):
    """Enumeration of source types."""
    NEWS_ARTICLE = "news_article"
    ACADEMIC_PAPER = "academic_paper"
    GOVERNMENT_DOCUMENT = "government_document"
    OFFICIAL_STATEMENT = "official_statement"
    SOCIAL_MEDIA = "social_media"
    BLOG_POST = "blog_post"
    UNKNOWN = "unknown"


class BaseWorkflowState(TypedDict):
    """Base state for all workflows."""
    workflow_id: str
    session_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: str
    error_message: Optional[str]


class ScrapedContent(TypedDict):
    """Structure for scraped content data."""
    url: str
    title: Optional[str]
    content: str
    metadata: Dict[str, Any]
    scraped_at: datetime
    method_used: str
    success: bool
    error_message: Optional[str]


class Claim(TypedDict):
    """Structure for extracted claims."""
    id: str
    text: str
    confidence: float
    category: str
    entities: List[str]
    sentiment: str


class Source(TypedDict):
    """Structure for verification sources."""
    url: str
    title: Optional[str]
    content: str
    source_type: SourceType
    credibility_score: float
    relevance_score: float
    verification_status: str
    scraped_at: datetime


class FactCheckState(BaseWorkflowState):
    """
    State for fact-checking workflow.
    
    This state tracks the progress of fact-checking analysis including:
    - Original content and extracted claims
    - Source verification results
    - Fact analysis and verification
    - Final verdict and confidence scoring
    """
    # Input content
    original_url: str
    scraped_content: ScrapedContent
    
    # Extracted claims
    extracted_claims: List[Claim]
    claims_analysis: Dict[str, Any]
    
    # Source verification
    verification_sources: List[Source]
    source_analysis: Dict[str, Any]
    
    # Fact analysis
    fact_analysis: Dict[str, Any]
    cross_references: List[Dict[str, Any]]
    
    # Results
    confidence_score: float
    confidence_level: ConfidenceLevel
    verdict: VerdictType
    reasoning: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    
    # Metadata
    processing_time: float
    model_used: str
    search_queries: List[str]


class ContentAnalysisState(BaseWorkflowState):
    """
    State for content analysis workflow.
    
    This state tracks comprehensive content analysis including:
    - Sentiment and bias analysis
    - Content categorization and quality assessment
    - Source credibility evaluation
    - Content structure and readability analysis
    """
    # Input content
    original_url: str
    scraped_content: ScrapedContent
    
    # Sentiment analysis
    sentiment_analysis: Dict[str, Any]
    emotional_tone: str
    sentiment_score: float
    
    # Bias detection
    bias_analysis: Dict[str, Any]
    bias_level: str
    bias_types: List[str]
    political_leaning: Optional[str]
    
    # Content quality
    content_quality_score: float
    readability_score: float
    structure_analysis: Dict[str, Any]
    
    # Categorization
    content_category: str
    content_subcategory: str
    topics: List[str]
    entities: List[str]
    
    # Source credibility
    source_credibility_score: float
    domain_reputation: Dict[str, Any]
    
    # Summary
    analysis_summary: str
    key_insights: List[str]
    recommendations: List[str]


class SourceVerificationState(BaseWorkflowState):
    """
    State for source verification workflow.
    
    This state tracks source verification analysis including:
    - Domain analysis and reputation checking
    - Fact-checking database lookups
    - Cross-referencing with reliable sources
    - Historical accuracy assessment
    """
    # Input
    target_url: str
    domain: str
    
    # Domain analysis
    domain_analysis: Dict[str, Any]
    domain_age: Optional[str]
    ssl_valid: bool
    whois_info: Dict[str, Any]
    
    # Reputation analysis
    reputation_analysis: Dict[str, Any]
    reputation_score: float
    trust_indicators: List[str]
    red_flags: List[str]
    
    # Fact-checking databases
    fact_checking_results: Dict[str, Any]
    known_fact_checks: List[Dict[str, Any]]
    accuracy_history: Dict[str, Any]
    
    # Cross-referencing
    cross_reference_sources: List[Source]
    verification_consistency: float
    conflicting_reports: List[Dict[str, Any]]
    
    # Results
    verification_score: float
    verification_status: str
    confidence_level: ConfidenceLevel
    verification_summary: str
    recommendations: List[str]


class WorkflowState(TypedDict):
    """
    Union type for all workflow states.
    
    This allows for flexible state handling across different workflow types.
    """
    fact_check: Optional[FactCheckState]
    content_analysis: Optional[ContentAnalysisState]
    source_verification: Optional[SourceVerificationState]


# Utility functions for state management
def create_fact_check_state(
    workflow_id: str,
    original_url: str,
    session_id: Optional[str] = None
) -> FactCheckState:
    """
    Create initial fact-checking state.
    
    Args:
        workflow_id: Unique workflow identifier
        original_url: URL to fact-check
        session_id: Optional user session ID
        
    Returns:
        Initialized FactCheckState
    """
    now = datetime.utcnow()
    return FactCheckState(
        workflow_id=workflow_id,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        status="initialized",
        error_message=None,
        original_url=original_url,
        scraped_content=ScrapedContent(
            url="",
            title=None,
            content="",
            metadata={},
            scraped_at=now,
            method_used="",
            success=False,
            error_message=None
        ),
        extracted_claims=[],
        claims_analysis={},
        verification_sources=[],
        source_analysis={},
        fact_analysis={},
        cross_references=[],
        confidence_score=0.0,
        confidence_level=ConfidenceLevel.VERY_LOW,
        verdict=VerdictType.UNVERIFIABLE,
        reasoning="",
        supporting_evidence=[],
        contradicting_evidence=[],
        processing_time=0.0,
        model_used="",
        search_queries=[]
    )


def create_content_analysis_state(
    workflow_id: str,
    original_url: str,
    session_id: Optional[str] = None
) -> ContentAnalysisState:
    """
    Create initial content analysis state.
    
    Args:
        workflow_id: Unique workflow identifier
        original_url: URL to analyze
        session_id: Optional user session ID
        
    Returns:
        Initialized ContentAnalysisState
    """
    now = datetime.utcnow()
    return ContentAnalysisState(
        workflow_id=workflow_id,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        status="initialized",
        error_message=None,
        original_url=original_url,
        scraped_content=ScrapedContent(
            url="",
            title=None,
            content="",
            metadata={},
            scraped_at=now,
            method_used="",
            success=False,
            error_message=None
        ),
        sentiment_analysis={},
        emotional_tone="neutral",
        sentiment_score=0.0,
        bias_analysis={},
        bias_level="none",
        bias_types=[],
        political_leaning=None,
        content_quality_score=0.0,
        readability_score=0.0,
        structure_analysis={},
        content_category="unknown",
        content_subcategory="unknown",
        topics=[],
        entities=[],
        source_credibility_score=0.0,
        domain_reputation={},
        analysis_summary="",
        key_insights=[],
        recommendations=[]
    )


def create_source_verification_state(
    workflow_id: str,
    target_url: str,
    session_id: Optional[str] = None
) -> SourceVerificationState:
    """
    Create initial source verification state.
    
    Args:
        workflow_id: Unique workflow identifier
        target_url: URL to verify
        session_id: Optional user session ID
        
    Returns:
        Initialized SourceVerificationState
    """
    now = datetime.utcnow()
    from urllib.parse import urlparse
    
    domain = urlparse(target_url).netloc
    return SourceVerificationState(
        workflow_id=workflow_id,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        status="initialized",
        error_message=None,
        target_url=target_url,
        domain=domain,
        domain_analysis={},
        domain_age=None,
        ssl_valid=False,
        whois_info={},
        reputation_analysis={},
        reputation_score=0.0,
        trust_indicators=[],
        red_flags=[],
        fact_checking_results={},
        known_fact_checks=[],
        accuracy_history={},
        cross_reference_sources=[],
        verification_consistency=0.0,
        conflicting_reports=[],
        verification_score=0.0,
        verification_status="unknown",
        confidence_level=ConfidenceLevel.VERY_LOW,
        verification_summary="",
        recommendations=[]
    )


def update_state_timestamp(state: BaseWorkflowState) -> BaseWorkflowState:
    """
    Update the timestamp of a workflow state.
    
    Args:
        state: Workflow state to update
        
    Returns:
        Updated state with new timestamp
    """
    state["updated_at"] = datetime.utcnow()
    return state


def get_state_status(state: BaseWorkflowState) -> str:
    """
    Get the current status of a workflow state.
    
    Args:
        state: Workflow state
        
    Returns:
        Current status string
    """
    return state.get("status", "unknown")


def set_state_error(state: BaseWorkflowState, error_message: str) -> BaseWorkflowState:
    """
    Set error state for a workflow.
    
    Args:
        state: Workflow state to update
        error_message: Error message to set
        
    Returns:
        Updated state with error information
    """
    state["status"] = "error"
    state["error_message"] = error_message
    state["updated_at"] = datetime.utcnow()
    return state 