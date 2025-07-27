"""
Structured Output Models for TruthSeeQ Workflow Nodes

This module defines Pydantic models for structured output from AI models
to ensure consistent and valid JSON responses across all workflow nodes.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ========================================
# Sentiment Analysis Models
# ========================================

class SentimentAnalysisOutput(BaseModel):
    """Structured output for sentiment analysis."""
    
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the content"
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Sentiment score from -1 (very negative) to 1 (very positive)"
    )
    emotional_tone: str = Field(
        description="Primary emotional tone of the content"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the sentiment analysis"
    )
    key_emotions: List[str] = Field(
        default_factory=list,
        description="List of key emotions detected"
    )
    reasoning: str = Field(
        description="Explanation of the sentiment analysis"
    )


# ========================================
# Claims Extraction Models
# ========================================

class ExtractedClaim(BaseModel):
    """Structure for a single extracted claim."""
    
    id: str = Field(description="Unique identifier for the claim")
    text: str = Field(description="The factual claim text")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the claim extraction"
    )
    category: str = Field(description="Category of the claim (statistic, statement, etc.)")
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities mentioned in the claim"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the claim"
    )


class ClaimsExtractionOutput(BaseModel):
    """Structured output for claims extraction."""
    
    claims: List[ExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted factual claims"
    )
    total_claims: int = Field(
        ge=0,
        description="Total number of claims extracted"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the extraction"
    )


# ========================================
# Bias Detection Models
# ========================================

class BiasAnalysis(BaseModel):
    """Structure for bias analysis results."""
    
    bias_level: Literal["low", "medium", "high", "none"] = Field(
        description="Overall bias level detected"
    )
    bias_types: List[str] = Field(
        default_factory=list,
        description="Types of bias detected"
    )
    political_leaning: Optional[str] = Field(
        default=None,
        description="Political leaning if detected"
    )
    bias_score: float = Field(
        ge=0.0, le=1.0,
        description="Bias score from 0 (unbiased) to 1 (highly biased)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the bias analysis"
    )
    reasoning: str = Field(
        description="Explanation of bias detection"
    )


# ========================================
# Quality Assessment Models
# ========================================

class QualityMetrics(BaseModel):
    """Structure for content quality metrics."""
    
    content_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall content quality score"
    )
    readability_score: float = Field(
        ge=0.0, le=1.0,
        description="Readability score"
    )
    structure_quality: float = Field(
        ge=0.0, le=1.0,
        description="Structure and organization quality"
    )
    factual_accuracy: float = Field(
        ge=0.0, le=1.0,
        description="Perceived factual accuracy"
    )
    source_credibility: float = Field(
        ge=0.0, le=1.0,
        description="Source credibility assessment"
    )


class QualityAssessmentOutput(BaseModel):
    """Structured output for quality assessment."""
    
    metrics: QualityMetrics = Field(description="Quality metrics")
    content_category: str = Field(description="Category of content")
    content_subcategory: str = Field(description="Subcategory of content")
    topics: List[str] = Field(
        default_factory=list,
        description="Main topics covered"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities mentioned"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the assessment"
    )


# ========================================
# Fact Analysis Models
# ========================================

class FactAnalysisResult(BaseModel):
    """Structure for fact analysis results."""
    
    verdict: Literal["true", "false", "inconclusive", "satire"] = Field(
        description="Fact-checking verdict"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the verdict"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the verdict"
    )
    key_evidence: List[str] = Field(
        default_factory=list,
        description="Key evidence supporting the verdict"
    )
    sources_checked: List[str] = Field(
        default_factory=list,
        description="Sources consulted for verification"
    )
    ai_model_used: str = Field(
        description="AI model used for analysis"
    )


# ========================================
# Confidence Scoring Models
# ========================================

class ConfidenceAnalysis(BaseModel):
    """Structure for confidence analysis."""
    
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence score"
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        description="Confidence level category"
    )
    factors: List[str] = Field(
        default_factory=list,
        description="Factors affecting confidence"
    )
    explanation: str = Field(
        description="Explanation of confidence calculation"
    )


# ========================================
# Summary Generation Models
# ========================================

class SummaryOutput(BaseModel):
    """Structured output for summary generation."""
    
    summary: str = Field(description="Generated summary")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points from the content"
    )
    word_count: int = Field(
        ge=0,
        description="Word count of the summary"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the summary quality"
    )


# ========================================
# Source Verification Models
# ========================================

class SourceVerificationResult(BaseModel):
    """Structure for source verification results."""
    
    url: str = Field(description="Source URL")
    credibility_score: float = Field(
        ge=0.0, le=1.0,
        description="Credibility score of the source"
    )
    source_type: str = Field(description="Type of source")
    domain_reputation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain reputation information"
    )
    verification_status: Literal["verified", "unverified", "failed"] = Field(
        description="Verification status"
    )
    content_quality: float = Field(
        ge=0.0, le=1.0,
        description="Quality of the source content"
    )


class SourceVerificationOutput(BaseModel):
    """Structured output for source verification."""
    
    sources: List[SourceVerificationResult] = Field(
        default_factory=list,
        description="List of verified sources"
    )
    average_credibility: float = Field(
        ge=0.0, le=1.0,
        description="Average credibility score"
    )
    total_sources: int = Field(
        ge=0,
        description="Total number of sources checked"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the verification"
    ) 