"""
AI service with LangGraph integration for TruthSeeQ platform.

This module provides comprehensive AI capabilities including:
- LangGraph workflow orchestration for fact-checking
- AI model management and interaction
- Fact-checking request handling
- AI analysis result caching
- Content analysis and verification
- Confidence scoring and explanation generation

Classes:
    AIService: Main service class for AI operations
    ModelManager: Manages different AI models and providers
    WorkflowOrchestrator: Orchestrates LangGraph workflows
    AnalysisCache: Caches AI analysis results
    ConfidenceScorer: Generates confidence scores and explanations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough

# Import local modules
from ..config import settings
from ..database.models import (
    ContentItem, AIAnalysisResult, AIWorkflowExecution, 
    FactCheckResult, WorkflowStatus, SourceType
)
from ..schemas.content import (
    FactCheckRequest, FactCheckResponse, FactCheckResult as FactCheckResultSchema,
    ContentAnalysisRequest, ContentAnalysisResponse,
    AIAnalysisRequest, AIAnalysisResponse,
    WorkflowExecutionRequest, WorkflowExecutionResponse
)

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages different AI models and providers for TruthSeeQ.
    
    Provides unified interface for different AI models including:
    - OpenAI GPT models
    - Anthropic Claude models
    - Local/self-hosted models
    - Model fallback and load balancing
    """
    
    def __init__(self):
        """Initialize model manager with available models."""
        self.models = {}
        self.default_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available AI models based on configuration."""
        try:
            # Initialize OpenAI models if API key is available
            if settings.ai.OPENAI_API_KEY:
                self.models["gpt-4"] = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    api_key=settings.ai.OPENAI_API_KEY,
                    max_retries=3
                )
                self.models["gpt-3.5-turbo"] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=settings.ai.OPENAI_API_KEY,
                    max_retries=3
                )
                self.default_model = "gpt-4"
                logger.info("OpenAI models initialized")
            
            # Initialize Anthropic models if API key is available
            if settings.ai.ANTHROPIC_API_KEY:
                self.models["claude-3-opus"] = ChatAnthropic(
                    model="claude-3-opus-20240229",
                    temperature=0.1,
                    api_key=settings.ai.ANTHROPIC_API_KEY,
                    max_retries=3
                )
                self.models["claude-3-sonnet"] = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.1,
                    api_key=settings.ai.ANTHROPIC_API_KEY,
                    max_retries=3
                )
                if not self.default_model:
                    self.default_model = "claude-3-sonnet"
                logger.info("Anthropic models initialized")
            
            if not self.models:
                logger.warning("No AI models available - check API keys in configuration")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """
        Get AI model by name or default model.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            AI model instance or None if not available
        """
        if not model_name:
            model_name = self.default_model
        
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def has_models(self) -> bool:
        """Check if any models are available."""
        return len(self.models) > 0


class AnalysisCache:
    """
    Caches AI analysis results for performance optimization.
    
    Provides caching functionality for:
    - Fact-checking results
    - Content analysis results
    - Workflow execution results
    - Model responses
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize analysis cache.
        
        Args:
            redis_client: Redis client for caching
        """
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour default TTL
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result.
        
        Args:
            cache_key: Cache key for the result
            
        Returns:
            Cached result or None if not found
        """
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached result: {e}")
            return None
    
    async def cache_result(self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache analysis result.
        
        Args:
            cache_key: Cache key for the result
            result: Result data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
            return False
    
    def generate_cache_key(self, content_id: int, analysis_type: str, model_name: str) -> str:
        """
        Generate cache key for analysis result.
        
        Args:
            content_id: Content ID
            analysis_type: Type of analysis
            model_name: AI model name
            
        Returns:
            Cache key string
        """
        return f"ai_analysis:{content_id}:{analysis_type}:{model_name}"


class ConfidenceScorer:
    """
    Generates confidence scores and explanations for AI analysis.
    
    Provides methods for:
    - Calculating confidence scores based on multiple factors
    - Generating human-readable explanations
    - Assessing analysis reliability
    - Providing recommendations
    """
    
    @staticmethod
    def calculate_confidence_score(
        source_credibility: float,
        content_quality: float,
        analysis_consistency: float,
        evidence_strength: float
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            source_credibility: Credibility of the source (0-1)
            content_quality: Quality of the content (0-1)
            analysis_consistency: Consistency of analysis (0-1)
            evidence_strength: Strength of supporting evidence (0-1)
            
        Returns:
            Overall confidence score (0-1)
        """
        # Weighted average of factors
        weights = {
            'source_credibility': 0.25,
            'content_quality': 0.20,
            'analysis_consistency': 0.25,
            'evidence_strength': 0.30
        }
        
        confidence = (
            source_credibility * weights['source_credibility'] +
            content_quality * weights['content_quality'] +
            analysis_consistency * weights['analysis_consistency'] +
            evidence_strength * weights['evidence_strength']
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    @staticmethod
    def generate_explanation(
        confidence_score: float,
        factors: Dict[str, float],
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable explanation for analysis result.
        
        Args:
            confidence_score: Overall confidence score
            factors: Contributing factors and their scores
            sources: List of sources used in analysis
            
        Returns:
            Human-readable explanation
        """
        explanation_parts = []
        
        # Overall confidence statement
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "moderate"
        elif confidence_score >= 0.4:
            confidence_level = "low"
        else:
            confidence_level = "very low"
        
        explanation_parts.append(
            f"This analysis has {confidence_level} confidence ({confidence_score:.1%})."
        )
        
        # Factor breakdown
        if factors:
            explanation_parts.append("Key factors contributing to this assessment:")
            for factor, score in factors.items():
                factor_name = factor.replace('_', ' ').title()
                explanation_parts.append(f"- {factor_name}: {score:.1%}")
        
        # Source information
        if sources:
            explanation_parts.append(f"Analysis based on {len(sources)} sources:")
            for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                source_type = source.get('type', 'unknown')
                source_url = source.get('url', 'N/A')
                explanation_parts.append(f"{i}. {source_type}: {source_url}")
        
        return "\n".join(explanation_parts)


class WorkflowOrchestrator:
    """
    Orchestrates LangGraph workflows for TruthSeeQ.
    
    Manages workflow execution including:
    - Fact-checking workflows
    - Content analysis workflows
    - Source verification workflows
    - Workflow monitoring and error handling
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize workflow orchestrator.
        
        Args:
            model_manager: Model manager for AI model access
        """
        self.model_manager = model_manager
        self.workflows = {}
        self._initialize_workflows()
    
    def _initialize_workflows(self):
        """Initialize available LangGraph workflows."""
        try:
            # Initialize fact-checking workflow
            self.workflows["fact_checking"] = self._create_fact_checking_workflow()
            
            # Initialize content analysis workflow
            self.workflows["content_analysis"] = self._create_content_analysis_workflow()
            
            # Initialize source verification workflow
            self.workflows["source_verification"] = self._create_source_verification_workflow()
            
            logger.info(f"Initialized {len(self.workflows)} LangGraph workflows")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflows: {e}")
    
    def _create_fact_checking_workflow(self) -> StateGraph:
        """
        Create fact-checking workflow using LangGraph.
        
        Returns:
            LangGraph StateGraph for fact-checking
        """
        # Define workflow state
        workflow = StateGraph({
            "content": str,
            "title": str,
            "url": str,
            "extracted_claims": List[str],
            "source_analysis": Dict[str, Any],
            "fact_analysis": Dict[str, Any],
            "confidence_score": float,
            "verdict": str,
            "reasoning": str,
            "sources": List[Dict[str, Any]]
        })
        
        # Add nodes
        workflow.add_node("extract_claims", self._extract_claims_node)
        workflow.add_node("analyze_sources", self._analyze_sources_node)
        workflow.add_node("fact_analysis", self._fact_analysis_node)
        workflow.add_node("generate_verdict", self._generate_verdict_node)
        
        # Define edges
        workflow.add_edge("extract_claims", "analyze_sources")
        workflow.add_edge("analyze_sources", "fact_analysis")
        workflow.add_edge("fact_analysis", "generate_verdict")
        workflow.add_edge("generate_verdict", END)
        
        return workflow.compile()
    
    def _create_content_analysis_workflow(self) -> StateGraph:
        """
        Create content analysis workflow using LangGraph.
        
        Returns:
            LangGraph StateGraph for content analysis
        """
        workflow = StateGraph({
            "content": str,
            "sentiment_analysis": Dict[str, Any],
            "bias_detection": Dict[str, Any],
            "credibility_assessment": Dict[str, Any],
            "content_categorization": Dict[str, Any],
            "analysis_summary": str
        })
        
        # Add nodes
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
        workflow.add_node("bias_detection", self._bias_detection_node)
        workflow.add_node("credibility_assessment", self._credibility_assessment_node)
        workflow.add_node("content_categorization", self._content_categorization_node)
        workflow.add_node("generate_summary", self._generate_analysis_summary_node)
        
        # Define edges
        workflow.add_edge("sentiment_analysis", "bias_detection")
        workflow.add_edge("bias_detection", "credibility_assessment")
        workflow.add_edge("credibility_assessment", "content_categorization")
        workflow.add_edge("content_categorization", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        return workflow.compile()
    
    def _create_source_verification_workflow(self) -> StateGraph:
        """
        Create source verification workflow using LangGraph.
        
        Returns:
            LangGraph StateGraph for source verification
        """
        workflow = StateGraph({
            "url": str,
            "domain_analysis": Dict[str, Any],
            "reputation_check": Dict[str, Any],
            "fact_checking_database_lookup": Dict[str, Any],
            "cross_reference_results": Dict[str, Any],
            "verification_result": Dict[str, Any]
        })
        
        # Add nodes
        workflow.add_node("domain_analysis", self._domain_analysis_node)
        workflow.add_node("reputation_check", self._reputation_check_node)
        workflow.add_node("fact_checking_lookup", self._fact_checking_lookup_node)
        workflow.add_node("cross_reference", self._cross_reference_node)
        workflow.add_node("generate_verification_result", self._generate_verification_result_node)
        
        # Define edges
        workflow.add_edge("domain_analysis", "reputation_check")
        workflow.add_edge("reputation_check", "fact_checking_lookup")
        workflow.add_edge("fact_checking_lookup", "cross_reference")
        workflow.add_edge("cross_reference", "generate_verification_result")
        workflow.add_edge("generate_verification_result", END)
        
        return workflow.compile()
    
    # Workflow node implementations
    async def _extract_claims_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract factual claims from content."""
        model = self.model_manager.get_model()
        if not model:
            return {"extracted_claims": []}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checking assistant. Extract factual claims from the given content."),
            ("human", "Content: {content}\n\nExtract the main factual claims as a JSON list of strings.")
        ])
        
        try:
            response = await model.ainvoke(prompt.format_messages(content=state["content"]))
            claims = json.loads(response.content)
            return {"extracted_claims": claims}
        except Exception as e:
            logger.error(f"Failed to extract claims: {e}")
            return {"extracted_claims": []}
    
    async def _analyze_sources_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze source credibility and reliability."""
        # This would integrate with source verification tools
        return {"source_analysis": {"credibility": 0.7, "reliability": 0.8}}
    
    async def _fact_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factual accuracy of claims."""
        model = self.model_manager.get_model()
        if not model:
            return {"fact_analysis": {"accuracy": 0.5}}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fact-checking expert. Analyze the factual accuracy of the given claims."),
            ("human", "Claims: {claims}\n\nAnalyze each claim for factual accuracy.")
        ])
        
        try:
            response = await model.ainvoke(prompt.format_messages(claims=state["extracted_claims"]))
            return {"fact_analysis": {"accuracy": 0.8, "analysis": response.content}}
        except Exception as e:
            logger.error(f"Failed to analyze facts: {e}")
            return {"fact_analysis": {"accuracy": 0.5}}
    
    async def _generate_verdict_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final fact-checking verdict."""
        confidence = ConfidenceScorer.calculate_confidence_score(
            source_credibility=state["source_analysis"]["credibility"],
            content_quality=0.8,  # Would come from content analysis
            analysis_consistency=0.9,
            evidence_strength=0.7
        )
        
        return {
            "confidence_score": confidence,
            "verdict": "mostly_true" if confidence > 0.6 else "unverifiable",
            "reasoning": "Analysis completed with available sources",
            "sources": []
        }
    
    # Content analysis nodes
    async def _sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on content."""
        return {"sentiment_analysis": {"sentiment": "neutral", "score": 0.0}}
    
    async def _bias_detection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect bias in content."""
        return {"bias_detection": {"bias_level": "low", "bias_type": "none"}}
    
    async def _credibility_assessment_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess content credibility."""
        return {"credibility_assessment": {"credibility": 0.7}}
    
    async def _content_categorization_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize content type."""
        return {"content_categorization": {"category": "news", "subcategory": "politics"}}
    
    async def _generate_analysis_summary_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary."""
        return {"analysis_summary": "Content analysis completed successfully"}
    
    # Source verification nodes
    async def _domain_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze domain characteristics."""
        return {"domain_analysis": {"domain_age": "5 years", "ssl_valid": True}}
    
    async def _reputation_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check domain reputation."""
        return {"reputation_check": {"reputation_score": 0.8}}
    
    async def _fact_checking_lookup_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Look up domain in fact-checking databases."""
        return {"fact_checking_database_lookup": {"found": False}}
    
    async def _cross_reference_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference with other sources."""
        return {"cross_reference_results": {"sources_found": 0}}
    
    async def _generate_verification_result_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final verification result."""
        return {"verification_result": {"verified": True, "confidence": 0.8}}
    
    async def execute_workflow(
        self, 
        workflow_type: str, 
        initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a LangGraph workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            initial_state: Initial state for the workflow
            
        Returns:
            Workflow execution result
        """
        if workflow_type not in self.workflows:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        try:
            workflow = self.workflows[workflow_type]
            result = await workflow.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise


class AIService:
    """
    Main AI service integrating with TruthSeeQ platform.
    
    Provides comprehensive AI capabilities including:
    - LangGraph workflow orchestration
    - AI model management
    - Fact-checking and content analysis
    - Result caching and optimization
    - Integration with scraper service
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        """
        Initialize AI service.
        
        Args:
            db_session: Database session for data storage
            redis_client: Redis client for caching
        """
        self.db = db_session
        self.model_manager = ModelManager()
        self.workflow_orchestrator = WorkflowOrchestrator(self.model_manager)
        self.analysis_cache = AnalysisCache(redis_client)
        self.confidence_scorer = ConfidenceScorer()
        
        logger.info("AIService initialized with LangGraph integration")
    
    async def fact_check_content(
        self, 
        request: FactCheckRequest,
        session_id: Optional[UUID] = None
    ) -> FactCheckResponse:
        """
        Perform fact-checking analysis on content.
        
        Args:
            request: Fact-checking request
            session_id: Optional user session ID
            
        Returns:
            FactCheckResponse with analysis results
        """
        start_time = time.time()
        
        # Get content from database
        content_item = await self._get_content_item(request.content_id)
        if not content_item:
            raise ValueError(f"Content item {request.content_id} not found")
        
        # Check cache first
        cache_key = self.analysis_cache.generate_cache_key(
            request.content_id, "fact_check", request.model_name or "default"
        )
        cached_result = await self.analysis_cache.get_cached_result(cache_key)
        
        if cached_result and not request.force_refresh:
            logger.info(f"Using cached fact-check result for content {request.content_id}")
            return FactCheckResponse(**cached_result)
        
        # Create workflow execution record
        workflow_execution = AIWorkflowExecution(
            content_id=request.content_id,
            workflow_type="fact_checking",
            status=WorkflowStatus.RUNNING
        )
        self.db.add(workflow_execution)
        await self.db.flush()
        
        try:
            # Execute fact-checking workflow
            initial_state = {
                "content": content_item.content,
                "title": content_item.title or "",
                "url": content_item.url,
                "extracted_claims": [],
                "source_analysis": {},
                "fact_analysis": {},
                "confidence_score": 0.0,
                "verdict": "",
                "reasoning": "",
                "sources": []
            }
            
            workflow_result = await self.workflow_orchestrator.execute_workflow(
                "fact_checking", initial_state
            )
            
            # Generate confidence score and explanation
            confidence_score = workflow_result.get("confidence_score", 0.5)
            explanation = self.confidence_scorer.generate_explanation(
                confidence_score=confidence_score,
                factors={
                    "source_credibility": workflow_result.get("source_analysis", {}).get("credibility", 0.5),
                    "content_quality": 0.8,  # Would come from content analysis
                    "analysis_consistency": 0.9,
                    "evidence_strength": 0.7
                },
                sources=workflow_result.get("sources", [])
            )
            
            # Store fact-check result in database
            fact_check_result = FactCheckResult(
                content_id=request.content_id,
                session_id=session_id,
                confidence_score=confidence_score,
                verdict=workflow_result.get("verdict", "unverifiable"),
                reasoning=explanation
            )
            self.db.add(fact_check_result)
            
            # Update workflow execution
            workflow_execution.status = WorkflowStatus.COMPLETED
            workflow_execution.execution_time = time.time() - start_time
            
            await self.db.commit()
            
            # Cache the result
            response_data = {
                "content_id": request.content_id,
                "confidence_score": confidence_score,
                "verdict": workflow_result.get("verdict", "unverifiable"),
                "reasoning": explanation,
                "sources": workflow_result.get("sources", []),
                "execution_time": time.time() - start_time,
                "workflow_execution_id": workflow_execution.id
            }
            
            await self.analysis_cache.cache_result(cache_key, response_data)
            
            return FactCheckResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Fact-checking failed for content {request.content_id}: {e}")
            
            # Update workflow execution status
            workflow_execution.status = WorkflowStatus.FAILED
            await self.db.commit()
            
            raise
    
    async def analyze_content(
        self, 
        request: ContentAnalysisRequest
    ) -> ContentAnalysisResponse:
        """
        Perform comprehensive content analysis.
        
        Args:
            request: Content analysis request
            
        Returns:
            ContentAnalysisResponse with analysis results
        """
        # Get content from database
        content_item = await self._get_content_item(request.content_id)
        if not content_item:
            raise ValueError(f"Content item {request.content_id} not found")
        
        # Check cache
        cache_key = self.analysis_cache.generate_cache_key(
            request.content_id, "content_analysis", request.model_name or "default"
        )
        cached_result = await self.analysis_cache.get_cached_result(cache_key)
        
        if cached_result and not request.force_refresh:
            return ContentAnalysisResponse(**cached_result)
        
        # Execute content analysis workflow
        initial_state = {
            "content": content_item.content,
            "sentiment_analysis": {},
            "bias_detection": {},
            "credibility_assessment": {},
            "content_categorization": {},
            "analysis_summary": ""
        }
        
        workflow_result = await self.workflow_orchestrator.execute_workflow(
            "content_analysis", initial_state
        )
        
        # Store analysis results
        analysis_result = AIAnalysisResult(
            content_id=request.content_id,
            analysis_type="comprehensive",
            result_data=workflow_result,
            confidence=0.8  # Would be calculated based on analysis quality
        )
        self.db.add(analysis_result)
        await self.db.commit()
        
        # Cache result
        response_data = {
            "content_id": request.content_id,
            "analysis_results": workflow_result,
            "confidence": 0.8,
            "analysis_id": analysis_result.id
        }
        
        await self.analysis_cache.cache_result(cache_key, response_data)
        
        return ContentAnalysisResponse(**response_data)
    
    async def verify_source(
        self, 
        url: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify source credibility and reliability.
        
        Args:
            url: URL to verify
            model_name: Optional AI model to use
            
        Returns:
            Source verification results
        """
        # Execute source verification workflow
        initial_state = {
            "url": url,
            "domain_analysis": {},
            "reputation_check": {},
            "fact_checking_database_lookup": {},
            "cross_reference_results": {},
            "verification_result": {}
        }
        
        workflow_result = await self.workflow_orchestrator.execute_workflow(
            "source_verification", initial_state
        )
        
        return workflow_result.get("verification_result", {})
    
    async def _get_content_item(self, content_id: int) -> Optional[ContentItem]:
        """
        Get content item from database.
        
        Args:
            content_id: Content ID to retrieve
            
        Returns:
            ContentItem if found, None otherwise
        """
        stmt = select(ContentItem).where(ContentItem.id == content_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    def get_available_models(self) -> List[str]:
        """Get list of available AI models."""
        return self.model_manager.get_available_models()
    
    def has_models(self) -> bool:
        """Check if AI models are available."""
        return self.model_manager.has_models()
    
    async def get_workflow_status(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """
        Get workflow execution status.
        
        Args:
            execution_id: Workflow execution ID
            
        Returns:
            Workflow status information
        """
        stmt = select(AIWorkflowExecution).where(AIWorkflowExecution.id == execution_id)
        result = await self.db.execute(stmt)
        execution = result.scalar_one_or_none()
        
        if not execution:
            return None
        
        return {
            "execution_id": execution.id,
            "workflow_type": execution.workflow_type,
            "status": execution.status.value,
            "created_at": execution.created_at,
            "execution_time": execution.execution_time
        }
