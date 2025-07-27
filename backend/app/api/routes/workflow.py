"""
Workflow API routes for TruthSeeQ backend.

This module provides RESTful API endpoints for executing and managing
LangGraph-based workflows including fact-checking, content analysis, and source verification.

Routes:
    POST /workflow/fact-check - Execute fact-checking workflow
    POST /workflow/content-analysis - Execute content analysis workflow
    POST /workflow/source-verification - Execute source verification workflow
    POST /workflow/comprehensive - Execute all workflows for comprehensive analysis
    GET /workflow/{workflow_id}/status - Get workflow execution status
    GET /workflow/history - Get workflow execution history
    GET /workflow/metrics - Get workflow performance metrics
    GET /workflow/health - Get workflow service health
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.exceptions import (
    ValidationError,
    RateLimitExceededError,
    WorkflowExecutionError
)
from ...database.database import get_db
from ...database.models import UserSession
from ...schemas.content import (
    FactCheckRequest,
    FactCheckResponse,
    ContentAnalysisRequest,
    ContentAnalysisResponse,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse
)
from ...workflow.orchestrator import WorkflowOrchestrator
from ...core.rate_limiting import RateLimiter
from ..dependencies import get_current_session, get_rate_limiter

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/workflow", tags=["workflow"])


# ========================================
# Dependency Injection
# ========================================

async def get_workflow_orchestrator(
    db: AsyncSession = Depends(get_db)
) -> WorkflowOrchestrator:
    """
    Get WorkflowOrchestrator instance with database session.
    
    Args:
        db: Database session
        
    Returns:
        Initialized WorkflowOrchestrator
    """
    # In a real implementation, you'd also inject Redis client
    # For now, we'll create a basic orchestrator
    import redis.asyncio as redis
    
    # Create Redis client (this would be injected in production)
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )
    
    return WorkflowOrchestrator(db, redis_client)


# ========================================
# Workflow Execution Endpoints
# ========================================

@router.post("/fact-check", response_model=FactCheckResponse)
async def execute_fact_checking(
    request: FactCheckRequest,
    background_tasks: BackgroundTasks,
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Execute fact-checking workflow.
    
    Performs comprehensive fact-checking analysis using LangGraph workflows
    including claims extraction, source verification, and confidence scoring.
    
    Args:
        request: Fact-checking request with URL and options
        background_tasks: FastAPI background tasks
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        FactCheckResponse with fact-checking results
        
    Raises:
        HTTPException: If rate limit exceeded or workflow fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"fact_check:{current_session.session_id if current_session else 'anonymous'}",
            limit=5,  # 5 fact-check requests per minute
            window=60
        )
        
        logger.info(f"Starting fact-checking workflow for {request.url}")
        
        # Execute workflow
        result = await orchestrator.execute_fact_checking(
            request,
            session_id=current_session.session_id if current_session else None
        )
        
        logger.info(f"Fact-checking completed: workflow_id={result.workflow_id}, verdict={result.verdict}")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for fact-checking requests")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Fact-checking workflow failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during fact-checking")


@router.post("/content-analysis", response_model=ContentAnalysisResponse)
async def execute_content_analysis(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Execute content analysis workflow.
    
    Performs comprehensive content analysis including sentiment analysis,
    bias detection, quality assessment, and credibility analysis.
    
    Args:
        request: Content analysis request with URL and options
        background_tasks: FastAPI background tasks
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        ContentAnalysisResponse with analysis results
        
    Raises:
        HTTPException: If rate limit exceeded or workflow fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"content_analysis:{current_session.session_id if current_session else 'anonymous'}",
            limit=10,  # 10 content analysis requests per minute
            window=60
        )
        
        logger.info(f"Starting content analysis workflow for {request.url}")
        
        # Execute workflow
        result = await orchestrator.execute_content_analysis(request)
        
        logger.info(f"Content analysis completed: workflow_id={result.workflow_id}, quality_score={result.content_quality_score}")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for content analysis requests")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Content analysis workflow failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during content analysis")


@router.post("/source-verification")
async def execute_source_verification(
    url: str = Query(..., description="URL to verify"),
    model_name: Optional[str] = Query(None, description="AI model to use"),
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Execute source verification workflow.
    
    Performs source credibility analysis including domain analysis,
    reputation checking, and trust indicator evaluation.
    
    Args:
        url: URL to verify
        model_name: Optional AI model name
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        Source verification results
        
    Raises:
        HTTPException: If rate limit exceeded or workflow fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"source_verification:{current_session.session_id if current_session else 'anonymous'}",
            limit=15,  # 15 source verification requests per minute
            window=60
        )
        
        logger.info(f"Starting source verification workflow for {url}")
        
        # Execute workflow
        result = await orchestrator.execute_source_verification(
            url,
            model_name,
            session_id=current_session.session_id if current_session else None
        )
        
        logger.info(f"Source verification completed: workflow_id={result['workflow_id']}, reputation_score={result['reputation_score']}")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for source verification requests")
    except Exception as e:
        logger.error(f"Source verification workflow failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during source verification")


@router.post("/comprehensive")
async def execute_comprehensive_analysis(
    url: str = Query(..., description="URL to analyze"),
    model_name: Optional[str] = Query(None, description="AI model to use"),
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
):
    """
    Execute comprehensive analysis workflow.
    
    Performs all workflows (fact-checking, content analysis, source verification)
    in parallel for comprehensive content evaluation.
    
    Args:
        url: URL to analyze
        model_name: Optional AI model name
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        rate_limiter: Rate limiting dependency
        
    Returns:
        Comprehensive analysis results
        
    Raises:
        HTTPException: If rate limit exceeded or workflow fails
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(
            key=f"comprehensive:{current_session.session_id if current_session else 'anonymous'}",
            limit=3,  # 3 comprehensive analysis requests per minute
            window=60
        )
        
        logger.info(f"Starting comprehensive analysis workflow for {url}")
        
        # Execute workflow
        result = await orchestrator.execute_comprehensive_analysis(
            url,
            session_id=current_session.session_id if current_session else None,
            model_name=model_name
        )
        
        logger.info(f"Comprehensive analysis completed: workflow_id={result['workflow_id']}")
        
        return result
        
    except RateLimitExceededError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded for comprehensive analysis requests")
    except Exception as e:
        logger.error(f"Comprehensive analysis workflow failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during comprehensive analysis")


# ========================================
# Workflow Status and Management Endpoints
# ========================================

@router.get("/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Get workflow execution status.
    
    Retrieves the current status of a workflow execution including
    progress, completion status, and any error messages.
    
    Args:
        workflow_id: Workflow identifier
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        
    Returns:
        Workflow status information
        
    Raises:
        HTTPException: If workflow not found
    """
    try:
        status_info = await orchestrator.get_workflow_status(workflow_id)
        
        if not status_info:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        logger.info(f"Retrieved workflow status {workflow_id}: {status_info['status']}")
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve workflow status {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history")
async def get_workflow_history(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Get workflow execution history.
    
    Retrieves a paginated list of workflow executions for the current session.
    
    Args:
        skip: Number of items to skip
        limit: Number of items to return
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        
    Returns:
        List of workflow execution records
    """
    try:
        history = await orchestrator.get_workflow_history(
            session_id=current_session.session_id if current_session else None,
            limit=limit
        )
        
        # Apply pagination
        paginated_history = history[skip:skip + limit]
        
        logger.info(f"Retrieved workflow history: {len(paginated_history)} items")
        
        return {
            "items": paginated_history,
            "total": len(history),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve workflow history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics")
async def get_workflow_metrics(
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Get workflow performance metrics.
    
    Retrieves performance metrics for workflow executions including
    success rates, average execution times, and workflow type statistics.
    
    Args:
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (optional)
        
    Returns:
        Workflow performance metrics
    """
    try:
        metrics = await orchestrator.get_workflow_metrics()
        
        logger.info(f"Retrieved workflow metrics: {metrics['total_workflows']} total workflows")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to retrieve workflow metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Workflow Management Endpoints
# ========================================

@router.post("/cleanup")
async def cleanup_expired_workflows(
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator),
    current_session: Optional[UserSession] = Depends(get_current_session)
):
    """
    Clean up expired workflow data.
    
    Removes old workflow execution records to prevent database bloat.
    Requires an active session.
    
    Args:
        orchestrator: Injected workflow orchestrator
        current_session: Current user session (required)
        
    Returns:
        Cleanup results
        
    Raises:
        HTTPException: If unauthorized or cleanup fails
    """
    try:
        if not current_session:
            raise HTTPException(status_code=401, detail="Authentication required for cleanup")
        
        cleaned_count = await orchestrator.cleanup_expired_workflows()
        
        logger.info(f"Workflow cleanup completed: {cleaned_count} workflows cleaned")
        
        return {
            "message": f"Cleaned up {cleaned_count} expired workflows",
            "cleaned_count": cleaned_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during cleanup")


# ========================================
# Health Check Endpoint
# ========================================

@router.get("/health")
async def workflow_health_check(
    orchestrator: WorkflowOrchestrator = Depends(get_workflow_orchestrator)
):
    """
    Health check endpoint for workflow service.
    
    Returns:
        Workflow service health status
    """
    try:
        health_status = await orchestrator.health_check()
        
        return {
            "service": "workflow",
            "status": health_status["status"],
            "database": health_status["database"],
            "redis": health_status["redis"],
            "active_workflows": health_status["active_workflows"],
            "total_workflows": health_status["total_workflows"],
            "average_execution_time": health_status["average_execution_time"]
        }
        
    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        return {
            "service": "workflow",
            "status": "unhealthy",
            "error": str(e)
        } 