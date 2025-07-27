"""
Workflow Orchestrator for TruthSeeQ

This module provides a unified orchestrator for managing and executing
all TruthSeeQ workflows, integrating with database, caching, and monitoring systems.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

# Import local modules
from .workflows import (
    FactCheckingWorkflow, ContentAnalysisWorkflow, SourceVerificationWorkflow,
    WorkflowManager, fact_check_url, analyze_content, verify_source
)
from .state import (
    FactCheckState, ContentAnalysisState, SourceVerificationState,
    VerdictType, ConfidenceLevel
)
from ..config import settings
from ..database.models import (
    ContentItem, AIAnalysisResult, AIWorkflowExecution, 
    FactCheckResult, WorkflowStatus, SourceType
)
from ..schemas.content import (
    FactCheckRequest, FactCheckResponse, FactCheckResult as FactCheckResultSchema,
    ContentAnalysisRequest, ContentAnalysisResponse,
    WorkflowExecutionRequest, WorkflowExecutionResponse
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Main orchestrator for TruthSeeQ workflows.
    
    This class provides a unified interface for:
    - Executing different types of workflows
    - Managing workflow state and persistence
    - Caching workflow results
    - Monitoring workflow performance
    - Error handling and recovery
    """
    
    def __init__(
        self, 
        db_session: AsyncSession, 
        redis_client: redis.Redis,
        default_model: str = "gpt-4.1-mini-2025-04-14"
    ):
        """
        Initialize workflow orchestrator.
        
        Args:
            db_session: Database session for persistence
            redis_client: Redis client for caching
            default_model: Default AI model to use
        """
        self.db = db_session
        self.redis = redis_client
        self.default_model = default_model
        self.workflow_manager = WorkflowManager(default_model)
        
        # Cache settings
        self.cache_ttl = 3600  # 1 hour default TTL
        self.max_concurrent_workflows = 10
        self.active_workflows = {}
        
        logger.info("WorkflowOrchestrator initialized")
    
    async def execute_fact_checking(
        self, 
        request: FactCheckRequest,
        session_id: Optional[UUID] = None
    ) -> FactCheckResponse:
        """
        Execute fact-checking workflow with full integration.
        
        Args:
            request: Fact-checking request
            session_id: Optional user session ID
            
        Returns:
            FactCheckResponse with results
        """
        start_time = time.time()
        workflow_id = str(uuid4())
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key("fact_check", request.url, request.model_name)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result and not request.force_refresh:
                logger.info(f"Using cached fact-check result for {request.url}")
                return FactCheckResponse(**cached_result)
            
            # Create workflow execution record
            workflow_execution = AIWorkflowExecution(
                content_id=request.content_id if hasattr(request, 'content_id') else None,
                workflow_type="fact_checking",
                status=WorkflowStatus.RUNNING,
                workflow_id=workflow_id
            )
            self.db.add(workflow_execution)
            await self.db.flush()
            
            # Execute workflow
            result = await self.workflow_manager.execute_fact_checking(
                request.url, 
                str(session_id) if session_id else None,
                workflow_id
            )
            
            # Store result in database
            if result["status"] == "completed":
                fact_check_result = FactCheckResult(
                    content_id=request.content_id if hasattr(request, 'content_id') else None,
                    session_id=session_id,
                    confidence_score=result["confidence_score"],
                    verdict=result["verdict"].value,
                    reasoning=result["reasoning"],
                    workflow_execution_id=workflow_execution.id
                )
                self.db.add(fact_check_result)
                
                # Update workflow execution
                workflow_execution.status = WorkflowStatus.COMPLETED
                workflow_execution.execution_time = time.time() - start_time
                
                await self.db.commit()
                
                # Cache the result
                response_data = {
                    "workflow_id": workflow_id,
                    "url": request.url,
                    "confidence_score": result["confidence_score"],
                    "verdict": result["verdict"].value,
                    "reasoning": result["reasoning"],
                    "supporting_evidence": result.get("supporting_evidence", []),
                    "contradicting_evidence": result.get("contradicting_evidence", []),
                    "processing_time": time.time() - start_time,
                    "model_used": result.get("model_used", self.default_model),
                    "sources": result.get("verification_sources", [])
                }
                
                await self._cache_result(cache_key, response_data)
                
                return FactCheckResponse(**response_data)
            else:
                # Handle error
                workflow_execution.status = WorkflowStatus.FAILED
                workflow_execution.execution_time = time.time() - start_time
                await self.db.commit()
                
                raise Exception(f"Workflow failed: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Fact-checking failed for {request.url}: {e}")
            
            # Update workflow execution status
            if 'workflow_execution' in locals():
                workflow_execution.status = WorkflowStatus.FAILED
                workflow_execution.execution_time = time.time() - start_time
                await self.db.commit()
            
            raise
    
    async def execute_content_analysis(
        self, 
        request: ContentAnalysisRequest
    ) -> ContentAnalysisResponse:
        """
        Execute content analysis workflow.
        
        Args:
            request: Content analysis request
            
        Returns:
            ContentAnalysisResponse with results
        """
        start_time = time.time()
        workflow_id = str(uuid4())
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key("content_analysis", request.url, request.model_name)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result and not request.force_refresh:
                logger.info(f"Using cached content analysis result for {request.url}")
                return ContentAnalysisResponse(**cached_result)
            
            # Execute workflow
            result = await self.workflow_manager.execute_content_analysis(
                request.url,
                str(request.session_id) if hasattr(request, 'session_id') else None,
                workflow_id
            )
            
            # Store analysis results
            if result["status"] == "completed":
                analysis_result = AIAnalysisResult(
                    content_id=request.content_id if hasattr(request, 'content_id') else None,
                    analysis_type="content_analysis",
                    result_data=result,
                    confidence=result.get("content_quality_score", 0.5),
                    workflow_execution_id=workflow_id
                )
                self.db.add(analysis_result)
                await self.db.commit()
                
                # Cache result
                response_data = {
                    "workflow_id": workflow_id,
                    "url": request.url,
                    "sentiment_analysis": result.get("sentiment_analysis", {}),
                    "bias_analysis": result.get("bias_analysis", {}),
                    "content_quality_score": result.get("content_quality_score", 0.0),
                    "readability_score": result.get("readability_score", 0.0),
                    "content_category": result.get("content_category", "unknown"),
                    "topics": result.get("topics", []),
                    "entities": result.get("entities", []),
                    "analysis_summary": result.get("analysis_summary", ""),
                    "processing_time": time.time() - start_time,
                    "model_used": result.get("model_used", self.default_model)
                }
                
                await self._cache_result(cache_key, response_data)
                
                return ContentAnalysisResponse(**response_data)
            else:
                raise Exception(f"Content analysis failed: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Content analysis failed for {request.url}: {e}")
            raise
    
    async def execute_source_verification(
        self, 
        url: str,
        model_name: Optional[str] = None,
        session_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Execute source verification workflow.
        
        Args:
            url: URL to verify
            model_name: Optional AI model name
            session_id: Optional user session ID
            
        Returns:
            Source verification results
        """
        start_time = time.time()
        workflow_id = str(uuid4())
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key("source_verification", url, model_name)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"Using cached source verification result for {url}")
                return cached_result
            
            # Execute workflow
            result = await self.workflow_manager.execute_source_verification(
                url,
                str(session_id) if session_id else None,
                workflow_id
            )
            
            # Cache result
            response_data = {
                "workflow_id": workflow_id,
                "url": url,
                "domain": result.get("domain", ""),
                "reputation_score": result.get("reputation_score", 0.0),
                "trust_indicators": result.get("trust_indicators", []),
                "red_flags": result.get("red_flags", []),
                "verification_score": result.get("verification_score", 0.0),
                "verification_status": result.get("verification_status", "unknown"),
                "confidence_level": result.get("confidence_level", ConfidenceLevel.VERY_LOW).value,
                "verification_summary": result.get("verification_summary", ""),
                "processing_time": time.time() - start_time,
                "model_used": result.get("model_used", self.default_model)
            }
            
            await self._cache_result(cache_key, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Source verification failed for {url}: {e}")
            raise
    
    async def execute_comprehensive_analysis(
        self, 
        url: str,
        session_id: Optional[UUID] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all workflows for comprehensive analysis.
        
        Args:
            url: URL to analyze
            session_id: Optional user session ID
            model_name: Optional AI model name
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        workflow_id = str(uuid4())
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key("comprehensive", url, model_name)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"Using cached comprehensive analysis result for {url}")
                return cached_result
            
            # Execute all workflows in parallel
            result = await self.workflow_manager.execute_comprehensive_analysis(
                url, 
                str(session_id) if session_id else None
            )
            
            # Add processing time
            result["processing_time"] = time.time() - start_time
            result["workflow_id"] = workflow_id
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {url}: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific workflow.
        
        Args:
            workflow_id: Workflow ID to check
            
        Returns:
            Workflow status information or None if not found
        """
        try:
            # Check database for workflow execution
            stmt = select(AIWorkflowExecution).where(AIWorkflowExecution.workflow_id == workflow_id)
            result = await self.db.execute(stmt)
            execution = result.scalar_one_or_none()
            
            if execution:
                return {
                    "workflow_id": workflow_id,
                    "workflow_type": execution.workflow_type,
                    "status": execution.status.value,
                    "created_at": execution.created_at.isoformat(),
                    "execution_time": execution.execution_time,
                    "content_id": execution.content_id
                }
            
            # Check active workflows
            if workflow_id in self.active_workflows:
                return {
                    "workflow_id": workflow_id,
                    "status": "running",
                    "started_at": self.active_workflows[workflow_id]["started_at"].isoformat(),
                    "elapsed_time": (datetime.utcnow() - self.active_workflows[workflow_id]["started_at"]).total_seconds()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow status for {workflow_id}: {e}")
            return None
    
    async def get_workflow_history(
        self, 
        session_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get workflow execution history.
        
        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of workflow execution records
        """
        try:
            stmt = select(AIWorkflowExecution).order_by(AIWorkflowExecution.created_at.desc()).limit(limit)
            
            if session_id:
                stmt = stmt.where(AIWorkflowExecution.session_id == session_id)
            
            result = await self.db.execute(stmt)
            executions = result.scalars().all()
            
            return [
                {
                    "workflow_id": exec.workflow_id,
                    "workflow_type": exec.workflow_type,
                    "status": exec.status.value,
                    "created_at": exec.created_at.isoformat(),
                    "execution_time": exec.execution_time,
                    "content_id": exec.content_id
                }
                for exec in executions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get workflow history: {e}")
            return []
    
    def _generate_cache_key(self, workflow_type: str, url: str, model_name: Optional[str] = None) -> str:
        """
        Generate cache key for workflow result.
        
        Args:
            workflow_type: Type of workflow
            url: URL being processed
            model_name: Optional AI model name
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a hash of the URL to keep cache keys manageable
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        model_suffix = f"_{model_name}" if model_name else ""
        
        return f"workflow:{workflow_type}:{url_hash}{model_suffix}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached workflow result.
        
        Args:
            cache_key: Cache key to retrieve
            
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
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache workflow result.
        
        Args:
            cache_key: Cache key for the result
            result: Result data to cache
            ttl: Time to live in seconds
            
        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            ttl = ttl or self.cache_ttl
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
            return False
    
    async def cleanup_expired_workflows(self) -> int:
        """
        Clean up expired workflow data.
        
        Returns:
            Number of cleaned up workflows
        """
        try:
            # Remove old workflow executions (older than 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            stmt = select(AIWorkflowExecution).where(
                and_(
                    AIWorkflowExecution.created_at < cutoff_date,
                    AIWorkflowExecution.status.in_([WorkflowStatus.COMPLETED, WorkflowStatus.FAILED])
                )
            )
            
            result = await self.db.execute(stmt)
            old_executions = result.scalars().all()
            
            # Delete old executions
            for execution in old_executions:
                await self.db.delete(execution)
            
            await self.db.commit()
            
            logger.info(f"Cleaned up {len(old_executions)} expired workflow executions")
            return len(old_executions)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired workflows: {e}")
            return 0
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get workflow performance metrics.
        
        Returns:
            Dictionary with workflow metrics
        """
        try:
            # Get workflow execution statistics
            stmt = select(
                AIWorkflowExecution.workflow_type,
                AIWorkflowExecution.status,
                func.count(AIWorkflowExecution.id).label('count'),
                func.avg(AIWorkflowExecution.execution_time).label('avg_time')
            ).group_by(AIWorkflowExecution.workflow_type, AIWorkflowExecution.status)
            
            result = await self.db.execute(stmt)
            stats = result.all()
            
            # Process statistics
            metrics = {
                "total_workflows": 0,
                "completed_workflows": 0,
                "failed_workflows": 0,
                "average_execution_time": 0.0,
                "workflow_types": {},
                "active_workflows": len(self.active_workflows)
            }
            
            total_time = 0.0
            total_count = 0
            
            for stat in stats:
                workflow_type = stat.workflow_type
                status = stat.status.value
                count = stat.count
                avg_time = stat.avg_time or 0.0
                
                metrics["total_workflows"] += count
                total_time += avg_time * count
                total_count += count
                
                if status == "completed":
                    metrics["completed_workflows"] += count
                elif status == "failed":
                    metrics["failed_workflows"] += count
                
                if workflow_type not in metrics["workflow_types"]:
                    metrics["workflow_types"][workflow_type] = {}
                
                metrics["workflow_types"][workflow_type][status] = {
                    "count": count,
                    "average_time": avg_time
                }
            
            if total_count > 0:
                metrics["average_execution_time"] = total_time / total_count
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get workflow metrics: {e}")
            return {
                "error": str(e),
                "total_workflows": 0,
                "active_workflows": len(self.active_workflows)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the orchestrator.
        
        Returns:
            Health check results
        """
        try:
            # Check database connection
            db_healthy = True
            try:
                await self.db.execute(select(1))
            except Exception:
                db_healthy = False
            
            # Check Redis connection
            redis_healthy = True
            try:
                await self.redis.ping()
            except Exception:
                redis_healthy = False
            
            # Get basic metrics
            metrics = await self.get_workflow_metrics()
            
            return {
                "status": "healthy" if db_healthy and redis_healthy else "unhealthy",
                "database": "connected" if db_healthy else "disconnected",
                "redis": "connected" if redis_healthy else "disconnected",
                "active_workflows": len(self.active_workflows),
                "total_workflows": metrics.get("total_workflows", 0),
                "average_execution_time": metrics.get("average_execution_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            } 