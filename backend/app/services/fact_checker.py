"""
Fact-checking service for TruthSeeQ platform.

This module provides comprehensive fact-checking capabilities including:
- Coordination between scraper and AI services
- Fact-checking workflow management
- User verification request handling
- Confidence score generation and explanations
- Integration with social feed system
- Batch processing and result aggregation

Classes:
    FactCheckerService: Main service class for fact-checking operations
    VerificationManager: Manages verification requests and workflows
    ResultAggregator: Aggregates and summarizes fact-checking results
    QualityAssessor: Assesses quality and reliability of fact-checking results
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from uuid import UUID, uuid4
from enum import Enum

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

# Import local modules
from ..config import settings
from ..database.models import (
    ContentItem, FactCheckResult, FactCheckSource, FeedPost,
    AIWorkflowExecution, WorkflowStatus, SourceType, VerdictType
)
from ..schemas.content import (
    FactCheckRequest, FactCheckResponse, FactCheckResult as FactCheckResultSchema,
    BatchVerificationRequest, BatchVerificationResponse,
    VerificationHistoryRequest, VerificationHistoryResponse
)
from .ai_service import AIService
from .scraper_service import ScraperService
from .feed_service import FeedService

logger = logging.getLogger(__name__)


class VerificationPriority(Enum):
    """Priority levels for verification requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class VerificationStatus(Enum):
    """Status of verification requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityAssessor:
    """
    Assesses quality and reliability of fact-checking results.
    
    Provides methods for:
    - Evaluating result quality based on multiple factors
    - Identifying potential issues or inconsistencies
    - Suggesting improvements or additional verification
    - Quality scoring and recommendations
    """
    
    @staticmethod
    def assess_result_quality(
        confidence_score: float,
        source_count: int,
        analysis_depth: str,
        evidence_strength: float,
        consistency_score: float
    ) -> Dict[str, Any]:
        """
        Assess the quality of a fact-checking result.
        
        Args:
            confidence_score: Overall confidence score (0-1)
            source_count: Number of sources used
            analysis_depth: Depth of analysis performed
            evidence_strength: Strength of supporting evidence (0-1)
            consistency_score: Consistency of analysis (0-1)
            
        Returns:
            Quality assessment results
        """
        quality_score = 0.0
        issues = []
        recommendations = []
        
        # Factor weights
        weights = {
            'confidence': 0.3,
            'sources': 0.2,
            'depth': 0.2,
            'evidence': 0.2,
            'consistency': 0.1
        }
        
        # Assess confidence score
        if confidence_score >= 0.8:
            quality_score += weights['confidence']
        elif confidence_score >= 0.6:
            quality_score += weights['confidence'] * 0.7
        else:
            quality_score += weights['confidence'] * 0.4
            issues.append("Low confidence score")
            recommendations.append("Consider additional verification sources")
        
        # Assess source count
        if source_count >= 5:
            quality_score += weights['sources']
        elif source_count >= 3:
            quality_score += weights['sources'] * 0.8
        elif source_count >= 1:
            quality_score += weights['sources'] * 0.5
            issues.append("Limited number of sources")
            recommendations.append("Add more verification sources")
        else:
            issues.append("No verification sources")
            recommendations.append("Critical: Add verification sources")
        
        # Assess analysis depth
        depth_scores = {
            'comprehensive': 1.0,
            'detailed': 0.8,
            'moderate': 0.6,
            'basic': 0.4,
            'minimal': 0.2
        }
        depth_score = depth_scores.get(analysis_depth, 0.3)
        quality_score += weights['depth'] * depth_score
        
        if depth_score < 0.6:
            issues.append("Limited analysis depth")
            recommendations.append("Perform more detailed analysis")
        
        # Assess evidence strength
        if evidence_strength >= 0.8:
            quality_score += weights['evidence']
        elif evidence_strength >= 0.6:
            quality_score += weights['evidence'] * 0.7
        else:
            quality_score += weights['evidence'] * 0.4
            issues.append("Weak supporting evidence")
            recommendations.append("Strengthen evidence base")
        
        # Assess consistency
        if consistency_score >= 0.8:
            quality_score += weights['consistency']
        elif consistency_score >= 0.6:
            quality_score += weights['consistency'] * 0.7
        else:
            quality_score += weights['consistency'] * 0.4
            issues.append("Inconsistent analysis results")
            recommendations.append("Review analysis methodology")
        
        # Overall quality assessment
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "issues": issues,
            "recommendations": recommendations,
            "factor_scores": {
                "confidence": confidence_score,
                "sources": min(source_count / 5, 1.0),
                "depth": depth_score,
                "evidence": evidence_strength,
                "consistency": consistency_score
            }
        }


class ResultAggregator:
    """
    Aggregates and summarizes fact-checking results.
    
    Provides methods for:
    - Combining multiple verification results
    - Generating summary statistics
    - Identifying trends and patterns
    - Creating comprehensive reports
    """
    
    @staticmethod
    def aggregate_results(results: List[FactCheckResult]) -> Dict[str, Any]:
        """
        Aggregate multiple fact-checking results.
        
        Args:
            results: List of fact-checking results
            
        Returns:
            Aggregated results summary
        """
        if not results:
            return {
                "total_results": 0,
                "average_confidence": 0.0,
                "verdict_distribution": {},
                "quality_summary": {}
            }
        
        # Calculate basic statistics
        total_results = len(results)
        confidence_scores = [r.confidence_score for r in results]
        average_confidence = sum(confidence_scores) / total_results
        
        # Verdict distribution
        verdict_counts = {}
        for result in results:
            verdict = result.verdict.value if result.verdict else "unknown"
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Quality assessment
        quality_scores = []
        for result in results:
            quality = QualityAssessor.assess_result_quality(
                confidence_score=result.confidence_score,
                source_count=len(result.sources) if result.sources else 0,
                analysis_depth="moderate",  # Would come from analysis metadata
                evidence_strength=0.7,  # Would be calculated from sources
                consistency_score=0.8  # Would be calculated from analysis
            )
            quality_scores.append(quality["quality_score"])
        
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "total_results": total_results,
            "average_confidence": average_confidence,
            "verdict_distribution": verdict_counts,
            "quality_summary": {
                "average_quality": average_quality,
                "quality_level": QualityAssessor._get_quality_level(average_quality)
            },
            "confidence_range": {
                "min": min(confidence_scores),
                "max": max(confidence_scores),
                "median": sorted(confidence_scores)[len(confidence_scores) // 2]
            }
        }
    
    @staticmethod
    def generate_summary_report(
        results: List[FactCheckResult],
        time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            results: List of fact-checking results
            time_period: Optional time period for the report
            
        Returns:
            Summary report
        """
        aggregated = ResultAggregator.aggregate_results(results)
        
        # Trend analysis
        if len(results) > 1:
            # Sort by creation time
            sorted_results = sorted(results, key=lambda x: x.created_at)
            
            # Calculate trend
            recent_confidence = [r.confidence_score for r in sorted_results[-5:]]
            older_confidence = [r.confidence_score for r in sorted_results[:5]]
            
            if len(recent_confidence) > 0 and len(older_confidence) > 0:
                recent_avg = sum(recent_confidence) / len(recent_confidence)
                older_avg = sum(older_confidence) / len(older_confidence)
                trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
            else:
                trend = "insufficient_data"
        else:
            trend = "insufficient_data"
        
        return {
            "report_period": time_period or "all_time",
            "summary": aggregated,
            "trend_analysis": {
                "trend": trend,
                "total_verifications": aggregated["total_results"]
            },
            "recommendations": ResultAggregator._generate_recommendations(aggregated)
        }
    
    @staticmethod
    def _generate_recommendations(aggregated: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on aggregated results."""
        recommendations = []
        
        if aggregated["average_confidence"] < 0.6:
            recommendations.append("Consider improving verification methodology")
        
        if aggregated["quality_summary"]["average_quality"] < 0.6:
            recommendations.append("Enhance quality control processes")
        
        verdict_dist = aggregated["verdict_distribution"]
        if verdict_dist.get("unverifiable", 0) > aggregated["total_results"] * 0.3:
            recommendations.append("High rate of unverifiable content - review source selection")
        
        return recommendations
    
    @staticmethod
    def _get_quality_level(score: float) -> str:
        """Get quality level from score."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"


class VerificationManager:
    """
    Manages verification requests and workflows.
    
    Provides methods for:
    - Request queue management
    - Priority-based processing
    - Workflow coordination
    - Status tracking and updates
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize verification manager.
        
        Args:
            redis_client: Redis client for queue management
        """
        self.redis = redis_client
        self.queue_prefix = "verification_queue"
        self.status_prefix = "verification_status"
    
    async def add_verification_request(
        self,
        request_id: str,
        content_id: int,
        priority: VerificationPriority = VerificationPriority.NORMAL,
        session_id: Optional[UUID] = None
    ) -> bool:
        """
        Add verification request to queue.
        
        Args:
            request_id: Unique request identifier
            content_id: Content ID to verify
            priority: Request priority level
            session_id: Optional user session ID
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Add to priority queue
            queue_key = f"{self.queue_prefix}:{priority.value}"
            request_data = {
                "request_id": request_id,
                "content_id": content_id,
                "session_id": str(session_id) if session_id else None,
                "created_at": datetime.utcnow().isoformat(),
                "status": VerificationStatus.PENDING.value
            }
            
            # Use priority score for queue ordering
            priority_scores = {
                VerificationPriority.LOW: 1,
                VerificationPriority.NORMAL: 2,
                VerificationPriority.HIGH: 3,
                VerificationPriority.URGENT: 4
            }
            
            score = priority_scores[priority]
            await self.redis.zadd(queue_key, {str(request_data): score})
            
            # Set status
            status_key = f"{self.status_prefix}:{request_id}"
            await self.redis.setex(status_key, 3600, VerificationStatus.PENDING.value)
            
            logger.info(f"Added verification request {request_id} to queue with priority {priority.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add verification request: {e}")
            return False
    
    async def get_next_request(self, priority: Optional[VerificationPriority] = None) -> Optional[Dict[str, Any]]:
        """
        Get next verification request from queue.
        
        Args:
            priority: Optional priority filter
            
        Returns:
            Next request data or None if queue is empty
        """
        try:
            if priority:
                queue_key = f"{self.queue_prefix}:{priority.value}"
                result = await self.redis.zpopmax(queue_key)
                if result:
                    return eval(result[0][0])  # Convert string back to dict
            else:
                # Check all priority queues in order
                for p in [VerificationPriority.URGENT, VerificationPriority.HIGH, 
                         VerificationPriority.NORMAL, VerificationPriority.LOW]:
                    queue_key = f"{self.queue_prefix}:{p.value}"
                    result = await self.redis.zpopmax(queue_key)
                    if result:
                        return eval(result[0][0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next request: {e}")
            return None
    
    async def update_request_status(
        self,
        request_id: str,
        status: VerificationStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update verification request status.
        
        Args:
            request_id: Request identifier
            status: New status
            metadata: Optional status metadata
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            status_key = f"{self.status_prefix}:{request_id}"
            status_data = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if metadata:
                status_data.update(metadata)
            
            await self.redis.setex(status_key, 3600, str(status_data))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update request status: {e}")
            return False
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get verification request status.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Status information or None if not found
        """
        try:
            status_key = f"{self.status_prefix}:{request_id}"
            status_data = await self.redis.get(status_key)
            
            if status_data:
                return eval(status_data)  # Convert string back to dict
            return None
            
        except Exception as e:
            logger.error(f"Failed to get request status: {e}")
            return None


class FactCheckerService:
    """
    Main fact-checking service integrating with TruthSeeQ platform.
    
    Provides comprehensive fact-checking capabilities including:
    - Coordination between scraper and AI services
    - Fact-checking workflow management
    - User verification request handling
    - Confidence score generation and explanations
    - Integration with social feed system
    - Batch processing and result aggregation
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        ai_service: AIService,
        scraper_service: ScraperService,
        feed_service: Optional[FeedService] = None
    ):
        """
        Initialize fact-checking service.
        
        Args:
            db_session: Database session for data storage
            redis_client: Redis client for caching and queues
            ai_service: AI service for analysis
            scraper_service: Scraper service for content retrieval
            feed_service: Optional feed service for social features
        """
        self.db = db_session
        self.redis = redis_client
        self.ai_service = ai_service
        self.scraper_service = scraper_service
        self.feed_service = feed_service
        
        # Initialize components
        self.verification_manager = VerificationManager(redis_client)
        self.result_aggregator = ResultAggregator()
        self.quality_assessor = QualityAssessor()
        
        logger.info("FactCheckerService initialized")
    
    async def verify_content(
        self,
        request: FactCheckRequest,
        session_id: Optional[UUID] = None,
        priority: VerificationPriority = VerificationPriority.NORMAL
    ) -> FactCheckResponse:
        """
        Verify content using comprehensive fact-checking workflow.
        
        Args:
            request: Fact-checking request
            session_id: Optional user session ID
            priority: Request priority level
            
        Returns:
            FactCheckResponse with verification results
        """
        start_time = time.time()
        request_id = str(uuid4())
        
        # Add to verification queue
        await self.verification_manager.add_verification_request(
            request_id, request.content_id, priority, session_id
        )
        
        # Update status to in progress
        await self.verification_manager.update_request_status(
            request_id, VerificationStatus.IN_PROGRESS
        )
        
        try:
            # Get content from database
            content_item = await self._get_content_item(request.content_id)
            if not content_item:
                raise ValueError(f"Content item {request.content_id} not found")
            
            # Check if content needs to be scraped/updated
            if self._needs_content_update(content_item):
                logger.info(f"Updating content for item {request.content_id}")
                await self.scraper_service.update_content(request.content_id)
                # Refresh content item
                content_item = await self._get_content_item(request.content_id)
            
            # Perform AI-based fact-checking
            ai_response = await self.ai_service.fact_check_content(request, session_id)
            
            # Enhance with additional verification sources
            enhanced_result = await self._enhance_with_sources(ai_response, content_item)
            
            # Generate quality assessment
            quality_assessment = self.quality_assessor.assess_result_quality(
                confidence_score=enhanced_result.confidence_score,
                source_count=len(enhanced_result.sources),
                analysis_depth="comprehensive",
                evidence_strength=self._calculate_evidence_strength(enhanced_result.sources),
                consistency_score=0.9  # Would be calculated from analysis consistency
            )
            
            # Update status to completed
            await self.verification_manager.update_request_status(
                request_id, VerificationStatus.COMPLETED, {
                    "execution_time": time.time() - start_time,
                    "quality_score": quality_assessment["quality_score"]
                }
            )
            
            # Create social feed post if feed service is available
            if self.feed_service and enhanced_result.confidence_score > 0.6:
                await self._create_feed_post(enhanced_result, content_item, session_id)
            
            # Add quality assessment to response
            response_data = enhanced_result.dict()
            response_data["quality_assessment"] = quality_assessment
            response_data["request_id"] = request_id
            
            return FactCheckResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Verification failed for content {request.content_id}: {e}")
            
            # Update status to failed
            await self.verification_manager.update_request_status(
                request_id, VerificationStatus.FAILED, {
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            )
            
            raise
    
    async def batch_verify_content(
        self,
        request: BatchVerificationRequest,
        session_id: Optional[UUID] = None
    ) -> BatchVerificationResponse:
        """
        Perform batch verification of multiple content items.
        
        Args:
            request: Batch verification request
            session_id: Optional user session ID
            
        Returns:
            BatchVerificationResponse with results
        """
        results = []
        failed_items = []
        
        # Process items concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent verifications
        
        async def verify_single_item(content_id: int) -> Tuple[int, Optional[FactCheckResponse]]:
            async with semaphore:
                try:
                    single_request = FactCheckRequest(
                        content_id=content_id,
                        model_name=request.model_name,
                        force_refresh=request.force_refresh
                    )
                    result = await self.verify_content(single_request, session_id)
                    return content_id, result
                except Exception as e:
                    logger.error(f"Batch verification failed for content {content_id}: {e}")
                    return content_id, None
        
        # Create verification tasks
        tasks = [verify_single_item(content_id) for content_id in request.content_ids]
        
        # Execute all verifications
        verification_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in verification_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            content_id, verification_result = result
            if verification_result:
                results.append(verification_result)
            else:
                failed_items.append(content_id)
        
        # Generate summary
        summary = self.result_aggregator.aggregate_results([
            FactCheckResult(
                confidence_score=r.confidence_score,
                verdict=r.verdict,
                reasoning=r.reasoning
            ) for r in results
        ])
        
        return BatchVerificationResponse(
            results=results,
            failed_items=failed_items,
            summary=summary,
            total_processed=len(request.content_ids),
            successful_count=len(results)
        )
    
    async def get_verification_history(
        self,
        request: VerificationHistoryRequest,
        session_id: Optional[UUID] = None
    ) -> VerificationHistoryResponse:
        """
        Get verification history for a session or content item.
        
        Args:
            request: History request parameters
            session_id: Optional user session ID
            
        Returns:
            VerificationHistoryResponse with history data
        """
        # Build query
        query = select(FactCheckResult).options(
            selectinload(FactCheckResult.sources)
        )
        
        # Add filters
        if request.content_id:
            query = query.where(FactCheckResult.content_id == request.content_id)
        
        if session_id:
            query = query.where(FactCheckResult.session_id == session_id)
        
        if request.start_date:
            query = query.where(FactCheckResult.created_at >= request.start_date)
        
        if request.end_date:
            query = query.where(FactCheckResult.created_at <= request.end_date)
        
        # Add ordering
        query = query.order_by(desc(FactCheckResult.created_at))
        
        # Add pagination
        if request.limit:
            query = query.limit(request.limit)
        
        if request.offset:
            query = query.offset(request.offset)
        
        # Execute query
        result = await self.db.execute(query)
        history_items = result.scalars().all()
        
        # Convert to schema objects
        history_results = []
        for item in history_items:
            history_results.append(FactCheckResultSchema(
                id=item.id,
                content_id=item.content_id,
                confidence_score=item.confidence_score,
                verdict=item.verdict.value if item.verdict else "unknown",
                reasoning=item.reasoning,
                created_at=item.created_at,
                sources=[{
                    "url": source.url,
                    "type": source.source_type.value if source.source_type else "unknown",
                    "relevance_score": source.relevance_score
                } for source in item.sources] if item.sources else []
            ))
        
        # Generate summary
        summary = self.result_aggregator.generate_summary_report(
            history_items,
            time_period=f"{request.start_date} to {request.end_date}" if request.start_date and request.end_date else None
        )
        
        return VerificationHistoryResponse(
            history=history_results,
            summary=summary,
            total_count=len(history_results)
        )
    
    async def get_verification_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get verification request status.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Status information or None if not found
        """
        return await self.verification_manager.get_request_status(request_id)
    
    async def _enhance_with_sources(
        self,
        ai_result: FactCheckResponse,
        content_item: ContentItem
    ) -> FactCheckResponse:
        """
        Enhance AI result with additional verification sources.
        
        Args:
            ai_result: AI fact-checking result
            content_item: Content item being verified
            
        Returns:
            Enhanced fact-checking result
        """
        # Add source verification
        source_verification = await self.ai_service.verify_source(content_item.url)
        
        # Add additional sources based on content analysis
        additional_sources = await self._find_additional_sources(content_item)
        
        # Combine sources
        all_sources = ai_result.sources + additional_sources
        
        # Recalculate confidence score with additional sources
        enhanced_confidence = self._recalculate_confidence(
            ai_result.confidence_score,
            source_verification.get("confidence", 0.5),
            len(all_sources)
        )
        
        # Update reasoning with source information
        enhanced_reasoning = f"{ai_result.reasoning}\n\nAdditional verification sources: {len(additional_sources)}"
        
        return FactCheckResponse(
            content_id=ai_result.content_id,
            confidence_score=enhanced_confidence,
            verdict=ai_result.verdict,
            reasoning=enhanced_reasoning,
            sources=all_sources,
            execution_time=ai_result.execution_time,
            workflow_execution_id=ai_result.workflow_execution_id
        )
    
    async def _find_additional_sources(self, content_item: ContentItem) -> List[Dict[str, Any]]:
        """
        Find additional verification sources for content.
        
        Args:
            content_item: Content item to find sources for
            
        Returns:
            List of additional sources
        """
        # This would integrate with external fact-checking databases
        # For now, return empty list
        return []
    
    def _calculate_evidence_strength(self, sources: List[Dict[str, Any]]) -> float:
        """
        Calculate evidence strength based on sources.
        
        Args:
            sources: List of verification sources
            
        Returns:
            Evidence strength score (0-1)
        """
        if not sources:
            return 0.0
        
        # Calculate based on source quality and quantity
        total_strength = 0.0
        for source in sources:
            source_type = source.get("type", "unknown")
            relevance = source.get("relevance_score", 0.5)
            
            # Weight by source type
            type_weights = {
                "fact_checking_database": 1.0,
                "reliable_news": 0.9,
                "academic": 0.8,
                "government": 0.9,
                "expert_opinion": 0.7,
                "social_media": 0.3,
                "unknown": 0.5
            }
            
            weight = type_weights.get(source_type, 0.5)
            total_strength += relevance * weight
        
        return min(total_strength / len(sources), 1.0)
    
    def _recalculate_confidence(
        self,
        base_confidence: float,
        source_confidence: float,
        source_count: int
    ) -> float:
        """
        Recalculate confidence score with additional factors.
        
        Args:
            base_confidence: Base confidence score
            source_confidence: Source verification confidence
            source_count: Number of sources
            
        Returns:
            Recalculated confidence score
        """
        # Weight factors
        base_weight = 0.6
        source_weight = 0.3
        count_weight = 0.1
        
        # Source count bonus (diminishing returns)
        count_bonus = min(source_count / 10, 0.2)
        
        confidence = (
            base_confidence * base_weight +
            source_confidence * source_weight +
            count_bonus * count_weight
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _needs_content_update(self, content_item: ContentItem) -> bool:
        """
        Check if content needs to be updated.
        
        Args:
            content_item: Content item to check
            
        Returns:
            True if update is needed, False otherwise
        """
        if not content_item.scraped_at:
            return True
        
        # Check if content is older than 24 hours
        update_threshold = datetime.utcnow() - timedelta(hours=24)
        return content_item.scraped_at < update_threshold
    
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
    
    async def _create_feed_post(
        self,
        verification_result: FactCheckResponse,
        content_item: ContentItem,
        session_id: Optional[UUID] = None
    ) -> Optional[FeedPost]:
        """
        Create social feed post for verification result.
        
        Args:
            verification_result: Verification result
            content_item: Content item
            session_id: Optional user session ID
            
        Returns:
            Created feed post or None if failed
        """
        if not self.feed_service:
            return None
        
        try:
            # Create post title and summary
            title = f"Fact-check: {content_item.title or 'Content Verification'}"
            
            verdict_emoji = {
                "true": "✅",
                "mostly_true": "✅",
                "partially_true": "⚠️",
                "mostly_false": "❌",
                "false": "❌",
                "unverifiable": "❓"
            }.get(verification_result.verdict, "❓")
            
            summary = (
                f"{verdict_emoji} {verification_result.verdict.replace('_', ' ').title()}\n"
                f"Confidence: {verification_result.confidence_score:.1%}\n"
                f"Source: {content_item.url}"
            )
            
            # Create feed post
            post = await self.feed_service.create_post(
                content_id=content_item.id,
                session_id=session_id,
                title=title,
                summary=summary,
                verdict=verification_result.verdict,
                confidence=verification_result.confidence_score
            )
            
            return post
            
        except Exception as e:
            logger.error(f"Failed to create feed post: {e}")
            return None
