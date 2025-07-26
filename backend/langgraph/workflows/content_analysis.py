"""
Content analysis workflow for TruthSeeQ platform.

This module implements a content analysis workflow using LangGraph,
providing content quality assessment, sentiment analysis, bias detection,
and content categorization capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..nodes.content_extraction import ContentExtractionNode
from ..tools.content_analysis import (
    SentimentAnalysisTool, 
    BiasDetectionTool, 
    ContentQualityTool,
    ClaimExtractionTool
)

logger = logging.getLogger(__name__)


class ContentAnalysisWorkflow:
    """
    Content analysis workflow using LangGraph.
    
    This workflow provides comprehensive content analysis including:
    1. Content Quality Assessment - Evaluate content structure and readability
    2. Sentiment Analysis - Analyze emotional tone and sentiment
    3. Bias Detection - Identify various types of bias
    4. Content Categorization - Categorize content by type and topic
    """
    
    def __init__(self):
        """Initialize the content analysis workflow."""
        self.content_extraction_node = ContentExtractionNode()
        self.sentiment_analyzer = SentimentAnalysisTool()
        self.bias_detector = BiasDetectionTool()
        self.quality_assessor = ContentQualityTool()
        self.claim_extractor = ClaimExtractionTool()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow.
        
        Returns:
            Configured StateGraph for content analysis workflow
        """
        # Create the workflow graph
        workflow = StateGraph(StateType=Dict[str, Any])
        
        # Add nodes to the workflow
        workflow.add_node("extract_content", self._extract_content)
        workflow.add_node("assess_content_quality", self._assess_content_quality)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment)
        workflow.add_node("detect_bias", self._detect_bias)
        workflow.add_node("extract_claims", self._extract_claims)
        workflow.add_node("categorize_content", self._categorize_content)
        workflow.add_node("aggregate_analysis", self._aggregate_analysis)
        workflow.add_node("generate_report", self._generate_report)
        
        # Define the workflow flow
        workflow.set_entry_point("extract_content")
        
        # Parallel analysis branches
        workflow.add_edge("extract_content", "assess_content_quality")
        workflow.add_edge("extract_content", "analyze_sentiment")
        workflow.add_edge("extract_content", "detect_bias")
        workflow.add_edge("extract_content", "extract_claims")
        
        # Aggregation and reporting
        workflow.add_edge("assess_content_quality", "categorize_content")
        workflow.add_edge("analyze_sentiment", "categorize_content")
        workflow.add_edge("detect_bias", "categorize_content")
        workflow.add_edge("extract_claims", "categorize_content")
        
        workflow.add_edge("categorize_content", "aggregate_analysis")
        workflow.add_edge("aggregate_analysis", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def run_workflow(self, content_id: int, content_text: str = "", content_url: str = "") -> Dict[str, Any]:
        """
        Run the complete content analysis workflow.
        
        Args:
            content_id: ID of the content to analyze
            content_text: Text content to analyze (optional if content_url provided)
            content_url: URL of the content (optional if content_text provided)
            
        Returns:
            Complete content analysis results
        """
        try:
            # Prepare initial state
            initial_state = {
                "content_id": content_id,
                "content_text": content_text,
                "content_url": content_url,
                "workflow_status": "started"
            }
            
            logger.info(f"Starting content analysis workflow for content_id: {content_id}")
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"content_analysis_{content_id}"}}
            result = self.workflow.invoke(initial_state, config)
            
            logger.info(f"Content analysis workflow completed for content_id: {content_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running content analysis workflow: {str(e)}")
            return {
                "content_id": content_id,
                "workflow_status": "failed",
                "error": str(e)
            }
    
    # Node execution methods
    def _extract_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content extraction node."""
        return self.content_extraction_node.extract_content(state)
    
    def _assess_content_quality(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content quality assessment."""
        try:
            content_text = state.get("content_text", "")
            if not content_text:
                return {
                    **state,
                    "content_quality": {},
                    "quality_assessment_status": "failed"
                }
            
            quality_result = self.quality_assessor._run(content_text)
            
            return {
                **state,
                "content_quality": quality_result,
                "quality_assessment_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in content quality assessment: {str(e)}")
            return {
                **state,
                "quality_assessment_status": "failed",
                "error": str(e)
            }
    
    def _analyze_sentiment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis."""
        try:
            content_text = state.get("content_text", "")
            if not content_text:
                return {
                    **state,
                    "sentiment_analysis": {},
                    "sentiment_analysis_status": "failed"
                }
            
            sentiment_result = self.sentiment_analyzer._run(content_text)
            
            return {
                **state,
                "sentiment_analysis": sentiment_result,
                "sentiment_analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                **state,
                "sentiment_analysis_status": "failed",
                "error": str(e)
            }
    
    def _detect_bias(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bias detection."""
        try:
            content_text = state.get("content_text", "")
            if not content_text:
                return {
                    **state,
                    "bias_analysis": {},
                    "bias_detection_status": "failed"
                }
            
            bias_result = self.bias_detector._run(content_text)
            
            return {
                **state,
                "bias_analysis": bias_result,
                "bias_detection_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in bias detection: {str(e)}")
            return {
                **state,
                "bias_detection_status": "failed",
                "error": str(e)
            }
    
    def _extract_claims(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute claim extraction."""
        try:
            content_text = state.get("content_text", "")
            if not content_text:
                return {
                    **state,
                    "claim_analysis": {},
                    "claim_extraction_status": "failed"
                }
            
            claims_result = self.claim_extractor._run(content_text)
            
            return {
                **state,
                "claim_analysis": claims_result,
                "claim_extraction_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in claim extraction: {str(e)}")
            return {
                **state,
                "claim_extraction_status": "failed",
                "error": str(e)
            }
    
    def _categorize_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content categorization."""
        try:
            content_text = state.get("content_text", "")
            sentiment_analysis = state.get("sentiment_analysis", {})
            bias_analysis = state.get("bias_analysis", {})
            claim_analysis = state.get("claim_analysis", {})
            
            # Determine content type
            content_type = self._determine_content_type(
                content_text, sentiment_analysis, bias_analysis, claim_analysis
            )
            
            # Determine topic categories
            topic_categories = self._determine_topic_categories(content_text)
            
            # Determine complexity level
            complexity_level = self._determine_complexity_level(
                state.get("content_quality", {})
            )
            
            categorization = {
                "content_type": content_type,
                "topic_categories": topic_categories,
                "complexity_level": complexity_level,
                "analysis_confidence": 0.8  # Placeholder
            }
            
            return {
                **state,
                "content_categorization": categorization,
                "categorization_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in content categorization: {str(e)}")
            return {
                **state,
                "categorization_status": "failed",
                "error": str(e)
            }
    
    def _aggregate_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis aggregation."""
        try:
            content_quality = state.get("content_quality", {})
            sentiment_analysis = state.get("sentiment_analysis", {})
            bias_analysis = state.get("bias_analysis", {})
            claim_analysis = state.get("claim_analysis", {})
            content_categorization = state.get("content_categorization", {})
            
            # Calculate overall content score
            overall_score = self._calculate_overall_content_score(
                content_quality, sentiment_analysis, bias_analysis, claim_analysis
            )
            
            # Generate content insights
            insights = self._generate_content_insights(
                content_quality, sentiment_analysis, bias_analysis, claim_analysis
            )
            
            aggregated_analysis = {
                "overall_score": overall_score,
                "insights": insights,
                "analysis_summary": {
                    "quality_level": content_quality.get("quality_level", "unknown"),
                    "sentiment": sentiment_analysis.get("sentiment", "neutral"),
                    "bias_level": bias_analysis.get("bias_level", "unknown"),
                    "claim_count": claim_analysis.get("total_claims", 0),
                    "content_type": content_categorization.get("content_type", "unknown")
                }
            }
            
            return {
                **state,
                "aggregated_analysis": aggregated_analysis,
                "aggregation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in analysis aggregation: {str(e)}")
            return {
                **state,
                "aggregation_status": "failed",
                "error": str(e)
            }
    
    def _generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation."""
        try:
            content_id = state.get("content_id")
            aggregated_analysis = state.get("aggregated_analysis", {})
            content_quality = state.get("content_quality", {})
            sentiment_analysis = state.get("sentiment_analysis", {})
            bias_analysis = state.get("bias_analysis", {})
            claim_analysis = state.get("claim_analysis", {})
            content_categorization = state.get("content_categorization", {})
            
            # Generate comprehensive report
            report = {
                "content_id": content_id,
                "timestamp": "2024-01-01T00:00:00Z",  # Would be dynamic
                "overall_assessment": aggregated_analysis.get("overall_score", 0),
                "content_quality": content_quality,
                "sentiment_analysis": sentiment_analysis,
                "bias_analysis": bias_analysis,
                "claim_analysis": claim_analysis,
                "content_categorization": content_categorization,
                "insights": aggregated_analysis.get("insights", []),
                "recommendations": self._generate_recommendations(state),
                "analysis_metadata": {
                    "workflow_version": "1.0",
                    "analysis_completed": True
                }
            }
            
            return {
                **state,
                "final_report": report,
                "report_generation_status": "completed",
                "workflow_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {
                **state,
                "report_generation_status": "failed",
                "error": str(e)
            }
    
    # Helper methods
    def _determine_content_type(self, content_text: str, sentiment_analysis: Dict, bias_analysis: Dict, claim_analysis: Dict) -> str:
        """Determine the type of content."""
        # Simple heuristics for content type determination
        claim_count = claim_analysis.get("total_claims", 0)
        bias_score = bias_analysis.get("overall_bias_score", 0)
        sentiment = sentiment_analysis.get("sentiment", "neutral")
        
        if claim_count > 5:
            return "factual_article"
        elif bias_score > 0.1:
            return "opinion_piece"
        elif sentiment in ["positive", "negative"]:
            return "emotional_content"
        else:
            return "general_content"
    
    def _determine_topic_categories(self, content_text: str) -> List[str]:
        """Determine topic categories for the content."""
        # Simple keyword-based topic detection
        topics = []
        content_lower = content_text.lower()
        
        # Define topic keywords
        topic_keywords = {
            "politics": ["politics", "government", "election", "policy", "democrat", "republican"],
            "health": ["health", "medical", "disease", "treatment", "vaccine", "doctor"],
            "technology": ["technology", "tech", "software", "computer", "digital", "ai"],
            "science": ["science", "research", "study", "scientific", "experiment"],
            "business": ["business", "economy", "market", "finance", "company", "stock"],
            "sports": ["sports", "game", "team", "player", "championship", "league"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ["general"]
    
    def _determine_complexity_level(self, content_quality: Dict) -> str:
        """Determine the complexity level of the content."""
        readability_score = content_quality.get("readability_score", 0)
        avg_sentence_length = content_quality.get("avg_sentence_length", 0)
        
        if readability_score < 30 or avg_sentence_length > 25:
            return "complex"
        elif readability_score < 60 or avg_sentence_length > 15:
            return "moderate"
        else:
            return "simple"
    
    def _calculate_overall_content_score(self, content_quality: Dict, sentiment_analysis: Dict, bias_analysis: Dict, claim_analysis: Dict) -> float:
        """Calculate overall content score."""
        quality_score = content_quality.get("overall_quality_score", 0) / 100  # Normalize to 0-1
        bias_score = 1 - bias_analysis.get("overall_bias_score", 0)  # Invert bias score
        claim_density = min(claim_analysis.get("claim_density", 0) * 10, 1)  # Normalize claim density
        
        # Weighted combination
        overall_score = (
            quality_score * 0.4 +
            bias_score * 0.3 +
            claim_density * 0.3
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _generate_content_insights(self, content_quality: Dict, sentiment_analysis: Dict, bias_analysis: Dict, claim_analysis: Dict) -> List[str]:
        """Generate insights about the content."""
        insights = []
        
        # Quality insights
        quality_score = content_quality.get("overall_quality_score", 0)
        if quality_score < 50:
            insights.append("Content quality is below average")
        elif quality_score > 80:
            insights.append("Content has high quality")
        
        # Sentiment insights
        sentiment = sentiment_analysis.get("sentiment", "neutral")
        if sentiment != "neutral":
            insights.append(f"Content has {sentiment} sentiment")
        
        # Bias insights
        bias_level = bias_analysis.get("bias_level", "unknown")
        if bias_level != "low":
            insights.append(f"Content shows {bias_level} levels of bias")
        
        # Claim insights
        claim_count = claim_analysis.get("total_claims", 0)
        if claim_count > 0:
            insights.append(f"Content contains {claim_count} verifiable claims")
        
        return insights
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        content_quality = state.get("content_quality", {})
        bias_analysis = state.get("bias_analysis", {})
        
        # Quality recommendations
        quality_score = content_quality.get("overall_quality_score", 0)
        if quality_score < 50:
            recommendations.append("Consider improving content structure and readability")
        
        # Bias recommendations
        bias_level = bias_analysis.get("bias_level", "unknown")
        if bias_level in ["medium", "high"]:
            recommendations.append("Content shows bias - consider using more neutral language")
        
        return recommendations


# Factory function for creating workflow instances
def create_content_analysis_workflow() -> ContentAnalysisWorkflow:
    """
    Create a new content analysis workflow instance.
    
    Returns:
        Configured ContentAnalysisWorkflow instance
    """
    return ContentAnalysisWorkflow()
