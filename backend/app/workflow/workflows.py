"""
Main Workflow Implementations for TruthSeeQ

This module contains the main LangGraph workflow implementations for:
- Fact-checking workflow
- Content analysis workflow  
- Source verification workflow

Each workflow is implemented as a LangGraph StateGraph with defined nodes and edges.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import local modules
from .state import (
    FactCheckState, ContentAnalysisState, SourceVerificationState,
    create_fact_check_state, create_content_analysis_state, create_source_verification_state
)
from .nodes import (
    ContentExtractionNode, ClaimsExtractionNode, SourceVerificationNode,
    FactAnalysisNode, ConfidenceScoringNode, get_node_by_name
)
from .tools import get_workflow_tools

logger = logging.getLogger(__name__)


class FactCheckingWorkflow:
    """
    Main fact-checking workflow using LangGraph.
    
    This workflow implements a comprehensive fact-checking process:
    1. Content Extraction - Scrape and preprocess content
    2. Claims Extraction - Identify factual claims using AI
    3. Source Verification - Find and verify supporting sources
    4. Fact Analysis - Analyze factual accuracy of claims
    5. Confidence Scoring - Generate final verdict and confidence score
    """
    
    def __init__(self, ai_model_name: Optional[str] = None):
        """
        Initialize fact-checking workflow.
        
        Args:
            ai_model_name: AI model to use for analysis
        """
        self.ai_model_name = ai_model_name or "gpt-4.1-mini-2025-04-14"
        self.graph = self._create_workflow_graph()
        
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the fact-checking workflow graph.
        
        Returns:
            LangGraph StateGraph for fact-checking
        """
        # Define the state structure
        workflow = StateGraph(FactCheckState)
        
        # Create nodes
        content_extraction_node = ContentExtractionNode(self.ai_model_name)
        claims_extraction_node = ClaimsExtractionNode(self.ai_model_name)
        source_verification_node = SourceVerificationNode(self.ai_model_name)
        fact_analysis_node = FactAnalysisNode(self.ai_model_name)
        confidence_scoring_node = ConfidenceScoringNode(self.ai_model_name)
        
        # Add nodes to graph
        workflow.add_node("content_extraction", content_extraction_node)
        workflow.add_node("claims_extraction", claims_extraction_node)
        workflow.add_node("source_verification", source_verification_node)
        workflow.add_node("fact_analysis_node", fact_analysis_node)
        workflow.add_node("confidence_scoring", confidence_scoring_node)
        
        # Define entrypoint
        workflow.set_entry_point("content_extraction")
        
        # Add conditional edges for error handling and flow control
        workflow.add_conditional_edges(
            "content_extraction",
            self._should_continue_after_content_extraction,
            {
                "continue": "claims_extraction",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "claims_extraction", 
            self._should_continue_after_claims_extraction,
            {
                "continue": "source_verification",
                "skip_verification": "confidence_scoring",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "source_verification",
            self._should_continue_after_source_verification,
            {
                "continue": "fact_analysis_node",
                "skip_analysis": "confidence_scoring",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "fact_analysis_node",
            self._should_continue_after_fact_analysis,
            {
                "continue": "confidence_scoring",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "confidence_scoring",
            self._should_continue_after_confidence_scoring,
            {
                "continue": END,
                "error": END
            }
        )
        
        return workflow.compile()
    
    def _should_continue_after_content_extraction(self, state: Dict[str, Any]) -> str:
        """Determine next step after content extraction."""
        status = state.get("status", "")
        if status == "content_extracted":
            return "continue"
        elif status == "content_extraction_failed" or status == "error":
            return "error"
        else:
            # Default to continue for any other status
            return "continue"
    
    def _should_continue_after_claims_extraction(self, state: Dict[str, Any]) -> str:
        """Determine next step after claims extraction."""
        claims = state.get("extracted_claims", [])
        status = state.get("status", "")
        
        if status == "claims_extracted" and claims:
            return "continue"
        elif status in ["no_content", "model_unavailable", "parsing_failed"] or not claims:
            return "skip_verification"
        elif status == "claims_extraction_failed" or status == "error":
            return "error"
        else:
            # Default to continue for any other status
            return "continue"
    
    def _should_continue_after_source_verification(self, state: Dict[str, Any]) -> str:
        """Determine next step after source verification."""
        sources = state.get("verification_sources", [])
        status = state.get("status", "")
        
        if status == "sources_verified" and sources:
            return "continue"
        elif status == "no_claims" or not sources:
            return "skip_analysis"
        elif status == "source_verification_failed" or status == "error":
            return "error"
        else:
            # Default to continue for any other status
            return "continue"
    
    def _should_continue_after_fact_analysis(self, state: Dict[str, Any]) -> str:
        """Determine next step after fact analysis."""
        status = state.get("status", "")
        
        if status == "facts_analyzed":
            return "continue"
        elif status in ["model_unavailable", "no_claims", "no_sources", "parsing_failed"]:
            return "continue"  # Still continue to confidence scoring even with issues
        elif status == "fact_analysis_failed" or status == "error":
            return "error"
        else:
            # Default to continue for any other status
            return "continue"
    
    def _should_continue_after_confidence_scoring(self, state: Dict[str, Any]) -> str:
        """Determine next step after confidence scoring."""
        status = state.get("status", "")
        
        if status == "verdict_generated":
            return "continue"
        elif status == "no_claims":
            return "continue"  # Still end the workflow even with no claims
        elif status == "confidence_scoring_failed" or status == "error":
            return "error"
        else:
            # Default to continue for any other status
            return "continue"
    
    async def execute(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> FactCheckState:
        """
        Execute the fact-checking workflow.
        
        Args:
            url: URL to fact-check
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Completed FactCheckState
        """
        start_time = time.time()
        
        if not workflow_id:
            workflow_id = f"fact_check_{int(time.time())}"
        
        # Create initial state
        initial_state = create_fact_check_state(workflow_id, url, session_id)
        initial_state["status"] = "running"
        initial_state["model_used"] = self.ai_model_name
        
        logger.info(f"Starting fact-checking workflow for {url}")
        
        try:
            # Execute workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Update final state - only update status if not in error state
            result["processing_time"] = time.time() - start_time
            if result.get("status") not in ["error", "content_extraction_failed", "claims_extraction_failed", "source_verification_failed", "fact_analysis_failed", "confidence_scoring_failed"]:
                result["status"] = "completed"
            result["updated_at"] = datetime.utcnow()
            
            logger.info(f"Fact-checking workflow completed for {url} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Fact-checking workflow failed for {url}: {e}")
            
            # Return error state
            error_state = initial_state.copy()
            error_state["status"] = "error"
            error_state["error_message"] = str(e)
            error_state["processing_time"] = time.time() - start_time
            error_state["updated_at"] = datetime.utcnow()
            
            return error_state


class ContentAnalysisWorkflow:
    """
    Content analysis workflow using LangGraph.
    
    This workflow provides comprehensive content analysis including:
    1. Content Extraction - Scrape and preprocess content
    2. Sentiment Analysis - Analyze emotional tone and sentiment
    3. Bias Detection - Identify potential biases and political leanings
    4. Content Quality Assessment - Evaluate content structure and readability
    5. Source Credibility Analysis - Assess source reliability
    """
    
    def __init__(self, ai_model_name: Optional[str] = None):
        """
        Initialize content analysis workflow.
        
        Args:
            ai_model_name: AI model to use for analysis
        """
        self.ai_model_name = ai_model_name or "gpt-4.1-mini-2025-04-14"
        self.graph = self._create_workflow_graph()
    
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the content analysis workflow graph.
        
        Returns:
            LangGraph StateGraph for content analysis
        """
        # Define the state structure
        workflow = StateGraph(ContentAnalysisState)
        
        # Create nodes
        content_extraction_node = ContentExtractionNode(self.ai_model_name)
        sentiment_analysis_node = get_node_by_name("sentiment_analysis", self.ai_model_name)
        bias_detection_node = get_node_by_name("bias_detection", self.ai_model_name)
        quality_assessment_node = get_node_by_name("quality_assessment", self.ai_model_name)
        credibility_analysis_node = get_node_by_name("credibility_analysis", self.ai_model_name)
        summary_generation_node = get_node_by_name("summary_generation", self.ai_model_name)
        
        # Add nodes to graph
        workflow.add_node("content_extraction", content_extraction_node)
        workflow.add_node("sentiment_analysis_node", sentiment_analysis_node)
        workflow.add_node("bias_detection_node", bias_detection_node)
        workflow.add_node("quality_assessment_node", quality_assessment_node)
        workflow.add_node("credibility_analysis_node", credibility_analysis_node)
        workflow.add_node("summary_generation", summary_generation_node)
        
        # Define entrypoint
        workflow.set_entry_point("content_extraction")
        
        # Define edges
        workflow.add_edge("content_extraction", "sentiment_analysis_node")
        workflow.add_edge("sentiment_analysis_node", "bias_detection_node")
        workflow.add_edge("bias_detection_node", "quality_assessment_node")
        workflow.add_edge("quality_assessment_node", "credibility_analysis_node")
        workflow.add_edge("credibility_analysis_node", "summary_generation")
        workflow.add_edge("summary_generation", END)
        
        return workflow.compile()
    
    async def execute(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> ContentAnalysisState:
        """
        Execute the content analysis workflow.
        
        Args:
            url: URL to analyze
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Completed ContentAnalysisState
        """
        start_time = time.time()
        
        if not workflow_id:
            workflow_id = f"content_analysis_{int(time.time())}"
        
        # Create initial state
        initial_state = create_content_analysis_state(workflow_id, url, session_id)
        initial_state["status"] = "running"
        
        logger.info(f"Starting content analysis workflow for {url}")
        
        try:
            # Execute workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Update final state - only update status if not in error state
            result["processing_time"] = time.time() - start_time
            if result.get("status") not in ["error", "content_extraction_failed"]:
                result["status"] = "completed"
            result["updated_at"] = datetime.utcnow()
            
            logger.info(f"Content analysis workflow completed for {url} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Content analysis workflow failed for {url}: {e}")
            
            # Return error state
            error_state = initial_state.copy()
            error_state["status"] = "error"
            error_state["error_message"] = str(e)
            error_state["processing_time"] = time.time() - start_time
            error_state["updated_at"] = datetime.utcnow()
            
            return error_state


class SourceVerificationWorkflow:
    """
    Source verification workflow using LangGraph.
    
    This workflow provides comprehensive source verification including:
    1. Domain Analysis - Analyze domain characteristics and age
    2. Reputation Check - Check domain reputation and trust indicators
    3. Fact-Checking Database Lookup - Search known fact-checking databases
    4. Cross-Reference Analysis - Compare with other reliable sources
    5. Verification Result Generation - Generate final verification assessment
    """
    
    def __init__(self, ai_model_name: Optional[str] = None):
        """
        Initialize source verification workflow.
        
        Args:
            ai_model_name: AI model to use for analysis
        """
        self.ai_model_name = ai_model_name or "gpt-4.1-mini-2025-04-14"
        self.graph = self._create_workflow_graph()
    
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the source verification workflow graph.
        
        Returns:
            LangGraph StateGraph for source verification
        """
        # Define the state structure
        workflow = StateGraph(SourceVerificationState)
        
        # Create nodes
        domain_analysis_node = get_node_by_name("domain_analysis", self.ai_model_name)
        reputation_check_node = get_node_by_name("reputation_check", self.ai_model_name)
        fact_checking_lookup_node = get_node_by_name("fact_checking_lookup", self.ai_model_name)
        cross_reference_node = get_node_by_name("cross_reference", self.ai_model_name)
        verification_result_node = get_node_by_name("verification_result", self.ai_model_name)
        
        # Add nodes to graph
        workflow.add_node("domain_analysis_node", domain_analysis_node)
        workflow.add_node("reputation_check", reputation_check_node)
        workflow.add_node("fact_checking_lookup", fact_checking_lookup_node)
        workflow.add_node("cross_reference", cross_reference_node)
        workflow.add_node("verification_result_node", verification_result_node)
        
        # Define entrypoint
        workflow.set_entry_point("domain_analysis_node")
        
        # Define edges
        workflow.add_edge("domain_analysis_node", "reputation_check")
        workflow.add_edge("reputation_check", "fact_checking_lookup")
        workflow.add_edge("fact_checking_lookup", "cross_reference")
        workflow.add_edge("cross_reference", "verification_result_node")
        workflow.add_edge("verification_result_node", END)
        
        return workflow.compile()
    
    async def execute(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> SourceVerificationState:
        """
        Execute the source verification workflow.
        
        Args:
            url: URL to verify
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Completed SourceVerificationState
        """
        start_time = time.time()
        
        if not workflow_id:
            workflow_id = f"source_verification_{int(time.time())}"
        
        # Create initial state
        initial_state = create_source_verification_state(workflow_id, url, session_id)
        initial_state["status"] = "running"
        
        logger.info(f"Starting source verification workflow for {url}")
        
        try:
            # Execute workflow
            result = await self.graph.ainvoke(initial_state)
            
            # Update final state - only update status if not in error state
            result["processing_time"] = time.time() - start_time
            if result.get("status") not in ["error", "content_extraction_failed"]:
                result["status"] = "completed"
            result["updated_at"] = datetime.utcnow()
            
            logger.info(f"Source verification workflow completed for {url} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Source verification workflow failed for {url}: {e}")
            
            # Return error state
            error_state = initial_state.copy()
            error_state["status"] = "error"
            error_state["error_message"] = str(e)
            error_state["processing_time"] = time.time() - start_time
            error_state["updated_at"] = datetime.utcnow()
            
            return error_state


class WorkflowManager:
    """
    Manager for all TruthSeeQ workflows.
    
    This class provides a unified interface for executing different types
    of workflows and managing their lifecycle.
    """
    
    def __init__(self, default_model: str = "gpt-4.1-mini-2025-04-14"):
        """
        Initialize workflow manager.
        
        Args:
            default_model: Default AI model to use
        """
        self.default_model = default_model
        self.workflows = {
            "fact_checking": FactCheckingWorkflow(default_model),
            "content_analysis": ContentAnalysisWorkflow(default_model),
            "source_verification": SourceVerificationWorkflow(default_model)
        }
        
        logger.info(f"WorkflowManager initialized with {len(self.workflows)} workflows")
    
    async def execute_fact_checking(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> FactCheckState:
        """
        Execute fact-checking workflow.
        
        Args:
            url: URL to fact-check
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Fact-checking results
        """
        return await self.workflows["fact_checking"].execute(url, session_id, workflow_id)
    
    async def execute_content_analysis(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> ContentAnalysisState:
        """
        Execute content analysis workflow.
        
        Args:
            url: URL to analyze
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Content analysis results
        """
        return await self.workflows["content_analysis"].execute(url, session_id, workflow_id)
    
    async def execute_source_verification(
        self, 
        url: str, 
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> SourceVerificationState:
        """
        Execute source verification workflow.
        
        Args:
            url: URL to verify
            session_id: Optional user session ID
            workflow_id: Optional workflow ID
            
        Returns:
            Source verification results
        """
        return await self.workflows["source_verification"].execute(url, session_id, workflow_id)
    
    async def execute_comprehensive_analysis(
        self, 
        url: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all workflows for comprehensive analysis.
        
        Args:
            url: URL to analyze
            session_id: Optional user session ID
            
        Returns:
            Comprehensive analysis results
        """
        workflow_id = f"comprehensive_{int(time.time())}"
        
        logger.info(f"Starting comprehensive analysis for {url}")
        
        # Execute all workflows in parallel
        tasks = [
            self.execute_fact_checking(url, session_id, f"{workflow_id}_fact_check"),
            self.execute_content_analysis(url, session_id, f"{workflow_id}_content"),
            self.execute_source_verification(url, session_id, f"{workflow_id}_source")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        comprehensive_result = {
            "workflow_id": workflow_id,
            "url": url,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "fact_checking": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "content_analysis": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "source_verification": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
        }
        
        logger.info(f"Comprehensive analysis completed for {url}")
        return comprehensive_result
    
    def get_available_workflows(self) -> List[str]:
        """
        Get list of available workflow types.
        
        Returns:
            List of workflow type names
        """
        return list(self.workflows.keys())
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific workflow.
        
        Args:
            workflow_id: Workflow ID to check
            
        Returns:
            Workflow status information or None if not found
        """
        # This would typically query a database or cache for workflow status
        # For now, return a placeholder
        return {
            "workflow_id": workflow_id,
            "status": "unknown",
            "message": "Workflow status tracking not implemented"
        }


# Convenience functions for easy workflow execution
async def fact_check_url(
    url: str, 
    ai_model_name: Optional[str] = None,
    session_id: Optional[str] = None
) -> FactCheckState:
    """
    Convenience function to fact-check a URL.
    
    Args:
        url: URL to fact-check
        ai_model_name: Optional AI model name
        session_id: Optional user session ID
        
    Returns:
        Fact-checking results
    """
    workflow = FactCheckingWorkflow(ai_model_name)
    return await workflow.execute(url, session_id)


async def analyze_content(
    url: str, 
    ai_model_name: Optional[str] = None,
    session_id: Optional[str] = None
) -> ContentAnalysisState:
    """
    Convenience function to analyze content from a URL.
    
    Args:
        url: URL to analyze
        ai_model_name: Optional AI model name
        session_id: Optional user session ID
        
    Returns:
        Content analysis results
    """
    workflow = ContentAnalysisWorkflow(ai_model_name)
    return await workflow.execute(url, session_id)


async def verify_source(
    url: str, 
    ai_model_name: Optional[str] = None,
    session_id: Optional[str] = None
) -> SourceVerificationState:
    """
    Convenience function to verify a source URL.
    
    Args:
        url: URL to verify
        ai_model_name: Optional AI model name
        session_id: Optional user session ID
        
    Returns:
        Source verification results
    """
    workflow = SourceVerificationWorkflow(ai_model_name)
    return await workflow.execute(url, session_id) 