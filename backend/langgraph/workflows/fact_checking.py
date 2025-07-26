"""
Main fact-checking workflow for TruthSeeQ platform.

This module implements the primary fact-checking workflow using LangGraph,
orchestrating content extraction, source verification, fact analysis,
and confidence scoring to provide comprehensive fact-checking results.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..nodes.content_extraction import ContentExtractionNode
from ..nodes.source_checking import SourceCheckingNode
from ..nodes.fact_analysis import FactAnalysisNode
from ..nodes.confidence_scoring import ConfidenceScoringNode

logger = logging.getLogger(__name__)


class FactCheckingWorkflow:
    """
    Main fact-checking workflow using LangGraph.
    
    This workflow orchestrates the complete fact-checking process:
    1. Content Extraction - Extract and prepare content for analysis
    2. Source Verification - Check source credibility and find relevant sources
    3. Fact Analysis - Analyze claims and determine verdicts
    4. Confidence Scoring - Calculate final confidence and generate results
    """
    
    def __init__(self):
        """Initialize the fact-checking workflow."""
        self.content_extraction_node = ContentExtractionNode()
        self.source_checking_node = SourceCheckingNode()
        self.fact_analysis_node = FactAnalysisNode()
        self.confidence_scoring_node = ConfidenceScoringNode()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow.
        
        Returns:
            Configured StateGraph for fact-checking workflow
        """
        # Create the workflow graph
        workflow = StateGraph(StateType=Dict[str, Any])
        
        # Add nodes to the workflow
        workflow.add_node("extract_content", self._extract_content)
        workflow.add_node("validate_content", self._validate_content)
        workflow.add_node("prepare_for_analysis", self._prepare_for_analysis)
        workflow.add_node("verify_source_credibility", self._verify_source_credibility)
        workflow.add_node("search_relevant_sources", self._search_relevant_sources)
        workflow.add_node("check_fact_database", self._check_fact_database)
        workflow.add_node("aggregate_source_information", self._aggregate_source_information)
        workflow.add_node("analyze_claims", self._analyze_claims)
        workflow.add_node("determine_verdicts", self._determine_verdicts)
        workflow.add_node("aggregate_results", self._aggregate_results)
        workflow.add_node("calculate_final_confidence", self._calculate_final_confidence)
        workflow.add_node("generate_final_results", self._generate_final_results)
        
        # Define the workflow flow
        workflow.set_entry_point("extract_content")
        
        # Content extraction flow
        workflow.add_edge("extract_content", "validate_content")
        workflow.add_conditional_edges(
            "validate_content",
            self._should_continue_after_validation,
            {
                "continue": "prepare_for_analysis",
                "stop": END
            }
        )
        workflow.add_edge("prepare_for_analysis", "verify_source_credibility")
        
        # Source verification flow
        workflow.add_edge("verify_source_credibility", "search_relevant_sources")
        workflow.add_edge("search_relevant_sources", "check_fact_database")
        workflow.add_edge("check_fact_database", "aggregate_source_information")
        
        # Fact analysis flow
        workflow.add_edge("aggregate_source_information", "analyze_claims")
        workflow.add_edge("analyze_claims", "determine_verdicts")
        workflow.add_edge("determine_verdicts", "aggregate_results")
        
        # Confidence scoring and final results
        workflow.add_edge("aggregate_results", "calculate_final_confidence")
        workflow.add_edge("calculate_final_confidence", "generate_final_results")
        workflow.add_edge("generate_final_results", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def run_workflow(self, content_id: int, content_text: str = "", content_url: str = "", source_domain: str = "") -> Dict[str, Any]:
        """
        Run the complete fact-checking workflow.
        
        Args:
            content_id: ID of the content to analyze
            content_text: Text content to analyze (optional if content_url provided)
            content_url: URL of the content (optional if content_text provided)
            source_domain: Domain of the content source
            
        Returns:
            Complete fact-checking results
        """
        try:
            # Prepare initial state
            initial_state = {
                "content_id": content_id,
                "content_text": content_text,
                "content_url": content_url,
                "source_domain": source_domain,
                "workflow_status": "started"
            }
            
            logger.info(f"Starting fact-checking workflow for content_id: {content_id}")
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"fact_check_{content_id}"}}
            result = self.workflow.invoke(initial_state, config)
            
            logger.info(f"Fact-checking workflow completed for content_id: {content_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running fact-checking workflow: {str(e)}")
            return {
                "content_id": content_id,
                "workflow_status": "failed",
                "error": str(e)
            }
    
    # Node execution methods
    def _extract_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content extraction node."""
        return self.content_extraction_node.extract_content(state)
    
    def _validate_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content validation node."""
        return self.content_extraction_node.validate_content(state)
    
    def _prepare_for_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute preparation for analysis node."""
        return self.content_extraction_node.prepare_for_analysis(state)
    
    def _verify_source_credibility(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute source credibility verification node."""
        return self.source_checking_node.verify_source_credibility(state)
    
    def _search_relevant_sources(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute relevant source search node."""
        return self.source_checking_node.search_relevant_sources(state)
    
    def _check_fact_database(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact database check node."""
        return self.source_checking_node.check_fact_database(state)
    
    def _aggregate_source_information(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute source information aggregation node."""
        return self.source_checking_node.aggregate_source_information(state)
    
    def _analyze_claims(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute claim analysis node."""
        return self.fact_analysis_node.analyze_claims(state)
    
    def _determine_verdicts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verdict determination node."""
        return self.fact_analysis_node.determine_verdicts(state)
    
    def _aggregate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute result aggregation node."""
        return self.fact_analysis_node.aggregate_results(state)
    
    def _calculate_final_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final confidence calculation node."""
        return self.confidence_scoring_node.calculate_final_confidence(state)
    
    def _generate_final_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final result generation node."""
        return self.confidence_scoring_node.generate_final_results(state)
    
    # Conditional edge functions
    def _should_continue_after_validation(self, state: Dict[str, Any]) -> str:
        """
        Determine whether to continue after content validation.
        
        Args:
            state: Current workflow state
            
        Returns:
            "continue" or "stop"
        """
        content_validation = state.get("content_validation", {})
        is_suitable = content_validation.get("is_suitable", True)
        
        if is_suitable:
            return "continue"
        else:
            logger.warning(f"Content validation failed for content_id: {state.get('content_id')}")
            return "stop"
    
    def get_workflow_status(self, content_id: int) -> Dict[str, Any]:
        """
        Get the current status of a workflow execution.
        
        Args:
            content_id: ID of the content being processed
            
        Returns:
            Workflow status information
        """
        try:
            config = {"configurable": {"thread_id": f"fact_check_{content_id}"}}
            # This would typically query the workflow's checkpoint store
            # For now, return a basic status
            return {
                "content_id": content_id,
                "status": "completed",  # This would be dynamic in practice
                "timestamp": "2024-01-01T00:00:00Z"
            }
        except Exception as e:
            logger.error(f"Error getting workflow status: {str(e)}")
            return {
                "content_id": content_id,
                "status": "unknown",
                "error": str(e)
            }


class AsyncFactCheckingWorkflow:
    """
    Asynchronous version of the fact-checking workflow.
    
    This class provides async methods for running the fact-checking workflow
    in background tasks or async contexts.
    """
    
    def __init__(self):
        """Initialize the async fact-checking workflow."""
        self.workflow = FactCheckingWorkflow()
    
    async def run_workflow_async(self, content_id: int, content_text: str = "", content_url: str = "", source_domain: str = "") -> Dict[str, Any]:
        """
        Run the fact-checking workflow asynchronously.
        
        Args:
            content_id: ID of the content to analyze
            content_text: Text content to analyze
            content_url: URL of the content
            source_domain: Domain of the content source
            
        Returns:
            Complete fact-checking results
        """
        # For now, run synchronously - in production, this would be truly async
        return self.workflow.run_workflow(content_id, content_text, content_url, source_domain)
    
    async def get_workflow_status_async(self, content_id: int) -> Dict[str, Any]:
        """
        Get workflow status asynchronously.
        
        Args:
            content_id: ID of the content being processed
            
        Returns:
            Workflow status information
        """
        return self.workflow.get_workflow_status(content_id)


# Factory function for creating workflow instances
def create_fact_checking_workflow() -> FactCheckingWorkflow:
    """
    Create a new fact-checking workflow instance.
    
    Returns:
        Configured FactCheckingWorkflow instance
    """
    return FactCheckingWorkflow()


def create_async_fact_checking_workflow() -> AsyncFactCheckingWorkflow:
    """
    Create a new async fact-checking workflow instance.
    
    Returns:
        Configured AsyncFactCheckingWorkflow instance
    """
    return AsyncFactCheckingWorkflow()
