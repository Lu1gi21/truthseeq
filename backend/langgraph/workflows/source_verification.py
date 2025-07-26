"""
Source verification workflow for TruthSeeQ platform.

This module implements a source verification workflow using LangGraph,
providing comprehensive source credibility assessment, domain analysis,
and verification capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..tools.web_search import WebSearchTool, SourceCredibilityChecker, ContentExtractor
from ..tools.fact_database import FactDatabaseTool

logger = logging.getLogger(__name__)


class SourceVerificationWorkflow:
    """
    Source verification workflow using LangGraph.
    
    This workflow provides comprehensive source verification including:
    1. Domain Analysis - Analyze domain patterns and history
    2. Source Credibility Assessment - Evaluate source reputation
    3. Content Verification - Verify content against known facts
    4. Cross-Reference Analysis - Compare with other sources
    """
    
    def __init__(self):
        """Initialize the source verification workflow."""
        self.web_search_tool = WebSearchTool()
        self.credibility_checker = SourceCredibilityChecker()
        self.content_extractor = ContentExtractor()
        self.fact_database_tool = FactDatabaseTool()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow.
        
        Returns:
            Configured StateGraph for source verification workflow
        """
        # Create the workflow graph
        workflow = StateGraph(StateType=Dict[str, Any])
        
        # Add nodes to the workflow
        workflow.add_node("extract_source_info", self._extract_source_info)
        workflow.add_node("analyze_domain", self._analyze_domain)
        workflow.add_node("check_credibility", self._check_credibility)
        workflow.add_node("search_related_sources", self._search_related_sources)
        workflow.add_node("verify_content", self._verify_content)
        workflow.add_node("cross_reference", self._cross_reference)
        workflow.add_node("assess_reputation", self._assess_reputation)
        workflow.add_node("aggregate_verification", self._aggregate_verification)
        workflow.add_node("generate_verification_report", self._generate_verification_report)
        
        # Define the workflow flow
        workflow.set_entry_point("extract_source_info")
        
        # Main verification flow
        workflow.add_edge("extract_source_info", "analyze_domain")
        workflow.add_edge("analyze_domain", "check_credibility")
        workflow.add_edge("check_credibility", "search_related_sources")
        workflow.add_edge("search_related_sources", "verify_content")
        workflow.add_edge("verify_content", "cross_reference")
        workflow.add_edge("cross_reference", "assess_reputation")
        workflow.add_edge("assess_reputation", "aggregate_verification")
        workflow.add_edge("aggregate_verification", "generate_verification_report")
        workflow.add_edge("generate_verification_report", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def run_workflow(self, source_url: str, content_text: str = "") -> Dict[str, Any]:
        """
        Run the complete source verification workflow.
        
        Args:
            source_url: URL of the source to verify
            content_text: Optional content text for verification
            
        Returns:
            Complete source verification results
        """
        try:
            # Prepare initial state
            initial_state = {
                "source_url": source_url,
                "content_text": content_text,
                "workflow_status": "started"
            }
            
            logger.info(f"Starting source verification workflow for URL: {source_url}")
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"source_verification_{hash(source_url)}"}}
            result = self.workflow.invoke(initial_state, config)
            
            logger.info(f"Source verification workflow completed for URL: {source_url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running source verification workflow: {str(e)}")
            return {
                "source_url": source_url,
                "workflow_status": "failed",
                "error": str(e)
            }
    
    # Node execution methods
    def _extract_source_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute source information extraction."""
        try:
            source_url = state.get("source_url", "")
            
            if not source_url:
                return {
                    **state,
                    "source_info": {},
                    "extraction_status": "failed"
                }
            
            # Extract basic source information
            from urllib.parse import urlparse
            parsed_url = urlparse(source_url)
            
            source_info = {
                "url": source_url,
                "domain": parsed_url.netloc,
                "protocol": parsed_url.scheme,
                "path": parsed_url.path,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment
            }
            
            return {
                **state,
                "source_info": source_info,
                "extraction_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in source info extraction: {str(e)}")
            return {
                **state,
                "extraction_status": "failed",
                "error": str(e)
            }
    
    def _analyze_domain(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain analysis."""
        try:
            source_info = state.get("source_info", {})
            domain = source_info.get("domain", "")
            
            if not domain:
                return {
                    **state,
                    "domain_analysis": {},
                    "domain_analysis_status": "failed"
                }
            
            # Analyze domain patterns
            domain_analysis = self._perform_domain_analysis(domain)
            
            return {
                **state,
                "domain_analysis": domain_analysis,
                "domain_analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in domain analysis: {str(e)}")
            return {
                **state,
                "domain_analysis_status": "failed",
                "error": str(e)
            }
    
    def _check_credibility(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute credibility checking."""
        try:
            source_url = state.get("source_url", "")
            
            if not source_url:
                return {
                    **state,
                    "credibility_check": {},
                    "credibility_check_status": "failed"
                }
            
            # Check source credibility
            credibility_result = self.credibility_checker._run(source_url)
            
            return {
                **state,
                "credibility_check": credibility_result,
                "credibility_check_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in credibility check: {str(e)}")
            return {
                **state,
                "credibility_check_status": "failed",
                "error": str(e)
            }
    
    def _search_related_sources(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute related source search."""
        try:
            source_info = state.get("source_info", {})
            domain = source_info.get("domain", "")
            
            if not domain:
                return {
                    **state,
                    "related_sources": [],
                    "related_sources_status": "failed"
                }
            
            # Search for related sources
            search_query = f"site:{domain} credibility reputation"
            search_results = self.web_search_tool._run(
                query=search_query,
                max_results=5,
                search_type="web"
            )
            
            return {
                **state,
                "related_sources": search_results,
                "related_sources_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in related sources search: {str(e)}")
            return {
                **state,
                "related_sources_status": "failed",
                "error": str(e)
            }
    
    def _verify_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content verification."""
        try:
            content_text = state.get("content_text", "")
            source_url = state.get("source_url", "")
            
            if not content_text and not source_url:
                return {
                    **state,
                    "content_verification": {},
                    "content_verification_status": "failed"
                }
            
            # Extract content if not provided
            if not content_text and source_url:
                extraction_result = self.content_extractor._run(source_url)
                if extraction_result.get("success"):
                    content_text = extraction_result.get("content", "")
            
            # Verify content against fact database
            content_verification = {}
            if content_text:
                # Search fact database for relevant information
                fact_results = self.fact_database_tool._run(
                    query=content_text[:200],  # Use first 200 characters as query
                    max_results=3
                )
                content_verification["fact_database_results"] = fact_results
            
            return {
                **state,
                "content_verification": content_verification,
                "content_verification_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in content verification: {str(e)}")
            return {
                **state,
                "content_verification_status": "failed",
                "error": str(e)
            }
    
    def _cross_reference(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cross-reference analysis."""
        try:
            source_url = state.get("source_url", "")
            content_text = state.get("content_text", "")
            
            if not source_url:
                return {
                    **state,
                    "cross_reference": {},
                    "cross_reference_status": "failed"
                }
            
            # Search for similar content from other sources
            cross_reference_results = {}
            
            if content_text:
                # Create search query from content
                search_query = self._create_search_query_from_content(content_text)
                
                # Search for similar content
                similar_content = self.web_search_tool._run(
                    query=search_query,
                    max_results=5,
                    search_type="web"
                )
                
                cross_reference_results["similar_content"] = similar_content
            
            # Search for domain mentions
            source_info = state.get("source_info", {})
            domain = source_info.get("domain", "")
            
            if domain:
                domain_mentions = self.web_search_tool._run(
                    query=f'"{domain}" credibility reputation',
                    max_results=3,
                    search_type="web"
                )
                cross_reference_results["domain_mentions"] = domain_mentions
            
            return {
                **state,
                "cross_reference": cross_reference_results,
                "cross_reference_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in cross-reference analysis: {str(e)}")
            return {
                **state,
                "cross_reference_status": "failed",
                "error": str(e)
            }
    
    def _assess_reputation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reputation assessment."""
        try:
            credibility_check = state.get("credibility_check", {})
            domain_analysis = state.get("domain_analysis", {})
            related_sources = state.get("related_sources", [])
            cross_reference = state.get("cross_reference", {})
            
            # Assess overall reputation
            reputation_assessment = self._calculate_reputation_score(
                credibility_check, domain_analysis, related_sources, cross_reference
            )
            
            return {
                **state,
                "reputation_assessment": reputation_assessment,
                "reputation_assessment_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in reputation assessment: {str(e)}")
            return {
                **state,
                "reputation_assessment_status": "failed",
                "error": str(e)
            }
    
    def _aggregate_verification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verification aggregation."""
        try:
            credibility_check = state.get("credibility_check", {})
            domain_analysis = state.get("domain_analysis", {})
            content_verification = state.get("content_verification", {})
            cross_reference = state.get("cross_reference", {})
            reputation_assessment = state.get("reputation_assessment", {})
            
            # Aggregate all verification results
            aggregated_verification = {
                "overall_credibility_score": reputation_assessment.get("overall_score", 0),
                "credibility_level": reputation_assessment.get("credibility_level", "unknown"),
                "verification_summary": {
                    "domain_analysis": domain_analysis,
                    "credibility_check": credibility_check,
                    "content_verification": content_verification,
                    "cross_reference": cross_reference
                },
                "risk_factors": reputation_assessment.get("risk_factors", []),
                "recommendations": reputation_assessment.get("recommendations", [])
            }
            
            return {
                **state,
                "aggregated_verification": aggregated_verification,
                "aggregation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in verification aggregation: {str(e)}")
            return {
                **state,
                "aggregation_status": "failed",
                "error": str(e)
            }
    
    def _generate_verification_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verification report generation."""
        try:
            source_url = state.get("source_url", "")
            aggregated_verification = state.get("aggregated_verification", {})
            source_info = state.get("source_info", {})
            
            # Generate comprehensive verification report
            verification_report = {
                "source_url": source_url,
                "timestamp": "2024-01-01T00:00:00Z",  # Would be dynamic
                "overall_credibility_score": aggregated_verification.get("overall_credibility_score", 0),
                "credibility_level": aggregated_verification.get("credibility_level", "unknown"),
                "source_info": source_info,
                "verification_details": aggregated_verification.get("verification_summary", {}),
                "risk_assessment": {
                    "risk_factors": aggregated_verification.get("risk_factors", []),
                    "overall_risk": self._determine_overall_risk(aggregated_verification)
                },
                "recommendations": aggregated_verification.get("recommendations", []),
                "verification_metadata": {
                    "workflow_version": "1.0",
                    "verification_completed": True
                }
            }
            
            return {
                **state,
                "verification_report": verification_report,
                "report_generation_status": "completed",
                "workflow_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in verification report generation: {str(e)}")
            return {
                **state,
                "report_generation_status": "failed",
                "error": str(e)
            }
    
    # Helper methods
    def _perform_domain_analysis(self, domain: str) -> Dict[str, Any]:
        """Perform comprehensive domain analysis."""
        analysis = {
            "domain": domain,
            "domain_length": len(domain),
            "has_subdomain": "." in domain.split(".")[0],
            "is_short_domain": len(domain) < 15,
            "suspicious_patterns": [],
            "trust_indicators": []
        }
        
        # Check for suspicious patterns
        suspicious_keywords = ["fake", "hoax", "conspiracy", "clickbait", "scam"]
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                analysis["suspicious_patterns"].append(keyword)
        
        # Check for trust indicators
        trust_keywords = ["news", "times", "post", "tribune", "herald", "journal"]
        for keyword in trust_keywords:
            if keyword in domain.lower():
                analysis["trust_indicators"].append(keyword)
        
        # Check for educational/government domains
        if ".edu" in domain:
            analysis["trust_indicators"].append("educational")
        if ".gov" in domain:
            analysis["trust_indicators"].append("government")
        
        return analysis
    
    def _create_search_query_from_content(self, content_text: str) -> str:
        """Create a search query from content text."""
        # Simple query creation - in production, use more sophisticated NLP
        words = content_text.split()[:10]  # Take first 10 words
        return " ".join(words)
    
    def _calculate_reputation_score(self, credibility_check: Dict, domain_analysis: Dict, related_sources: List, cross_reference: Dict) -> Dict[str, Any]:
        """Calculate overall reputation score."""
        # Base score from credibility check
        base_score = credibility_check.get("credibility_score", 0.5)
        
        # Adjust based on domain analysis
        domain_score = 0.5
        if domain_analysis.get("trust_indicators"):
            domain_score += 0.2
        if domain_analysis.get("suspicious_patterns"):
            domain_score -= 0.3
        
        # Adjust based on related sources
        source_score = 0.5
        if related_sources:
            source_score += 0.1
        
        # Calculate overall score
        overall_score = (base_score * 0.5 + domain_score * 0.3 + source_score * 0.2)
        
        # Determine credibility level
        if overall_score >= 0.8:
            credibility_level = "high"
        elif overall_score >= 0.6:
            credibility_level = "medium"
        elif overall_score >= 0.4:
            credibility_level = "low"
        else:
            credibility_level = "very_low"
        
        # Identify risk factors
        risk_factors = []
        if domain_analysis.get("suspicious_patterns"):
            risk_factors.append("suspicious_domain_patterns")
        if base_score < 0.3:
            risk_factors.append("low_credibility_score")
        if not related_sources:
            risk_factors.append("no_related_sources")
        
        # Generate recommendations
        recommendations = []
        if overall_score < 0.5:
            recommendations.append("Source has low credibility - verify information independently")
        if risk_factors:
            recommendations.append("Multiple risk factors detected - exercise caution")
        
        return {
            "overall_score": overall_score,
            "credibility_level": credibility_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "score_breakdown": {
                "base_score": base_score,
                "domain_score": domain_score,
                "source_score": source_score
            }
        }
    
    def _determine_overall_risk(self, aggregated_verification: Dict) -> str:
        """Determine overall risk level."""
        risk_factors = aggregated_verification.get("risk_factors", [])
        credibility_score = aggregated_verification.get("overall_credibility_score", 0)
        
        if len(risk_factors) > 2 or credibility_score < 0.3:
            return "high"
        elif len(risk_factors) > 0 or credibility_score < 0.6:
            return "medium"
        else:
            return "low"


# Factory function for creating workflow instances
def create_source_verification_workflow() -> SourceVerificationWorkflow:
    """
    Create a new source verification workflow instance.
    
    Returns:
        Configured SourceVerificationWorkflow instance
    """
    return SourceVerificationWorkflow()
