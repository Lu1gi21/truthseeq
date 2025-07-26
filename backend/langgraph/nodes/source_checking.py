"""
Source checking node for LangGraph workflows.

This module contains the source checking node that verifies source credibility
and finds relevant information for fact-checking.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from ..tools.web_search import WebSearchTool, SourceCredibilityChecker
from ..tools.fact_database import FactDatabaseTool, FactVerificationTool

logger = logging.getLogger(__name__)


class SourceCheckingNode:
    """
    Node for checking source credibility and finding relevant information.
    
    This node verifies the credibility of sources, searches for relevant
    information, and prepares source data for fact-checking analysis.
    """
    
    def __init__(self):
        """Initialize the source checking node."""
        self.web_search_tool = WebSearchTool()
        self.credibility_checker = SourceCredibilityChecker()
        self.fact_database_tool = FactDatabaseTool()
        self.fact_verification_tool = FactVerificationTool(self.fact_database_tool)
    
    def verify_source_credibility(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the credibility of the content source.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with source credibility assessment
        """
        try:
            content_url = state.get("content_url", "")
            source_domain = state.get("source_domain", "")
            
            if not content_url and not source_domain:
                logger.warning("No source URL or domain provided for credibility check")
                return {
                    **state,
                    "source_credibility": {
                        "credibility_score": 0.0,
                        "reputation": "unknown",
                        "reason": "No source information available",
                        "is_known_source": False
                    },
                    "source_verification_status": "completed"
                }
            
            # Check credibility of the source
            credibility_result = self.credibility_checker._run(content_url or source_domain)
            
            # Additional domain analysis if available
            domain_analysis = {}
            if source_domain:
                domain_analysis = self._analyze_domain_patterns(source_domain)
            
            source_credibility = {
                "url": content_url,
                "domain": source_domain,
                "credibility_score": credibility_result.get("credibility_score", 0.0),
                "reputation": credibility_result.get("reputation", "unknown"),
                "reason": credibility_result.get("reason", "Unknown source"),
                "is_known_source": credibility_result.get("is_known_source", False),
                "domain_analysis": domain_analysis
            }
            
            return {
                **state,
                "source_credibility": source_credibility,
                "source_verification_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in source credibility verification: {str(e)}")
            return {
                **state,
                "source_verification_status": "failed",
                "error": f"Source credibility verification failed: {str(e)}"
            }
    
    def search_relevant_sources(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for relevant sources to verify claims.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with relevant sources
        """
        try:
            search_queries = state.get("search_queries", [])
            extracted_claims = state.get("extracted_claims", [])
            
            if not search_queries and not extracted_claims:
                logger.warning("No search queries or claims available for source search")
                return {
                    **state,
                    "relevant_sources": [],
                    "source_search_status": "completed"
                }
            
            all_sources = []
            
            # Search for each query
            for query_data in search_queries:
                query = query_data.get("query", "")
                original_claim = query_data.get("original_claim", "")
                
                if not query:
                    continue
                
                # Perform web search
                search_results = self.web_search_tool._run(
                    query=query,
                    max_results=5,
                    search_type="web",
                    freshness="pm"  # Past month for recent information
                )
                
                # Check credibility of each result
                verified_sources = []
                for result in search_results:
                    source_url = result.get("url", "")
                    if source_url:
                        credibility = self.credibility_checker._run(source_url)
                        
                        verified_sources.append({
                            **result,
                            "credibility_score": credibility.get("credibility_score", 0.0),
                            "reputation": credibility.get("reputation", "unknown"),
                            "is_known_source": credibility.get("is_known_source", False),
                            "original_claim": original_claim,
                            "search_query": query
                        })
                
                # Sort by credibility and relevance
                verified_sources.sort(
                    key=lambda x: (x.get("credibility_score", 0), x.get("relevance_score", 0)),
                    reverse=True
                )
                
                all_sources.extend(verified_sources)
            
            # Remove duplicates and limit results
            unique_sources = self._deduplicate_sources(all_sources)
            top_sources = unique_sources[:10]  # Limit to top 10 sources
            
            return {
                **state,
                "relevant_sources": top_sources,
                "source_search_status": "completed",
                "total_sources_found": len(all_sources),
                "unique_sources": len(unique_sources)
            }
            
        except Exception as e:
            logger.error(f"Error in source search: {str(e)}")
            return {
                **state,
                "source_search_status": "failed",
                "error": f"Source search failed: {str(e)}"
            }
    
    def check_fact_database(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check fact database for relevant verified information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with fact database results
        """
        try:
            extracted_claims = state.get("extracted_claims", [])
            
            if not extracted_claims:
                logger.warning("No claims available for fact database check")
                return {
                    **state,
                    "fact_database_results": [],
                    "fact_database_status": "completed"
                }
            
            fact_database_results = []
            
            # Check each claim against the fact database
            for claim_data in extracted_claims:
                claim_text = claim_data.get("claim", "")
                if not claim_text:
                    continue
                
                # Search fact database
                fact_results = self.fact_database_tool._run(
                    query=claim_text,
                    max_results=3
                )
                
                # Verify claim against facts
                verification_result = self.fact_verification_tool._run(claim_text)
                
                fact_database_results.append({
                    "claim": claim_text,
                    "fact_results": fact_results,
                    "verification_result": verification_result,
                    "original_confidence": claim_data.get("confidence", 0.5)
                })
            
            return {
                **state,
                "fact_database_results": fact_database_results,
                "fact_database_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in fact database check: {str(e)}")
            return {
                **state,
                "fact_database_status": "failed",
                "error": f"Fact database check failed: {str(e)}"
            }
    
    def aggregate_source_information(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate and analyze all source information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with aggregated source analysis
        """
        try:
            source_credibility = state.get("source_credibility", {})
            relevant_sources = state.get("relevant_sources", [])
            fact_database_results = state.get("fact_database_results", [])
            
            # Analyze source credibility distribution
            credibility_scores = [s.get("credibility_score", 0) for s in relevant_sources]
            avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0
            
            # Count source types
            source_types = {}
            for source in relevant_sources:
                domain = source.get("source_domain", "unknown")
                source_types[domain] = source_types.get(domain, 0) + 1
            
            # Analyze fact database verification results
            verification_summary = {
                "total_claims_checked": len(fact_database_results),
                "supported_claims": 0,
                "contradicted_claims": 0,
                "unverified_claims": 0,
                "mixed_evidence_claims": 0
            }
            
            for result in fact_database_results:
                verification = result.get("verification_result", {})
                status = verification.get("verification_status", "unverified")
                
                if status == "supported":
                    verification_summary["supported_claims"] += 1
                elif status == "contradicted":
                    verification_summary["contradicted_claims"] += 1
                elif status == "mixed":
                    verification_summary["mixed_evidence_claims"] += 1
                else:
                    verification_summary["unverified_claims"] += 1
            
            # Overall source quality assessment
            source_quality = self._assess_overall_source_quality(
                source_credibility, relevant_sources, fact_database_results
            )
            
            aggregated_sources = {
                "source_credibility": source_credibility,
                "relevant_sources": relevant_sources,
                "fact_database_results": fact_database_results,
                "source_analysis": {
                    "average_credibility": avg_credibility,
                    "source_type_distribution": source_types,
                    "total_sources": len(relevant_sources),
                    "high_credibility_sources": len([s for s in relevant_sources if s.get("credibility_score", 0) > 0.7]),
                    "known_sources": len([s for s in relevant_sources if s.get("is_known_source", False)])
                },
                "verification_summary": verification_summary,
                "overall_source_quality": source_quality
            }
            
            return {
                **state,
                "aggregated_sources": aggregated_sources,
                "source_aggregation_status": "completed",
                "next_step": "fact_analysis"
            }
            
        except Exception as e:
            logger.error(f"Error in source aggregation: {str(e)}")
            return {
                **state,
                "source_aggregation_status": "failed",
                "error": f"Source aggregation failed: {str(e)}"
            }
    
    def _analyze_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """
        Analyze domain patterns for additional credibility indicators.
        
        Args:
            domain: Domain name to analyze
            
        Returns:
            Dictionary with domain analysis results
        """
        analysis = {
            "domain_length": len(domain),
            "has_subdomain": "." in domain.split(".")[0],
            "is_short_domain": len(domain) < 15,
            "suspicious_patterns": []
        }
        
        # Check for suspicious patterns
        suspicious_keywords = ["fake", "hoax", "conspiracy", "clickbait", "scam"]
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                analysis["suspicious_patterns"].append(keyword)
        
        # Check for news-like patterns
        news_keywords = ["news", "times", "post", "tribune", "herald", "journal"]
        analysis["news_like"] = any(keyword in domain.lower() for keyword in news_keywords)
        
        return analysis
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate sources based on URL.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Deduplicated list of sources
        """
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        return unique_sources
    
    def _assess_overall_source_quality(self, source_credibility: Dict, relevant_sources: List, fact_database_results: List) -> Dict[str, Any]:
        """
        Assess overall source quality for the fact-checking process.
        
        Args:
            source_credibility: Source credibility information
            relevant_sources: List of relevant sources found
            fact_database_results: Results from fact database check
            
        Returns:
            Overall source quality assessment
        """
        # Calculate quality metrics
        primary_source_credibility = source_credibility.get("credibility_score", 0)
        
        relevant_source_credibility = [
            s.get("credibility_score", 0) for s in relevant_sources
        ]
        avg_relevant_credibility = sum(relevant_source_credibility) / len(relevant_source_credibility) if relevant_source_credibility else 0
        
        # Count high-quality sources
        high_quality_sources = len([s for s in relevant_sources if s.get("credibility_score", 0) > 0.7])
        
        # Fact database coverage
        fact_coverage = len([r for r in fact_database_results if r.get("fact_results")])
        
        # Overall quality score
        quality_score = (
            primary_source_credibility * 0.3 +
            avg_relevant_credibility * 0.4 +
            (high_quality_sources / max(len(relevant_sources), 1)) * 0.2 +
            (fact_coverage / max(len(fact_database_results), 1)) * 0.1
        )
        
        # Quality level
        if quality_score > 0.7:
            quality_level = "high"
        elif quality_score > 0.4:
            quality_level = "medium"
        else:
            quality_level = "low"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "primary_source_credibility": primary_source_credibility,
            "average_relevant_credibility": avg_relevant_credibility,
            "high_quality_sources_count": high_quality_sources,
            "fact_database_coverage": fact_coverage,
            "total_sources_analyzed": len(relevant_sources)
        }
