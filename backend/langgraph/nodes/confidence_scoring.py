"""
Confidence scoring node for LangGraph workflows.

This module contains the confidence scoring node that calculates final
confidence scores and generates comprehensive fact-checking results.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ConfidenceScoringNode:
    """
    Node for calculating confidence scores and generating final results.
    
    This node calculates final confidence scores based on all analysis
    results and generates comprehensive fact-checking reports.
    """
    
    def __init__(self):
        """Initialize the confidence scoring node."""
        pass
    
    def calculate_final_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final confidence scores for the fact-checking results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final confidence scores
        """
        try:
            fact_check_verdicts = state.get("fact_check_verdicts", [])
            aggregated_results = state.get("aggregated_results", {})
            source_credibility = state.get("source_credibility", {})
            content_quality = state.get("content_quality", {})
            
            if not fact_check_verdicts:
                logger.warning("No fact-check verdicts available for confidence calculation")
                return {
                    **state,
                    "final_confidence": 0.0,
                    "confidence_calculation_status": "completed"
                }
            
            # Calculate individual confidence factors
            claim_confidence = self._calculate_claim_confidence(fact_check_verdicts)
            source_confidence = self._calculate_source_confidence(source_credibility, aggregated_results)
            quality_confidence = self._calculate_quality_confidence(content_quality)
            evidence_confidence = self._calculate_evidence_confidence(aggregated_results)
            
            # Calculate weighted final confidence
            final_confidence = (
                claim_confidence * 0.4 +
                source_confidence * 0.3 +
                quality_confidence * 0.2 +
                evidence_confidence * 0.1
            )
            
            confidence_breakdown = {
                "claim_confidence": claim_confidence,
                "source_confidence": source_confidence,
                "quality_confidence": quality_confidence,
                "evidence_confidence": evidence_confidence,
                "final_confidence": final_confidence,
                "confidence_level": self._get_confidence_level(final_confidence)
            }
            
            return {
                **state,
                "confidence_breakdown": confidence_breakdown,
                "final_confidence": final_confidence,
                "confidence_calculation_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in confidence calculation: {str(e)}")
            return {
                **state,
                "confidence_calculation_status": "failed",
                "error": f"Confidence calculation failed: {str(e)}"
            }
    
    def generate_final_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive final fact-checking results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final results
        """
        try:
            content_id = state.get("content_id")
            fact_check_verdicts = state.get("fact_check_verdicts", [])
            aggregated_results = state.get("aggregated_results", {})
            confidence_breakdown = state.get("confidence_breakdown", {})
            source_credibility = state.get("source_credibility", {})
            content_quality = state.get("content_quality", {})
            
            # Generate comprehensive results
            final_results = {
                "content_id": content_id,
                "timestamp": datetime.now().isoformat(),
                "overall_verdict": aggregated_results.get("overall_verdict", "unverifiable"),
                "confidence_score": confidence_breakdown.get("final_confidence", 0.0),
                "confidence_level": confidence_breakdown.get("confidence_level", "low"),
                "total_claims_analyzed": aggregated_results.get("total_claims_analyzed", 0),
                "verdict_distribution": aggregated_results.get("verdict_distribution", {}),
                "overall_reasoning": aggregated_results.get("overall_reasoning", ""),
                "reliability_score": aggregated_results.get("reliability_score", 0.0),
                "source_quality": aggregated_results.get("source_quality", {}),
                "fact_check_summary": aggregated_results.get("fact_check_summary", {}),
                "detailed_verdicts": fact_check_verdicts,
                "confidence_breakdown": confidence_breakdown,
                "source_credibility": source_credibility,
                "content_quality": content_quality,
                "analysis_metadata": {
                    "workflow_version": "1.0",
                    "analysis_completed": True,
                    "total_processing_time": "completed"
                }
            }
            
            # Add recommendations
            final_results["recommendations"] = self._generate_recommendations(
                final_results, confidence_breakdown, aggregated_results
            )
            
            # Add risk assessment
            final_results["risk_assessment"] = self._assess_risk_level(
                final_results, confidence_breakdown
            )
            
            return {
                **state,
                "final_results": final_results,
                "result_generation_status": "completed",
                "workflow_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in final result generation: {str(e)}")
            return {
                **state,
                "result_generation_status": "failed",
                "error": f"Final result generation failed: {str(e)}"
            }
    
    def _calculate_claim_confidence(self, fact_check_verdicts: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on individual claim verdicts.
        
        Args:
            fact_check_verdicts: List of fact-check verdicts
            
        Returns:
            Claim confidence score (0.0 to 1.0)
        """
        if not fact_check_verdicts:
            return 0.0
        
        total_confidence = 0
        total_claims = len(fact_check_verdicts)
        
        for verdict in fact_check_verdicts:
            confidence = verdict.get("confidence", 0.0)
            total_confidence += confidence
        
        return total_confidence / total_claims if total_claims > 0 else 0.0
    
    def _calculate_source_confidence(self, source_credibility: Dict[str, Any], aggregated_results: Dict[str, Any]) -> float:
        """
        Calculate confidence based on source quality.
        
        Args:
            source_credibility: Source credibility information
            aggregated_results: Aggregated analysis results
            
        Returns:
            Source confidence score (0.0 to 1.0)
        """
        # Primary source credibility
        primary_credibility = source_credibility.get("credibility_score", 0.0)
        
        # Source quality from aggregated results
        source_quality = aggregated_results.get("source_quality", {})
        quality_score = source_quality.get("quality_score", 0.0)
        
        # Average credibility of relevant sources
        relevant_sources = aggregated_results.get("relevant_sources", [])
        if relevant_sources:
            avg_source_credibility = sum(
                s.get("credibility_score", 0.0) for s in relevant_sources
            ) / len(relevant_sources)
        else:
            avg_source_credibility = 0.0
        
        # Weighted combination
        source_confidence = (
            primary_credibility * 0.4 +
            quality_score * 0.4 +
            avg_source_credibility * 0.2
        )
        
        return min(1.0, max(0.0, source_confidence))
    
    def _calculate_quality_confidence(self, content_quality: Dict[str, Any]) -> float:
        """
        Calculate confidence based on content quality.
        
        Args:
            content_quality: Content quality assessment
            
        Returns:
            Quality confidence score (0.0 to 1.0)
        """
        quality_score = content_quality.get("overall_quality_score", 0.0)
        readability_score = content_quality.get("readability_score", 0.0)
        
        # Normalize readability score (0-100 to 0-1)
        normalized_readability = readability_score / 100.0
        
        # Weighted combination
        quality_confidence = (
            quality_score * 0.7 +
            normalized_readability * 0.3
        )
        
        return min(1.0, max(0.0, quality_confidence))
    
    def _calculate_evidence_confidence(self, aggregated_results: Dict[str, Any]) -> float:
        """
        Calculate confidence based on evidence strength.
        
        Args:
            aggregated_results: Aggregated analysis results
            
        Returns:
            Evidence confidence score (0.0 to 1.0)
        """
        # Count high-quality sources
        relevant_sources = aggregated_results.get("relevant_sources", [])
        high_quality_sources = len([
            s for s in relevant_sources 
            if s.get("credibility_score", 0.0) > 0.7
        ])
        
        # Fact database coverage
        fact_database_results = aggregated_results.get("fact_database_results", [])
        fact_coverage = len([r for r in fact_database_results if r.get("fact_results")])
        
        # Calculate evidence confidence
        total_sources = len(relevant_sources)
        source_evidence = high_quality_sources / max(total_sources, 1)
        
        fact_evidence = min(fact_coverage / max(len(fact_database_results), 1), 1.0)
        
        evidence_confidence = (source_evidence * 0.7 + fact_evidence * 0.3)
        
        return min(1.0, max(0.0, evidence_confidence))
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Get confidence level based on confidence score.
        
        Args:
            confidence_score: Confidence score (0.0 to 1.0)
            
        Returns:
            Confidence level string
        """
        if confidence_score >= 0.8:
            return "very_high"
        elif confidence_score >= 0.6:
            return "high"
        elif confidence_score >= 0.4:
            return "medium"
        elif confidence_score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _generate_recommendations(self, final_results: Dict[str, Any], confidence_breakdown: Dict[str, Any], aggregated_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            final_results: Final fact-checking results
            confidence_breakdown: Confidence breakdown
            aggregated_results: Aggregated analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Confidence-based recommendations
        confidence_level = confidence_breakdown.get("confidence_level", "low")
        if confidence_level in ["very_low", "low"]:
            recommendations.append("Consider seeking additional sources for verification.")
            recommendations.append("The analysis has low confidence - results should be interpreted with caution.")
        
        # Source quality recommendations
        source_quality = aggregated_results.get("source_quality", {})
        quality_level = source_quality.get("quality_level", "unknown")
        if quality_level == "low":
            recommendations.append("Source quality is low - consider using more authoritative sources.")
        
        # Content quality recommendations
        content_quality = final_results.get("content_quality", {})
        quality_score = content_quality.get("overall_quality_score", 0)
        if quality_score < 50:
            recommendations.append("Content quality is poor - consider using better quality content for analysis.")
        
        # Verdict-based recommendations
        overall_verdict = final_results.get("overall_verdict", "unverifiable")
        if overall_verdict == "mixed":
            recommendations.append("Content has mixed evidence - consider analyzing individual claims separately.")
        elif overall_verdict == "unverifiable":
            recommendations.append("Content could not be verified - consider using different sources or approaches.")
        
        # Evidence-based recommendations
        total_sources = len(aggregated_results.get("relevant_sources", []))
        if total_sources < 3:
            recommendations.append("Limited sources found - consider expanding the search for additional verification.")
        
        return recommendations
    
    def _assess_risk_level(self, final_results: Dict[str, Any], confidence_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level of the fact-checking results.
        
        Args:
            final_results: Final fact-checking results
            confidence_breakdown: Confidence breakdown
            
        Returns:
            Risk assessment dictionary
        """
        confidence_score = confidence_breakdown.get("final_confidence", 0.0)
        overall_verdict = final_results.get("overall_verdict", "unverifiable")
        
        # Determine risk level
        if confidence_score < 0.3:
            risk_level = "high"
        elif confidence_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Risk factors
        risk_factors = []
        
        if confidence_score < 0.5:
            risk_factors.append("low_confidence")
        
        if overall_verdict in ["unverifiable", "mixed"]:
            risk_factors.append("unclear_verdict")
        
        if final_results.get("reliability_score", 0.0) < 0.5:
            risk_factors.append("low_reliability")
        
        # Risk mitigation suggestions
        mitigation_suggestions = []
        
        if "low_confidence" in risk_factors:
            mitigation_suggestions.append("Seek additional authoritative sources")
            mitigation_suggestions.append("Consider manual review by experts")
        
        if "unclear_verdict" in risk_factors:
            mitigation_suggestions.append("Analyze claims individually")
            mitigation_suggestions.append("Provide detailed reasoning for each claim")
        
        if "low_reliability" in risk_factors:
            mitigation_suggestions.append("Improve source quality assessment")
            mitigation_suggestions.append("Use more diverse source types")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_suggestions": mitigation_suggestions,
            "confidence_threshold_met": confidence_score >= 0.5
        }
