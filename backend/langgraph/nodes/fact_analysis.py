"""
Fact analysis node for LangGraph workflows.

This module contains the fact analysis node that analyzes claims,
determines fact-checking verdicts, and provides reasoning for results.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from ..tools.content_analysis import SentimentAnalysisTool, BiasDetectionTool
from ..tools.fact_database import FactVerificationTool

logger = logging.getLogger(__name__)


class FactAnalysisNode:
    """
    Node for analyzing facts and determining fact-checking verdicts.
    
    This node analyzes claims against available evidence, determines
    fact-checking verdicts, and provides detailed reasoning for results.
    """
    
    def __init__(self):
        """Initialize the fact analysis node."""
        self.sentiment_analyzer = SentimentAnalysisTool()
        self.bias_detector = BiasDetectionTool()
        self.fact_verifier = FactVerificationTool()
    
    def analyze_claims(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze individual claims for fact-checking.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with claim analysis results
        """
        try:
            extracted_claims = state.get("extracted_claims", [])
            aggregated_sources = state.get("aggregated_sources", {})
            relevant_sources = aggregated_sources.get("relevant_sources", [])
            fact_database_results = aggregated_sources.get("fact_database_results", [])
            
            if not extracted_claims:
                logger.warning("No claims available for analysis")
                return {
                    **state,
                    "claim_analysis": [],
                    "claim_analysis_status": "completed"
                }
            
            claim_analyses = []
            
            for claim_data in extracted_claims:
                claim_text = claim_data.get("claim", "")
                original_confidence = claim_data.get("confidence", 0.5)
                
                if not claim_text:
                    continue
                
                # Analyze sentiment and bias of the claim
                sentiment_result = self.sentiment_analyzer._run(claim_text)
                bias_result = self.bias_detector._run(claim_text)
                
                # Find relevant sources for this claim
                claim_sources = self._find_claim_sources(claim_text, relevant_sources)
                
                # Get fact database verification for this claim
                fact_verification = self._get_claim_fact_verification(claim_text, fact_database_results)
                
                # Analyze claim against evidence
                claim_analysis = self._analyze_single_claim(
                    claim_text, claim_sources, fact_verification, sentiment_result, bias_result
                )
                
                claim_analyses.append({
                    "claim": claim_text,
                    "original_confidence": original_confidence,
                    "sentiment_analysis": sentiment_result,
                    "bias_analysis": bias_result,
                    "relevant_sources": claim_sources,
                    "fact_verification": fact_verification,
                    "analysis_result": claim_analysis
                })
            
            return {
                **state,
                "claim_analysis": claim_analyses,
                "claim_analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in claim analysis: {str(e)}")
            return {
                **state,
                "claim_analysis_status": "failed",
                "error": f"Claim analysis failed: {str(e)}"
            }
    
    def determine_verdicts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine fact-checking verdicts for claims.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with fact-checking verdicts
        """
        try:
            claim_analyses = state.get("claim_analysis", [])
            
            if not claim_analyses:
                logger.warning("No claim analyses available for verdict determination")
                return {
                    **state,
                    "fact_check_verdicts": [],
                    "verdict_determination_status": "completed"
                }
            
            verdicts = []
            
            for analysis in claim_analyses:
                claim_text = analysis.get("claim", "")
                analysis_result = analysis.get("analysis_result", {})
                
                # Determine verdict based on analysis
                verdict = self._determine_verdict(analysis_result)
                
                # Calculate confidence in verdict
                confidence = self._calculate_verdict_confidence(analysis_result)
                
                # Generate reasoning for verdict
                reasoning = self._generate_verdict_reasoning(analysis_result, verdict)
                
                verdicts.append({
                    "claim": claim_text,
                    "verdict": verdict,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "analysis_data": analysis_result,
                    "supporting_evidence": analysis_result.get("supporting_evidence", []),
                    "contradicting_evidence": analysis_result.get("contradicting_evidence", [])
                })
            
            return {
                **state,
                "fact_check_verdicts": verdicts,
                "verdict_determination_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in verdict determination: {str(e)}")
            return {
                **state,
                "verdict_determination_status": "failed",
                "error": f"Verdict determination failed: {str(e)}"
            }
    
    def aggregate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate fact-checking results and generate overall assessment.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with aggregated results
        """
        try:
            fact_check_verdicts = state.get("fact_check_verdicts", [])
            source_credibility = state.get("source_credibility", {})
            aggregated_sources = state.get("aggregated_sources", {})
            
            if not fact_check_verdicts:
                logger.warning("No fact-check verdicts available for aggregation")
                return {
                    **state,
                    "aggregated_results": {},
                    "result_aggregation_status": "completed"
                }
            
            # Count verdicts by type
            verdict_counts = {
                "true": 0,
                "mostly_true": 0,
                "mixed": 0,
                "mostly_false": 0,
                "false": 0,
                "unverifiable": 0,
                "opinion": 0
            }
            
            total_confidence = 0
            total_claims = len(fact_check_verdicts)
            
            for verdict_data in fact_check_verdicts:
                verdict = verdict_data.get("verdict", "unverifiable")
                confidence = verdict_data.get("confidence", 0)
                
                verdict_counts[verdict] += 1
                total_confidence += confidence
            
            # Calculate overall metrics
            average_confidence = total_confidence / total_claims if total_claims > 0 else 0
            
            # Determine overall content verdict
            overall_verdict = self._determine_overall_verdict(verdict_counts, average_confidence)
            
            # Generate overall reasoning
            overall_reasoning = self._generate_overall_reasoning(
                verdict_counts, average_confidence, source_credibility, aggregated_sources
            )
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(
                verdict_counts, average_confidence, source_credibility
            )
            
            aggregated_results = {
                "total_claims_analyzed": total_claims,
                "verdict_distribution": verdict_counts,
                "average_confidence": average_confidence,
                "overall_verdict": overall_verdict,
                "overall_reasoning": overall_reasoning,
                "reliability_score": reliability_score,
                "source_quality": aggregated_sources.get("overall_source_quality", {}),
                "fact_check_summary": {
                    "true_claims_percentage": (verdict_counts["true"] + verdict_counts["mostly_true"]) / total_claims * 100 if total_claims > 0 else 0,
                    "false_claims_percentage": (verdict_counts["false"] + verdict_counts["mostly_false"]) / total_claims * 100 if total_claims > 0 else 0,
                    "mixed_claims_percentage": verdict_counts["mixed"] / total_claims * 100 if total_claims > 0 else 0,
                    "unverifiable_claims_percentage": verdict_counts["unverifiable"] / total_claims * 100 if total_claims > 0 else 0
                }
            }
            
            return {
                **state,
                "aggregated_results": aggregated_results,
                "result_aggregation_status": "completed",
                "next_step": "confidence_scoring"
            }
            
        except Exception as e:
            logger.error(f"Error in result aggregation: {str(e)}")
            return {
                **state,
                "result_aggregation_status": "failed",
                "error": f"Result aggregation failed: {str(e)}"
            }
    
    def _analyze_single_claim(self, claim_text: str, claim_sources: List, fact_verification: Dict, sentiment_result: Dict, bias_result: Dict) -> Dict[str, Any]:
        """
        Analyze a single claim against available evidence.
        
        Args:
            claim_text: The claim to analyze
            claim_sources: Relevant sources for the claim
            fact_verification: Fact database verification results
            sentiment_result: Sentiment analysis results
            bias_result: Bias detection results
            
        Returns:
            Analysis result for the claim
        """
        # Count supporting and contradicting evidence
        supporting_evidence = []
        contradicting_evidence = []
        
        # Analyze sources
        for source in claim_sources:
            source_credibility = source.get("credibility_score", 0)
            if source_credibility > 0.7:  # High credibility threshold
                supporting_evidence.append({
                    "type": "source",
                    "url": source.get("url", ""),
                    "credibility": source_credibility,
                    "relevance": source.get("relevance_score", 0)
                })
        
        # Analyze fact database verification
        verification_status = fact_verification.get("verification_status", "unverified")
        if verification_status == "supported":
            supporting_evidence.append({
                "type": "fact_database",
                "confidence": fact_verification.get("confidence", 0),
                "reasoning": fact_verification.get("reasoning", "")
            })
        elif verification_status == "contradicted":
            contradicting_evidence.append({
                "type": "fact_database",
                "confidence": fact_verification.get("confidence", 0),
                "reasoning": fact_verification.get("reasoning", "")
            })
        
        # Analyze sentiment and bias
        sentiment_score = sentiment_result.get("sentiment", "neutral")
        bias_score = bias_result.get("overall_bias_score", 0)
        
        # Determine evidence strength
        evidence_strength = self._calculate_evidence_strength(supporting_evidence, contradicting_evidence)
        
        return {
            "claim": claim_text,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "evidence_strength": evidence_strength,
            "sentiment": sentiment_score,
            "bias_score": bias_score,
            "total_sources": len(claim_sources),
            "high_credibility_sources": len([s for s in claim_sources if s.get("credibility_score", 0) > 0.7])
        }
    
    def _determine_verdict(self, analysis_result: Dict[str, Any]) -> str:
        """
        Determine fact-checking verdict based on analysis.
        
        Args:
            analysis_result: Analysis result for a claim
            
        Returns:
            Fact-checking verdict
        """
        supporting_evidence = analysis_result.get("supporting_evidence", [])
        contradicting_evidence = analysis_result.get("contradicting_evidence", [])
        evidence_strength = analysis_result.get("evidence_strength", 0)
        bias_score = analysis_result.get("bias_score", 0)
        
        # High bias might indicate opinion rather than fact
        if bias_score > 0.15:
            return "opinion"
        
        # No evidence available
        if not supporting_evidence and not contradicting_evidence:
            return "unverifiable"
        
        # Strong supporting evidence, no contradicting evidence
        if evidence_strength > 0.7 and not contradicting_evidence:
            return "true"
        
        # Some supporting evidence, no contradicting evidence
        if evidence_strength > 0.4 and not contradicting_evidence:
            return "mostly_true"
        
        # Mixed evidence
        if supporting_evidence and contradicting_evidence:
            if evidence_strength > 0.3:
                return "mixed"
            else:
                return "unverifiable"
        
        # Contradicting evidence only
        if contradicting_evidence and not supporting_evidence:
            if evidence_strength > 0.7:
                return "false"
            else:
                return "mostly_false"
        
        # Default to unverifiable
        return "unverifiable"
    
    def _calculate_verdict_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calculate confidence in the verdict.
        
        Args:
            analysis_result: Analysis result for a claim
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        evidence_strength = analysis_result.get("evidence_strength", 0)
        total_sources = analysis_result.get("total_sources", 0)
        high_credibility_sources = analysis_result.get("high_credibility_sources", 0)
        
        # Base confidence on evidence strength
        confidence = evidence_strength * 0.6
        
        # Boost confidence with more sources
        if total_sources > 0:
            source_factor = min(high_credibility_sources / total_sources, 1.0)
            confidence += source_factor * 0.3
        
        # Boost confidence with more total sources
        if total_sources >= 5:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_verdict_reasoning(self, analysis_result: Dict[str, Any], verdict: str) -> str:
        """
        Generate reasoning for the verdict.
        
        Args:
            analysis_result: Analysis result for a claim
            verdict: The determined verdict
            
        Returns:
            Reasoning explanation
        """
        supporting_count = len(analysis_result.get("supporting_evidence", []))
        contradicting_count = len(analysis_result.get("contradicting_evidence", []))
        total_sources = analysis_result.get("total_sources", 0)
        
        if verdict == "true":
            return f"This claim is supported by {supporting_count} credible sources with no contradicting evidence found."
        elif verdict == "mostly_true":
            return f"This claim is mostly accurate, supported by {supporting_count} sources, though evidence is limited."
        elif verdict == "false":
            return f"This claim is contradicted by {contradicting_count} credible sources with no supporting evidence found."
        elif verdict == "mostly_false":
            return f"This claim is mostly inaccurate, contradicted by {contradicting_count} sources."
        elif verdict == "mixed":
            return f"This claim has mixed evidence - {supporting_count} supporting and {contradicting_count} contradicting sources found."
        elif verdict == "opinion":
            return "This appears to be an opinion rather than a factual claim, based on language analysis."
        else:  # unverifiable
            return f"This claim could not be verified with available sources ({total_sources} sources checked)."
    
    def _find_claim_sources(self, claim_text: str, relevant_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find sources relevant to a specific claim.
        
        Args:
            claim_text: The claim to find sources for
            relevant_sources: List of all relevant sources
            
        Returns:
            List of sources relevant to the claim
        """
        claim_words = set(claim_text.lower().split())
        claim_sources = []
        
        for source in relevant_sources:
            source_title = source.get("title", "").lower()
            source_description = source.get("description", "").lower()
            
            # Check for word overlap
            title_words = set(source_title.split())
            desc_words = set(source_description.split())
            
            title_overlap = len(claim_words.intersection(title_words))
            desc_overlap = len(claim_words.intersection(desc_words))
            
            # If there's significant overlap, consider it relevant
            if title_overlap >= 2 or desc_overlap >= 3:
                claim_sources.append(source)
        
        return claim_sources
    
    def _get_claim_fact_verification(self, claim_text: str, fact_database_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get fact database verification for a specific claim.
        
        Args:
            claim_text: The claim to verify
            fact_database_results: Results from fact database check
            
        Returns:
            Verification result for the claim
        """
        for result in fact_database_results:
            if result.get("claim", "") == claim_text:
                return result.get("verification_result", {})
        
        return {"verification_status": "unverified", "confidence": 0.0}
    
    def _calculate_evidence_strength(self, supporting_evidence: List, contradicting_evidence: List) -> float:
        """
        Calculate the strength of evidence.
        
        Args:
            supporting_evidence: List of supporting evidence
            contradicting_evidence: List of contradicting evidence
            
        Returns:
            Evidence strength score (0.0 to 1.0)
        """
        # Calculate supporting evidence strength
        supporting_strength = 0
        for evidence in supporting_evidence:
            if evidence.get("type") == "source":
                supporting_strength += evidence.get("credibility", 0) * evidence.get("relevance", 0)
            elif evidence.get("type") == "fact_database":
                supporting_strength += evidence.get("confidence", 0)
        
        # Calculate contradicting evidence strength
        contradicting_strength = 0
        for evidence in contradicting_evidence:
            if evidence.get("type") == "source":
                contradicting_strength += evidence.get("credibility", 0) * evidence.get("relevance", 0)
            elif evidence.get("type") == "fact_database":
                contradicting_strength += evidence.get("confidence", 0)
        
        # Return the stronger evidence
        return max(supporting_strength, contradicting_strength)
    
    def _determine_overall_verdict(self, verdict_counts: Dict[str, int], average_confidence: float) -> str:
        """
        Determine overall verdict for the content.
        
        Args:
            verdict_counts: Count of each verdict type
            average_confidence: Average confidence across all claims
            
        Returns:
            Overall verdict
        """
        total_claims = sum(verdict_counts.values())
        if total_claims == 0:
            return "unverifiable"
        
        # Calculate percentages
        true_percentage = (verdict_counts["true"] + verdict_counts["mostly_true"]) / total_claims
        false_percentage = (verdict_counts["false"] + verdict_counts["mostly_false"]) / total_claims
        mixed_percentage = verdict_counts["mixed"] / total_claims
        
        # Determine overall verdict
        if true_percentage > 0.7 and false_percentage < 0.2:
            return "true"
        elif true_percentage > 0.5 and false_percentage < 0.3:
            return "mostly_true"
        elif false_percentage > 0.7 and true_percentage < 0.2:
            return "false"
        elif false_percentage > 0.5 and true_percentage < 0.3:
            return "mostly_false"
        elif mixed_percentage > 0.3:
            return "mixed"
        else:
            return "unverifiable"
    
    def _generate_overall_reasoning(self, verdict_counts: Dict[str, int], average_confidence: float, source_credibility: Dict, aggregated_sources: Dict) -> str:
        """
        Generate overall reasoning for the content.
        
        Args:
            verdict_counts: Count of each verdict type
            average_confidence: Average confidence across all claims
            source_credibility: Source credibility information
            aggregated_sources: Aggregated source information
            
        Returns:
            Overall reasoning explanation
        """
        total_claims = sum(verdict_counts.values())
        true_count = verdict_counts["true"] + verdict_counts["mostly_true"]
        false_count = verdict_counts["false"] + verdict_counts["mostly_false"]
        
        reasoning_parts = []
        
        # Add claim analysis summary
        if total_claims > 0:
            reasoning_parts.append(f"Analyzed {total_claims} claims from the content.")
            
            if true_count > 0:
                reasoning_parts.append(f"{true_count} claims were found to be accurate or mostly accurate.")
            
            if false_count > 0:
                reasoning_parts.append(f"{false_count} claims were found to be inaccurate or mostly inaccurate.")
            
            if verdict_counts["mixed"] > 0:
                reasoning_parts.append(f"{verdict_counts['mixed']} claims had mixed evidence.")
        
        # Add source quality information
        source_quality = aggregated_sources.get("overall_source_quality", {})
        source_quality_level = source_quality.get("quality_level", "unknown")
        
        if source_quality_level != "unknown":
            reasoning_parts.append(f"Source quality was assessed as {source_quality_level}.")
        
        # Add confidence information
        if average_confidence > 0.7:
            reasoning_parts.append("High confidence in the analysis results.")
        elif average_confidence > 0.4:
            reasoning_parts.append("Moderate confidence in the analysis results.")
        else:
            reasoning_parts.append("Low confidence in the analysis results due to limited evidence.")
        
        return " ".join(reasoning_parts)
    
    def _calculate_reliability_score(self, verdict_counts: Dict[str, int], average_confidence: float, source_credibility: Dict) -> float:
        """
        Calculate overall reliability score for the content.
        
        Args:
            verdict_counts: Count of each verdict type
            average_confidence: Average confidence across all claims
            source_credibility: Source credibility information
            
        Returns:
            Reliability score (0.0 to 1.0)
        """
        total_claims = sum(verdict_counts.values())
        if total_claims == 0:
            return 0.0
        
        # Calculate accuracy percentage
        true_count = verdict_counts["true"] + verdict_counts["mostly_true"]
        accuracy_percentage = true_count / total_claims
        
        # Factor in source credibility
        source_credibility_score = source_credibility.get("credibility_score", 0.5)
        
        # Calculate reliability score
        reliability_score = (
            accuracy_percentage * 0.5 +
            average_confidence * 0.3 +
            source_credibility_score * 0.2
        )
        
        return min(1.0, max(0.0, reliability_score))
