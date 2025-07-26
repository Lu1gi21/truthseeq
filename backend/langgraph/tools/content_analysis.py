"""
Content analysis tools for LangGraph workflows.

This module provides tools for analyzing content text, including
sentiment analysis, bias detection, and content quality assessment.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from collections import Counter

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SentimentAnalysisInput(BaseModel):
    """Input schema for sentiment analysis."""
    
    text: str = Field(
        ..., 
        description="Text to analyze for sentiment",
        example="This is a positive statement about the new policy."
    )
    analysis_type: str = Field(
        default="general",
        description="Type of sentiment analysis (general, political, medical, etc.)",
        example="general"
    )


class SentimentAnalysisTool(BaseTool):
    """
    Tool for analyzing sentiment in text content.
    
    This tool provides sentiment analysis capabilities to identify
    emotional tone, bias, and subjective language in content.
    """
    
    name: str = "sentiment_analysis"
    description: str = """
    Analyze the sentiment and emotional tone of text content.
    Use this tool to identify positive, negative, or neutral sentiment
    and detect potential bias in the language used.
    """
    args_schema: type[BaseModel] = SentimentAnalysisInput
    
    def __init__(self):
        """Initialize the sentiment analysis tool."""
        super().__init__()
        # Load sentiment dictionaries (in production, use proper NLP libraries)
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.bias_indicators = self._load_bias_indicators()
    
    def _run(
        self,
        text: str,
        analysis_type: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sentiment in the provided text.
        
        Args:
            text: Text to analyze
            analysis_type: Type of sentiment analysis
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Basic sentiment analysis using word lists
            words = re.findall(r'\b\w+\b', text.lower())
            
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            bias_count = sum(1 for word in words if word in self.bias_indicators)
            
            total_words = len(words)
            
            # Calculate sentiment scores
            positive_score = positive_count / total_words if total_words > 0 else 0
            negative_score = negative_count / total_words if total_words > 0 else 0
            bias_score = bias_count / total_words if total_words > 0 else 0
            
            # Determine overall sentiment
            if positive_score > negative_score + 0.02:
                sentiment = "positive"
                confidence = min(0.9, positive_score * 2)
            elif negative_score > positive_score + 0.02:
                sentiment = "negative"
                confidence = min(0.9, negative_score * 2)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            # Detect bias indicators
            bias_indicators_found = [word for word in words if word in self.bias_indicators]
            
            return {
                "text_length": len(text),
                "word_count": total_words,
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "bias_score": bias_score,
                "bias_indicators": bias_indicators_found,
                "analysis_type": analysis_type,
                "details": {
                    "positive_words": positive_count,
                    "negative_words": negative_count,
                    "bias_indicators": bias_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "error": str(e),
                "sentiment": "unknown",
                "confidence": 0.0
            }
    
    def _load_positive_words(self) -> set:
        """Load positive sentiment words."""
        return {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "positive", "beneficial", "helpful", "successful", "effective",
            "safe", "reliable", "trustworthy", "accurate", "true", "proven",
            "improved", "better", "best", "outstanding", "superior", "quality"
        }
    
    def _load_negative_words(self) -> set:
        """Load negative sentiment words."""
        return {
            "bad", "terrible", "awful", "horrible", "dangerous", "harmful",
            "negative", "harmful", "damaging", "risky", "unsafe", "unreliable",
            "false", "fake", "hoax", "conspiracy", "scam", "fraud", "deceptive",
            "misleading", "inaccurate", "wrong", "failed", "worse", "worst"
        }
    
    def _load_bias_indicators(self) -> set:
        """Load bias indicator words."""
        return {
            "obviously", "clearly", "undoubtedly", "certainly", "definitely",
            "absolutely", "always", "never", "everyone", "nobody", "all",
            "none", "completely", "totally", "extremely", "very", "really",
            "obviously", "clearly", "undoubtedly", "certainly", "definitely"
        }


class BiasDetectionTool(BaseTool):
    """
    Tool for detecting bias in text content.
    
    This tool analyzes text for various types of bias including
    political bias, confirmation bias, and language bias.
    """
    
    name: str = "bias_detection"
    description: str = """
    Detect various types of bias in text content.
    Use this tool to identify political bias, confirmation bias,
    language bias, and other forms of subjective language.
    """
    
    def __init__(self):
        """Initialize the bias detection tool."""
        super().__init__()
        self.political_indicators = self._load_political_indicators()
        self.confirmation_indicators = self._load_confirmation_indicators()
        self.emotional_indicators = self._load_emotional_indicators()
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Detect bias in the provided text.
        
        Args:
            text: Text to analyze for bias
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with bias detection results
        """
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Count different types of bias indicators
            political_count = sum(1 for word in words if word in self.political_indicators)
            confirmation_count = sum(1 for word in words if word in self.confirmation_indicators)
            emotional_count = sum(1 for word in words if word in self.emotional_indicators)
            
            total_words = len(words)
            
            # Calculate bias scores
            political_bias = political_count / total_words if total_words > 0 else 0
            confirmation_bias = confirmation_count / total_words if total_words > 0 else 0
            emotional_bias = emotional_count / total_words if total_words > 0 else 0
            
            # Overall bias score
            overall_bias = (political_bias + confirmation_bias + emotional_bias) / 3
            
            # Determine bias level
            if overall_bias > 0.1:
                bias_level = "high"
            elif overall_bias > 0.05:
                bias_level = "medium"
            else:
                bias_level = "low"
            
            # Identify specific bias types
            bias_types = []
            if political_bias > 0.05:
                bias_types.append("political")
            if confirmation_bias > 0.05:
                bias_types.append("confirmation")
            if emotional_bias > 0.05:
                bias_types.append("emotional")
            
            return {
                "text_length": len(text),
                "word_count": total_words,
                "overall_bias_score": overall_bias,
                "bias_level": bias_level,
                "bias_types": bias_types,
                "political_bias": political_bias,
                "confirmation_bias": confirmation_bias,
                "emotional_bias": emotional_bias,
                "details": {
                    "political_indicators": political_count,
                    "confirmation_indicators": confirmation_count,
                    "emotional_indicators": emotional_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in bias detection: {str(e)}")
            return {
                "error": str(e),
                "overall_bias_score": 0.0,
                "bias_level": "unknown"
            }
    
    def _load_political_indicators(self) -> set:
        """Load political bias indicators."""
        return {
            "liberal", "conservative", "democrat", "republican", "left", "right",
            "progressive", "traditional", "radical", "establishment", "elite",
            "mainstream", "alternative", "conspiracy", "deep state", "swamp"
        }
    
    def _load_confirmation_indicators(self) -> set:
        """Load confirmation bias indicators."""
        return {
            "obviously", "clearly", "undoubtedly", "certainly", "definitely",
            "absolutely", "proves", "confirms", "shows", "demonstrates",
            "evidence", "fact", "truth", "reality", "obvious", "clear"
        }
    
    def _load_emotional_indicators(self) -> set:
        """Load emotional bias indicators."""
        return {
            "amazing", "terrible", "horrible", "wonderful", "fantastic",
            "outrageous", "shocking", "incredible", "unbelievable", "ridiculous",
            "absurd", "ludicrous", "preposterous", "outrageous", "scandalous"
        }


class ContentQualityTool(BaseTool):
    """
    Tool for assessing content quality and readability.
    
    This tool analyzes content for quality indicators including
    readability, structure, and content completeness.
    """
    
    name: str = "content_quality_assessment"
    description: str = """
    Assess the quality and readability of content.
    Use this tool to evaluate content structure, readability,
    and overall quality indicators.
    """
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Assess the quality of the provided text.
        
        Args:
            text: Text to assess for quality
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with quality assessment results
        """
        try:
            # Basic quality metrics
            sentences = re.split(r'[.!?]+', text)
            words = re.findall(r'\b\w+\b', text.lower())
            paragraphs = text.split('\n\n')
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Average sentence length
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Average paragraph length
            avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
            
            # Readability score (simplified Flesch Reading Ease)
            # Higher score = easier to read
            if word_count > 0 and sentence_count > 0:
                syllables = self._count_syllables(text)
                readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * syllables / word_count)
                readability_score = max(0, min(100, readability_score))
            else:
                readability_score = 0
            
            # Content structure assessment
            structure_score = self._assess_structure(text)
            
            # Content completeness
            completeness_score = self._assess_completeness(text)
            
            # Overall quality score
            quality_score = (readability_score + structure_score + completeness_score) / 3
            
            return {
                "text_length": len(text),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_sentence_length": avg_sentence_length,
                "avg_paragraph_length": avg_paragraph_length,
                "readability_score": readability_score,
                "structure_score": structure_score,
                "completeness_score": completeness_score,
                "overall_quality_score": quality_score,
                "quality_level": self._get_quality_level(quality_score),
                "details": {
                    "syllables": self._count_syllables(text),
                    "unique_words": len(set(words)),
                    "vocabulary_diversity": len(set(words)) / word_count if word_count > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in content quality assessment: {str(e)}")
            return {
                "error": str(e),
                "overall_quality_score": 0.0,
                "quality_level": "unknown"
            }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified method)."""
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            # Simplified syllable counting
            vowels = len(re.findall(r'[aeiouy]', word))
            if vowels == 0:
                syllable_count += 1
            else:
                syllable_count += vowels
        
        return syllable_count
    
    def _assess_structure(self, text: str) -> float:
        """Assess content structure quality."""
        score = 50.0  # Base score
        
        # Check for paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 20
        
        # Check for sentence variety
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(set(sentence_lengths)) > 3:
            score += 15
        
        # Check for proper punctuation
        if text.count('.') + text.count('!') + text.count('?') > 0:
            score += 15
        
        return min(100, score)
    
    def _assess_completeness(self, text: str) -> float:
        """Assess content completeness."""
        score = 50.0  # Base score
        
        # Check for minimum content length
        if len(text) > 100:
            score += 20
        
        if len(text) > 500:
            score += 20
        
        # Check for variety in vocabulary
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = len(set(words))
        if unique_words > 20:
            score += 10
        
        return min(100, score)
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"


class ClaimExtractionTool(BaseTool):
    """
    Tool for extracting claims and statements from text content.
    
    This tool identifies factual claims, statements, and assertions
    that can be fact-checked in content.
    """
    
    name: str = "claim_extraction"
    description: str = """
    Extract factual claims and statements from text content.
    Use this tool to identify specific claims that can be fact-checked
    or verified against reliable sources.
    """
    
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Extract claims from the provided text.
        
        Args:
            text: Text to extract claims from
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with extracted claims
        """
        try:
            sentences = re.split(r'[.!?]+', text)
            claims = []
            
            # Claim indicators
            claim_indicators = [
                "is", "are", "was", "were", "has", "have", "had",
                "shows", "proves", "demonstrates", "indicates", "suggests",
                "reveals", "confirms", "finds", "discovered", "found",
                "according to", "research shows", "studies show", "evidence shows"
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if sentence contains claim indicators
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in claim_indicators):
                    # Extract the claim
                    claim = self._extract_claim_from_sentence(sentence)
                    if claim:
                        claims.append({
                            "claim": claim,
                            "sentence": sentence,
                            "confidence": self._assess_claim_confidence(sentence)
                        })
            
            return {
                "text_length": len(text),
                "total_claims": len(claims),
                "claims": claims,
                "claim_density": len(claims) / len([s for s in sentences if s.strip()]) if sentences else 0
            }
            
        except Exception as e:
            logger.error(f"Error in claim extraction: {str(e)}")
            return {
                "error": str(e),
                "total_claims": 0,
                "claims": []
            }
    
    def _extract_claim_from_sentence(self, sentence: str) -> Optional[str]:
        """Extract the main claim from a sentence."""
        # Simple claim extraction - in production, use more sophisticated NLP
        sentence = sentence.strip()
        
        # Remove common prefixes
        prefixes = [
            "according to", "research shows", "studies show", "evidence shows",
            "it is", "this is", "that is", "there is", "there are"
        ]
        
        for prefix in prefixes:
            if sentence.lower().startswith(prefix):
                sentence = sentence[len(prefix):].strip()
                break
        
        return sentence if len(sentence) > 10 else None
    
    def _assess_claim_confidence(self, sentence: str) -> float:
        """Assess confidence in the extracted claim."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for stronger indicators
        strong_indicators = ["proves", "confirms", "demonstrates", "shows"]
        weak_indicators = ["suggests", "indicates", "might", "could", "may"]
        
        sentence_lower = sentence.lower()
        
        if any(indicator in sentence_lower for indicator in strong_indicators):
            confidence += 0.3
        elif any(indicator in sentence_lower for indicator in weak_indicators):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
