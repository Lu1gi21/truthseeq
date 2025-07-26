"""
Fact database tools for LangGraph workflows.

This module provides tools for accessing fact-checking databases and
known facts that can be used to verify claims and statements.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FactDatabaseInput(BaseModel):
    """Input schema for fact database search."""
    
    query: str = Field(
        ..., 
        description="Fact or claim to search for in the database",
        example="COVID-19 vaccine effectiveness"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20
    )
    fact_type: Optional[str] = Field(
        default=None,
        description="Type of fact to search for (medical, political, scientific, etc.)",
        example="medical"
    )


class FactDatabaseTool(BaseTool):
    """
    Tool for searching fact-checking databases and known facts.
    
    This tool provides access to curated fact-checking databases,
    scientific papers, and verified information that can be used
    to verify claims and statements.
    """
    
    name: str = "fact_database_search"
    description: str = """
    Search fact-checking databases and verified information sources.
    Use this tool to find authoritative, verified facts about topics
    or to check if a claim has been previously fact-checked.
    """
    args_schema: type[BaseModel] = FactDatabaseInput
    
    def __init__(self):
        """Initialize the fact database tool."""
        super().__init__()
        # Initialize with some sample fact-checking data
        # In production, this would connect to a real database
        self.fact_database = self._initialize_fact_database()
    
    def _run(
        self,
        query: str,
        max_results: int = 5,
        fact_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the fact database for relevant information.
        
        Args:
            query: Fact or claim to search for
            max_results: Maximum number of results to return
            fact_type: Type of fact to search for
            run_manager: LangChain callback manager
            
        Returns:
            List of relevant facts and fact-checks
        """
        try:
            # Simple keyword-based search (in production, use proper search engine)
            results = []
            query_lower = query.lower()
            
            for fact in self.fact_database:
                # Check if query matches fact content
                if (query_lower in fact["content"].lower() or 
                    query_lower in fact["title"].lower() or
                    any(keyword in query_lower for keyword in fact.get("keywords", []))):
                    
                    # Filter by fact type if specified
                    if fact_type and fact.get("type") != fact_type:
                        continue
                    
                    results.append(fact)
                    
                    if len(results) >= max_results:
                        break
            
            logger.info(f"Fact database search completed for query '{query}' with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in fact database search: {str(e)}")
            return []
    
    def _initialize_fact_database(self) -> List[Dict[str, Any]]:
        """
        Initialize the fact database with sample data.
        
        In production, this would load from a real database or API.
        
        Returns:
            List of fact-checking entries
        """
        return [
            {
                "id": "fc_001",
                "title": "COVID-19 Vaccines Are Safe and Effective",
                "content": "Multiple large-scale studies have confirmed that COVID-19 vaccines are safe and highly effective at preventing severe illness, hospitalization, and death. The vaccines have been tested in clinical trials involving tens of thousands of participants.",
                "verdict": "true",
                "confidence": 0.95,
                "type": "medical",
                "keywords": ["covid", "vaccine", "safety", "effectiveness"],
                "sources": [
                    "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/safety.html",
                    "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/covid-19-vaccines"
                ],
                "last_updated": "2024-01-15"
            },
            {
                "id": "fc_002",
                "title": "Climate Change Is Caused by Human Activities",
                "content": "The overwhelming scientific consensus is that climate change is primarily caused by human activities, particularly the burning of fossil fuels which releases greenhouse gases into the atmosphere.",
                "verdict": "true",
                "confidence": 0.98,
                "type": "scientific",
                "keywords": ["climate change", "global warming", "human activities", "fossil fuels"],
                "sources": [
                    "https://climate.nasa.gov/evidence/",
                    "https://www.ipcc.ch/reports/"
                ],
                "last_updated": "2024-01-10"
            },
            {
                "id": "fc_003",
                "title": "5G Networks Do Not Cause COVID-19",
                "content": "There is no scientific evidence that 5G networks cause COVID-19 or any other health problems. COVID-19 is caused by the SARS-CoV-2 virus, not by radio waves.",
                "verdict": "false",
                "confidence": 0.99,
                "type": "medical",
                "keywords": ["5g", "covid", "radio waves", "virus"],
                "sources": [
                    "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
                    "https://www.fda.gov/radiation-emitting-products/5g-and-cell-phone-safety"
                ],
                "last_updated": "2024-01-05"
            },
            {
                "id": "fc_004",
                "title": "The Earth Is Round, Not Flat",
                "content": "The Earth is an oblate spheroid (slightly flattened sphere), not flat. This has been scientifically proven through centuries of observations, measurements, and space exploration.",
                "verdict": "true",
                "confidence": 0.99,
                "type": "scientific",
                "keywords": ["earth", "flat", "round", "spheroid"],
                "sources": [
                    "https://www.nasa.gov/audience/forstudents/k-4/stories/nasa-knows/what-is-earth-k4.html",
                    "https://www.space.com/17638-how-big-is-earth.html"
                ],
                "last_updated": "2024-01-01"
            },
            {
                "id": "fc_005",
                "title": "Vaccines Do Not Cause Autism",
                "content": "Multiple large-scale studies have found no link between vaccines and autism. The original study that suggested this link has been thoroughly discredited and retracted.",
                "verdict": "true",
                "confidence": 0.97,
                "type": "medical",
                "keywords": ["vaccines", "autism", "link", "studies"],
                "sources": [
                    "https://www.cdc.gov/vaccinesafety/concerns/autism.html",
                    "https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders"
                ],
                "last_updated": "2024-01-12"
            }
        ]


class FactVerificationTool(BaseTool):
    """
    Tool for verifying specific claims against known facts.
    
    This tool compares claims to verified information in the fact database
    and provides confidence scores and reasoning for the verification.
    """
    
    name: str = "fact_verification"
    description: str = """
    Verify a specific claim against known facts and fact-checking databases.
    Use this tool to check if a claim is supported by verified information
    or if it contradicts established facts.
    """
    
    def __init__(self, fact_database_tool: Optional[FactDatabaseTool] = None):
        """Initialize the fact verification tool."""
        super().__init__()
        self.fact_database_tool = fact_database_tool or FactDatabaseTool()
    
    def _run(
        self,
        claim: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Verify a specific claim against the fact database.
        
        Args:
            claim: The claim to verify
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Search the fact database for relevant information
            search_results = self.fact_database_tool._run(
                query=claim,
                max_results=10
            )
            
            if not search_results:
                return {
                    "claim": claim,
                    "verification_status": "unverified",
                    "confidence": 0.0,
                    "reasoning": "No relevant facts found in database",
                    "supporting_facts": [],
                    "contradicting_facts": []
                }
            
            # Analyze the results to determine verification status
            supporting_facts = []
            contradicting_facts = []
            
            for fact in search_results:
                # Simple keyword matching for relevance
                claim_lower = claim.lower()
                fact_content_lower = fact["content"].lower()
                
                # Check if the fact supports or contradicts the claim
                if fact["verdict"] == "true" and self._is_supporting(claim_lower, fact_content_lower):
                    supporting_facts.append(fact)
                elif fact["verdict"] == "false" and self._is_contradicting(claim_lower, fact_content_lower):
                    contradicting_facts.append(fact)
            
            # Determine overall verification status
            if supporting_facts and not contradicting_facts:
                verification_status = "supported"
                confidence = max(fact["confidence"] for fact in supporting_facts)
                reasoning = f"Claim is supported by {len(supporting_facts)} verified facts"
            elif contradicting_facts and not supporting_facts:
                verification_status = "contradicted"
                confidence = max(fact["confidence"] for fact in contradicting_facts)
                reasoning = f"Claim is contradicted by {len(contradicting_facts)} verified facts"
            elif supporting_facts and contradicting_facts:
                verification_status = "mixed"
                confidence = 0.5
                reasoning = f"Claim has both supporting ({len(supporting_facts)}) and contradicting ({len(contradicting_facts)}) evidence"
            else:
                verification_status = "unverified"
                confidence = 0.0
                reasoning = "No relevant verified facts found"
            
            return {
                "claim": claim,
                "verification_status": verification_status,
                "confidence": confidence,
                "reasoning": reasoning,
                "supporting_facts": supporting_facts,
                "contradicting_facts": contradicting_facts,
                "total_facts_checked": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error in fact verification: {str(e)}")
            return {
                "claim": claim,
                "verification_status": "error",
                "confidence": 0.0,
                "reasoning": f"Error during verification: {str(e)}",
                "supporting_facts": [],
                "contradicting_facts": []
            }
    
    def _is_supporting(self, claim: str, fact_content: str) -> bool:
        """
        Check if a fact supports a claim.
        
        Args:
            claim: The claim to check
            fact_content: The fact content to compare against
            
        Returns:
            True if the fact supports the claim
        """
        # Simple keyword overlap check
        claim_words = set(claim.split())
        fact_words = set(fact_content.split())
        
        # Check for significant word overlap
        overlap = len(claim_words.intersection(fact_words))
        return overlap >= 2  # At least 2 words in common
    
    def _is_contradicting(self, claim: str, fact_content: str) -> bool:
        """
        Check if a fact contradicts a claim.
        
        Args:
            claim: The claim to check
            fact_content: The fact content to compare against
            
        Returns:
            True if the fact contradicts the claim
        """
        # Similar to supporting check, but for contradicting facts
        return self._is_supporting(claim, fact_content)


class FactUpdateTool(BaseTool):
    """
    Tool for updating the fact database with new information.
    
    This tool allows adding new facts or updating existing ones
    in the fact-checking database.
    """
    
    name: str = "fact_database_update"
    description: str = """
    Update the fact database with new verified information.
    Use this tool to add new facts or update existing ones
    when new verified information becomes available.
    """
    
    def __init__(self, fact_database_tool: Optional[FactDatabaseTool] = None):
        """Initialize the fact update tool."""
        super().__init__()
        self.fact_database_tool = fact_database_tool or FactDatabaseTool()
    
    def _run(
        self,
        title: str,
        content: str,
        verdict: str,
        confidence: float,
        fact_type: str,
        sources: List[str],
        keywords: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Add or update a fact in the database.
        
        Args:
            title: Title of the fact
            content: Content/description of the fact
            verdict: Verdict (true, false, mixed, unverifiable)
            confidence: Confidence score (0.0 to 1.0)
            fact_type: Type of fact (medical, political, scientific, etc.)
            sources: List of source URLs
            keywords: Optional list of keywords
            run_manager: LangChain callback manager
            
        Returns:
            Dictionary with update results
        """
        try:
            # In production, this would update a real database
            # For now, we'll just log the update
            new_fact = {
                "id": f"fc_{len(self.fact_database_tool.fact_database) + 1:03d}",
                "title": title,
                "content": content,
                "verdict": verdict,
                "confidence": confidence,
                "type": fact_type,
                "keywords": keywords or [],
                "sources": sources,
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Add to database (in production, this would be a database insert)
            self.fact_database_tool.fact_database.append(new_fact)
            
            logger.info(f"Fact database updated with new fact: {title}")
            
            return {
                "success": True,
                "fact_id": new_fact["id"],
                "message": f"Fact '{title}' added to database successfully",
                "fact": new_fact
            }
            
        except Exception as e:
            logger.error(f"Error updating fact database: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
