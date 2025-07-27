"""
Example usage of TruthSeeQ Workflow API endpoints.

This file demonstrates how to interact with the workflow API endpoints
for fact-checking, content analysis, and source verification.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any


class TruthSeeQWorkflowClient:
    """
    Client for interacting with TruthSeeQ Workflow API.
    
    This client provides methods to execute workflows and retrieve results
    from the TruthSeeQ backend API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        """
        Initialize workflow client.
        
        Args:
            base_url: Base URL for the TruthSeeQ API
        """
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fact_check_url(self, url: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Execute fact-checking workflow for a URL.
        
        Args:
            url: URL to fact-check
            model_name: AI model to use
            
        Returns:
            Fact-checking results
        """
        payload = {
            "url": url,
            "model_name": model_name,
            "force_refresh": False
        }
        
        async with self.session.post(
            f"{self.base_url}/workflow/fact-check",
            json=payload
        ) as response:
            return await response.json()
    
    async def analyze_content(self, url: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Execute content analysis workflow for a URL.
        
        Args:
            url: URL to analyze
            model_name: AI model to use
            
        Returns:
            Content analysis results
        """
        payload = {
            "url": url,
            "model_name": model_name,
            "force_refresh": False
        }
        
        async with self.session.post(
            f"{self.base_url}/workflow/content-analysis",
            json=payload
        ) as response:
            return await response.json()
    
    async def verify_source(self, url: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Execute source verification workflow for a URL.
        
        Args:
            url: URL to verify
            model_name: AI model to use
            
        Returns:
            Source verification results
        """
        params = {
            "url": url,
            "model_name": model_name
        }
        
        async with self.session.post(
            f"{self.base_url}/workflow/source-verification",
            params=params
        ) as response:
            return await response.json()
    
    async def comprehensive_analysis(self, url: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Execute comprehensive analysis (all workflows) for a URL.
        
        Args:
            url: URL to analyze
            model_name: AI model to use
            
        Returns:
            Comprehensive analysis results
        """
        params = {
            "url": url,
            "model_name": model_name
        }
        
        async with self.session.post(
            f"{self.base_url}/workflow/comprehensive",
            params=params
        ) as response:
            return await response.json()
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get status of a workflow execution.
        
        Args:
            workflow_id: Workflow ID to check
            
        Returns:
            Workflow status information
        """
        async with self.session.get(
            f"{self.base_url}/workflow/{workflow_id}/status"
        ) as response:
            return await response.json()
    
    async def get_workflow_history(self, skip: int = 0, limit: int = 10) -> Dict[str, Any]:
        """
        Get workflow execution history.
        
        Args:
            skip: Number of items to skip
            limit: Number of items to return
            
        Returns:
            Workflow history
        """
        params = {
            "skip": skip,
            "limit": limit
        }
        
        async with self.session.get(
            f"{self.base_url}/workflow/history",
            params=params
        ) as response:
            return await response.json()
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get workflow performance metrics.
        
        Returns:
            Workflow metrics
        """
        async with self.session.get(
            f"{self.base_url}/workflow/metrics"
        ) as response:
            return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check workflow service health.
        
        Returns:
            Health status
        """
        async with self.session.get(
            f"{self.base_url}/workflow/health"
        ) as response:
            return await response.json()


async def example_usage():
    """Example usage of the workflow API."""
    
    # Example URL to analyze
    test_url = "https://example.com/article"
    
    async with TruthSeeQWorkflowClient() as client:
        print("🔍 TruthSeeQ Workflow API Example")
        print("=" * 50)
        
        # Check service health
        print("\n1. Checking service health...")
        health = await client.health_check()
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   Database: {health.get('database', 'unknown')}")
        print(f"   Redis: {health.get('redis', 'unknown')}")
        
        # Execute fact-checking workflow
        print(f"\n2. Executing fact-checking workflow for {test_url}...")
        try:
            fact_result = await client.fact_check_url(test_url)
            print(f"   Workflow ID: {fact_result.get('workflow_id')}")
            print(f"   Verdict: {fact_result.get('verdict')}")
            print(f"   Confidence: {fact_result.get('confidence_score', 0):.2%}")
            print(f"   Processing Time: {fact_result.get('processing_time', 0):.2f}s")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Execute content analysis workflow
        print(f"\n3. Executing content analysis workflow for {test_url}...")
        try:
            content_result = await client.analyze_content(test_url)
            print(f"   Workflow ID: {content_result.get('workflow_id')}")
            print(f"   Quality Score: {content_result.get('content_quality_score', 0):.2%}")
            print(f"   Readability Score: {content_result.get('readability_score', 0):.2%}")
            print(f"   Content Category: {content_result.get('content_category', 'unknown')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Execute source verification workflow
        print(f"\n4. Executing source verification workflow for {test_url}...")
        try:
            source_result = await client.verify_source(test_url)
            print(f"   Workflow ID: {source_result.get('workflow_id')}")
            print(f"   Reputation Score: {source_result.get('reputation_score', 0):.2%}")
            print(f"   Verification Status: {source_result.get('verification_status', 'unknown')}")
            print(f"   Trust Indicators: {len(source_result.get('trust_indicators', []))}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Execute comprehensive analysis
        print(f"\n5. Executing comprehensive analysis for {test_url}...")
        try:
            comprehensive_result = await client.comprehensive_analysis(test_url)
            print(f"   Workflow ID: {comprehensive_result.get('workflow_id')}")
            print(f"   Processing Time: {comprehensive_result.get('processing_time', 0):.2f}s")
            
            # Check individual workflow results
            if 'fact_checking' in comprehensive_result:
                fact_check = comprehensive_result['fact_checking']
                if isinstance(fact_check, dict) and 'verdict' in fact_check:
                    print(f"   Fact-Checking Verdict: {fact_check['verdict']}")
            
            if 'content_analysis' in comprehensive_result:
                content_analysis = comprehensive_result['content_analysis']
                if isinstance(content_analysis, dict) and 'content_quality_score' in content_analysis:
                    print(f"   Content Quality: {content_analysis['content_quality_score']:.2%}")
            
            if 'source_verification' in comprehensive_result:
                source_verification = comprehensive_result['source_verification']
                if isinstance(source_verification, dict) and 'reputation_score' in source_verification:
                    print(f"   Source Reputation: {source_verification['reputation_score']:.2%}")
                    
        except Exception as e:
            print(f"   Error: {e}")
        
        # Get workflow metrics
        print("\n6. Getting workflow metrics...")
        try:
            metrics = await client.get_workflow_metrics()
            print(f"   Total Workflows: {metrics.get('total_workflows', 0)}")
            print(f"   Completed Workflows: {metrics.get('completed_workflows', 0)}")
            print(f"   Failed Workflows: {metrics.get('failed_workflows', 0)}")
            print(f"   Average Execution Time: {metrics.get('average_execution_time', 0):.2f}s")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Get workflow history
        print("\n7. Getting workflow history...")
        try:
            history = await client.get_workflow_history(limit=5)
            print(f"   Total History Items: {history.get('total', 0)}")
            print(f"   Recent Workflows:")
            for item in history.get('items', [])[:3]:
                print(f"     - {item.get('workflow_type', 'unknown')}: {item.get('status', 'unknown')}")
        except Exception as e:
            print(f"   Error: {e}")


async def example_workflow_status_tracking():
    """Example of tracking workflow status."""
    
    async with TruthSeeQWorkflowClient() as client:
        print("\n🔄 Workflow Status Tracking Example")
        print("=" * 50)
        
        # Start a comprehensive analysis
        test_url = "https://example.com/article"
        print(f"Starting comprehensive analysis for {test_url}...")
        
        try:
            result = await client.comprehensive_analysis(test_url)
            workflow_id = result.get('workflow_id')
            
            if workflow_id:
                print(f"Workflow ID: {workflow_id}")
                
                # Check status
                status = await client.get_workflow_status(workflow_id)
                print(f"Status: {status.get('status', 'unknown')}")
                print(f"Workflow Type: {status.get('workflow_type', 'unknown')}")
                
                if status.get('execution_time'):
                    print(f"Execution Time: {status.get('execution_time'):.2f}s")
                    
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())
    asyncio.run(example_workflow_status_tracking()) 