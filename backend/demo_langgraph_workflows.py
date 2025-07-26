#!/usr/bin/env python3
"""
Demo script for LangGraph workflows in TruthSeeQ platform.

This script demonstrates how to use the LangGraph workflows for
fact-checking, content analysis, and source verification.
"""

import asyncio
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the workflows
from langgraph.workflows.fact_checking import create_fact_checking_workflow
from langgraph.workflows.content_analysis import create_content_analysis_workflow
from langgraph.workflows.source_verification import create_source_verification_workflow


def demo_fact_checking_workflow():
    """Demo the fact-checking workflow."""
    print("\n" + "="*60)
    print("DEMO: Fact-Checking Workflow")
    print("="*60)
    
    # Create workflow instance
    workflow = create_fact_checking_workflow()
    
    # Sample content for fact-checking
    sample_content = """
    COVID-19 vaccines are completely safe and have been thoroughly tested. 
    Multiple studies have shown that vaccines are 95% effective at preventing 
    severe illness and hospitalization. The vaccines have been administered to 
    millions of people worldwide with minimal side effects.
    
    However, some people claim that vaccines cause autism, which has been 
    thoroughly debunked by numerous scientific studies. The original study 
    that suggested this link has been retracted and the author lost his medical license.
    """
    
    # Run the workflow
    print("Running fact-checking workflow...")
    result = workflow.run_workflow(
        content_id=1,
        content_text=sample_content,
        source_domain="example.com"
    )
    
    # Display results
    print("\nFact-Checking Results:")
    print("-" * 40)
    
    if result.get("workflow_status") == "completed":
        final_results = result.get("final_results", {})
        
        print(f"Overall Verdict: {final_results.get('overall_verdict', 'unknown')}")
        print(f"Confidence Score: {final_results.get('confidence_score', 0):.2f}")
        print(f"Total Claims Analyzed: {final_results.get('total_claims_analyzed', 0)}")
        
        # Show verdict distribution
        verdict_dist = final_results.get("verdict_distribution", {})
        print("\nVerdict Distribution:")
        for verdict, count in verdict_dist.items():
            if count > 0:
                print(f"  {verdict}: {count}")
        
        # Show recommendations
        recommendations = final_results.get("recommendations", [])
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    else:
        print(f"Workflow failed: {result.get('error', 'Unknown error')}")


def demo_content_analysis_workflow():
    """Demo the content analysis workflow."""
    print("\n" + "="*60)
    print("DEMO: Content Analysis Workflow")
    print("="*60)
    
    # Create workflow instance
    workflow = create_content_analysis_workflow()
    
    # Sample content for analysis
    sample_content = """
    The new government policy is absolutely terrible and will destroy our economy! 
    This is the worst decision ever made by any administration. The politicians 
    who voted for this are completely incompetent and should be removed from office immediately.
    
    Studies show that this policy will lead to increased unemployment and higher taxes. 
    The economic impact will be devastating for small businesses and working families.
    """
    
    # Run the workflow
    print("Running content analysis workflow...")
    result = workflow.run_workflow(
        content_id=2,
        content_text=sample_content
    )
    
    # Display results
    print("\nContent Analysis Results:")
    print("-" * 40)
    
    if result.get("workflow_status") == "completed":
        final_report = result.get("final_report", {})
        
        print(f"Overall Assessment: {final_report.get('overall_assessment', 0):.2f}")
        
        # Content quality
        quality = final_report.get("content_quality", {})
        print(f"\nContent Quality:")
        print(f"  Quality Score: {quality.get('overall_quality_score', 0):.1f}")
        print(f"  Quality Level: {quality.get('quality_level', 'unknown')}")
        print(f"  Readability Score: {quality.get('readability_score', 0):.1f}")
        
        # Sentiment analysis
        sentiment = final_report.get("sentiment_analysis", {})
        print(f"\nSentiment Analysis:")
        print(f"  Sentiment: {sentiment.get('sentiment', 'unknown')}")
        print(f"  Confidence: {sentiment.get('confidence', 0):.2f}")
        
        # Bias analysis
        bias = final_report.get("bias_analysis", {})
        print(f"\nBias Analysis:")
        print(f"  Bias Level: {bias.get('bias_level', 'unknown')}")
        print(f"  Overall Bias Score: {bias.get('overall_bias_score', 0):.2f}")
        
        # Content categorization
        categorization = final_report.get("content_categorization", {})
        print(f"\nContent Categorization:")
        print(f"  Content Type: {categorization.get('content_type', 'unknown')}")
        print(f"  Topic Categories: {', '.join(categorization.get('topic_categories', []))}")
        print(f"  Complexity Level: {categorization.get('complexity_level', 'unknown')}")
        
        # Insights
        insights = final_report.get("insights", [])
        if insights:
            print(f"\nInsights:")
            for insight in insights:
                print(f"  - {insight}")
    else:
        print(f"Workflow failed: {result.get('error', 'Unknown error')}")


def demo_source_verification_workflow():
    """Demo the source verification workflow."""
    print("\n" + "="*60)
    print("DEMO: Source Verification Workflow")
    print("="*60)
    
    # Create workflow instance
    workflow = create_source_verification_workflow()
    
    # Sample source URL and content
    sample_url = "https://www.reuters.com/article/health-coronavirus-vaccines"
    sample_content = """
    Pfizer and Moderna vaccines show high effectiveness in preventing COVID-19.
    Clinical trials demonstrate 95% efficacy rates for both vaccines.
    """
    
    # Run the workflow
    print("Running source verification workflow...")
    result = workflow.run_workflow(
        source_url=sample_url,
        content_text=sample_content
    )
    
    # Display results
    print("\nSource Verification Results:")
    print("-" * 40)
    
    if result.get("workflow_status") == "completed":
        verification_report = result.get("verification_report", {})
        
        print(f"Source URL: {verification_report.get('source_url', 'unknown')}")
        print(f"Overall Credibility Score: {verification_report.get('overall_credibility_score', 0):.2f}")
        print(f"Credibility Level: {verification_report.get('credibility_level', 'unknown')}")
        
        # Source info
        source_info = verification_report.get("source_info", {})
        print(f"\nSource Information:")
        print(f"  Domain: {source_info.get('domain', 'unknown')}")
        print(f"  Protocol: {source_info.get('protocol', 'unknown')}")
        
        # Risk assessment
        risk_assessment = verification_report.get("risk_assessment", {})
        print(f"\nRisk Assessment:")
        print(f"  Overall Risk: {risk_assessment.get('overall_risk', 'unknown')}")
        
        risk_factors = risk_assessment.get("risk_factors", [])
        if risk_factors:
            print(f"  Risk Factors:")
            for factor in risk_factors:
                print(f"    - {factor}")
        
        # Recommendations
        recommendations = verification_report.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    else:
        print(f"Workflow failed: {result.get('error', 'Unknown error')}")


def demo_workflow_integration():
    """Demo how workflows can be integrated together."""
    print("\n" + "="*60)
    print("DEMO: Workflow Integration")
    print("="*60)
    
    # Sample content that would go through multiple workflows
    sample_content = """
    A new study published in Nature shows that climate change is accelerating 
    faster than previously predicted. The research, conducted by leading scientists 
    from multiple universities, indicates that global temperatures could rise by 
    2.5 degrees Celsius by 2050 if current trends continue.
    
    The study analyzed data from over 100 weather stations worldwide and used 
    advanced climate models to make these predictions. The findings have been 
    peer-reviewed and published in one of the world's most prestigious scientific journals.
    """
    
    print("Sample content for multi-workflow analysis:")
    print(f'"{sample_content.strip()}"')
    
    print("\nThis content would be processed through:")
    print("1. Content Analysis Workflow - for quality, sentiment, and bias assessment")
    print("2. Source Verification Workflow - to verify the source credibility")
    print("3. Fact-Checking Workflow - to verify the factual claims")
    
    print("\nThe results would be combined to provide a comprehensive assessment")
    print("of the content's reliability and accuracy.")


def main():
    """Run all workflow demos."""
    print("TruthSeeQ LangGraph Workflows Demo")
    print("=" * 60)
    print("This demo shows the LangGraph workflows for fact-checking,")
    print("content analysis, and source verification.")
    
    try:
        # Run individual workflow demos
        demo_content_analysis_workflow()
        demo_source_verification_workflow()
        demo_fact_checking_workflow()
        
        # Show integration example
        demo_workflow_integration()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main() 