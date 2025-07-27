#!/usr/bin/env python3
"""
TruthSeeQ Workflow Demo

This script demonstrates how to use the TruthSeeQ workflow system for:
- Fact-checking URLs
- Content analysis
- Source verification
- Comprehensive analysis

Usage:
    python demo_workflow.py [URL]

Example:
    python demo_workflow.py "https://example.com/article"
"""

import asyncio
import json
import logging
import sys
from typing import Optional
from datetime import datetime

# Import local modules
from app.workflow.workflows import (
    FactCheckingWorkflow, ContentAnalysisWorkflow, SourceVerificationWorkflow,
    WorkflowManager, fact_check_url, analyze_content, verify_source
)
from app.workflow.state import (
    VerdictType, ConfidenceLevel, create_fact_check_state
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_fact_checking(url: str, model_name: Optional[str] = None):
    """
    Demonstrate fact-checking workflow.
    
    Args:
        url: URL to fact-check
        model_name: Optional AI model name
    """
    print(f"\n🔍 FACT-CHECKING DEMO: {url}")
    print("=" * 60)
    
    try:
        # Create workflow
        workflow = FactCheckingWorkflow(model_name)
        
        # Execute workflow
        print("Starting fact-checking workflow...")
        result = await workflow.execute(url)
        
        # Display results
        print(f"\n✅ Fact-Checking Results:")
        print(f"   Status: {result['status']}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   Model Used: {result.get('model_used', 'Unknown')}")
        
        if result['status'] == 'completed':
            print(f"\n📊 Analysis Summary:")
            print(f"   Confidence Score: {result['confidence_score']:.2%}")
            print(f"   Confidence Level: {result['confidence_level'].value}")
            print(f"   Verdict: {result['verdict'].value}")
            
            print(f"\n📝 Reasoning:")
            print(f"   {result['reasoning']}")
            
            print(f"\n🔍 Claims Analyzed: {len(result['extracted_claims'])}")
            for i, claim in enumerate(result['extracted_claims'][:3], 1):
                print(f"   {i}. {claim['text'][:100]}...")
                if 'accuracy' in claim:
                    print(f"      Accuracy: {claim['accuracy']}")
            
            print(f"\n📚 Sources Found: {len(result['verification_sources'])}")
            for i, source in enumerate(result['verification_sources'][:3], 1):
                print(f"   {i}. {source['url']}")
                print(f"      Credibility: {source['credibility_score']:.2%}")
                print(f"      Type: {source['source_type'].value}")
            
            if result['supporting_evidence']:
                print(f"\n✅ Supporting Evidence:")
                for evidence in result['supporting_evidence'][:2]:
                    print(f"   • {evidence}")
            
            if result['contradicting_evidence']:
                print(f"\n❌ Contradicting Evidence:")
                for evidence in result['contradicting_evidence'][:2]:
                    print(f"   • {evidence}")
        
        else:
            print(f"❌ Workflow failed: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Fact-checking failed: {e}")


async def demo_content_analysis(url: str, model_name: Optional[str] = None):
    """
    Demonstrate content analysis workflow.
    
    Args:
        url: URL to analyze
        model_name: Optional AI model name
    """
    print(f"\n📊 CONTENT ANALYSIS DEMO: {url}")
    print("=" * 60)
    
    try:
        # Create workflow
        workflow = ContentAnalysisWorkflow(model_name)
        
        # Execute workflow
        print("Starting content analysis workflow...")
        result = await workflow.execute(url)
        
        # Display results
        print(f"\n✅ Content Analysis Results:")
        print(f"   Status: {result['status']}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   Model Used: {result.get('model_used', 'Unknown')}")
        
        if result['status'] == 'completed':
            print(f"\n📊 Analysis Summary:")
            print(f"   Content Quality Score: {result['content_quality_score']:.2%}")
            print(f"   Readability Score: {result['readability_score']:.2%}")
            print(f"   Sentiment Score: {result['sentiment_score']:.2f}")
            print(f"   Emotional Tone: {result['emotional_tone']}")
            
            print(f"\n🎯 Content Classification:")
            print(f"   Category: {result['content_category']}")
            print(f"   Subcategory: {result['content_subcategory']}")
            print(f"   Topics: {', '.join(result['topics'][:5])}")
            
            print(f"\n🔍 Bias Analysis:")
            print(f"   Bias Level: {result['bias_level']}")
            print(f"   Bias Types: {', '.join(result['bias_types'])}")
            if result['political_leaning']:
                print(f"   Political Leaning: {result['political_leaning']}")
            
            print(f"\n📝 Key Insights:")
            for insight in result['key_insights'][:3]:
                print(f"   • {insight}")
            
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"   • {rec}")
        
        else:
            print(f"❌ Workflow failed: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Content analysis failed: {e}")


async def demo_source_verification(url: str, model_name: Optional[str] = None):
    """
    Demonstrate source verification workflow.
    
    Args:
        url: URL to verify
        model_name: Optional AI model name
    """
    print(f"\n🔐 SOURCE VERIFICATION DEMO: {url}")
    print("=" * 60)
    
    try:
        # Create workflow
        workflow = SourceVerificationWorkflow(model_name)
        
        # Execute workflow
        print("Starting source verification workflow...")
        result = await workflow.execute(url)
        
        # Display results
        print(f"\n✅ Source Verification Results:")
        print(f"   Status: {result['status']}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   Model Used: {result.get('model_used', 'Unknown')}")
        
        if result['status'] == 'completed':
            print(f"\n🌐 Domain Analysis:")
            print(f"   Domain: {result['domain']}")
            print(f"   Domain Age: {result.get('domain_age', 'Unknown')}")
            print(f"   SSL Valid: {result['ssl_valid']}")
            
            print(f"\n📊 Reputation Analysis:")
            print(f"   Reputation Score: {result['reputation_score']:.2%}")
            print(f"   Verification Score: {result['verification_score']:.2%}")
            print(f"   Verification Status: {result['verification_status']}")
            print(f"   Confidence Level: {result['confidence_level'].value}")
            
            if result['trust_indicators']:
                print(f"\n✅ Trust Indicators:")
                for indicator in result['trust_indicators']:
                    print(f"   • {indicator}")
            
            if result['red_flags']:
                print(f"\n🚩 Red Flags:")
                for flag in result['red_flags']:
                    print(f"   • {flag}")
            
            print(f"\n📝 Verification Summary:")
            print(f"   {result['verification_summary']}")
            
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"   • {rec}")
        
        else:
            print(f"❌ Workflow failed: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Source verification failed: {e}")


async def demo_comprehensive_analysis(url: str, model_name: Optional[str] = None):
    """
    Demonstrate comprehensive analysis using all workflows.
    
    Args:
        url: URL to analyze
        model_name: Optional AI model name
    """
    print(f"\n🚀 COMPREHENSIVE ANALYSIS DEMO: {url}")
    print("=" * 80)
    
    try:
        # Create workflow manager
        manager = WorkflowManager(model_name)
        
        # Execute comprehensive analysis
        print("Starting comprehensive analysis (all workflows in parallel)...")
        result = await manager.execute_comprehensive_analysis(url)
        
        # Display results
        print(f"\n✅ Comprehensive Analysis Results:")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Processing Time: {result['processing_time']:.2f}s")
        print(f"   Timestamp: {result['timestamp']}")
        
        # Fact-checking results
        if 'fact_checking' in result and not isinstance(result['fact_checking'], dict):
            fact_check = result['fact_checking']
            print(f"\n🔍 Fact-Checking Summary:")
            print(f"   Verdict: {fact_check.get('verdict', 'Unknown')}")
            print(f"   Confidence: {fact_check.get('confidence_score', 0):.2%}")
            print(f"   Claims Analyzed: {len(fact_check.get('extracted_claims', []))}")
        
        # Content analysis results
        if 'content_analysis' in result and not isinstance(result['content_analysis'], dict):
            content = result['content_analysis']
            print(f"\n📊 Content Analysis Summary:")
            print(f"   Quality Score: {content.get('content_quality_score', 0):.2%}")
            print(f"   Sentiment: {content.get('emotional_tone', 'Unknown')}")
            print(f"   Category: {content.get('content_category', 'Unknown')}")
        
        # Source verification results
        if 'source_verification' in result and not isinstance(result['source_verification'], dict):
            source = result['source_verification']
            print(f"\n🔐 Source Verification Summary:")
            print(f"   Reputation Score: {source.get('reputation_score', 0):.2%}")
            print(f"   Verification Status: {source.get('verification_status', 'Unknown')}")
            print(f"   Trust Indicators: {len(source.get('trust_indicators', []))}")
        
        # Check for errors
        errors = []
        for workflow_type, result_data in result.items():
            if isinstance(result_data, dict) and 'error' in result_data:
                errors.append(f"{workflow_type}: {result_data['error']}")
        
        if errors:
            print(f"\n❌ Errors encountered:")
            for error in errors:
                print(f"   • {error}")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")


async def demo_convenience_functions(url: str, model_name: Optional[str] = None):
    """
    Demonstrate convenience functions for quick analysis.
    
    Args:
        url: URL to analyze
        model_name: Optional AI model name
    """
    print(f"\n⚡ CONVENIENCE FUNCTIONS DEMO: {url}")
    print("=" * 60)
    
    try:
        print("Using convenience functions for quick analysis...")
        
        # Quick fact-check
        print("\n🔍 Quick Fact-Check:")
        fact_result = await fact_check_url(url, model_name)
        print(f"   Verdict: {fact_result.get('verdict', 'Unknown')}")
        print(f"   Confidence: {fact_result.get('confidence_score', 0):.2%}")
        
        # Quick content analysis
        print("\n📊 Quick Content Analysis:")
        content_result = await analyze_content(url, model_name)
        print(f"   Quality: {content_result.get('content_quality_score', 0):.2%}")
        print(f"   Category: {content_result.get('content_category', 'Unknown')}")
        
        # Quick source verification
        print("\n🔐 Quick Source Verification:")
        source_result = await verify_source(url, model_name)
        print(f"   Reputation: {source_result.get('reputation_score', 0):.2%}")
        print(f"   Status: {source_result.get('verification_status', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")


def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nAvailable demo modes:")
    print("  --fact-check     Run fact-checking workflow only")
    print("  --content        Run content analysis workflow only")
    print("  --source         Run source verification workflow only")
    print("  --comprehensive  Run all workflows in parallel")
    print("  --convenience    Run convenience functions")
    print("  --all            Run all demos (default)")


async def main():
    """Main demo function."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    url = sys.argv[1]
    
    # Check for help
    if url in ['-h', '--help', 'help']:
        print_usage()
        return
    
    # Parse options
    demo_mode = 'all'
    if len(sys.argv) > 2:
        demo_mode = sys.argv[2]
    
    # Model selection (optional)
    model_name = None
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    
    print(f"🚀 TruthSeeQ Workflow Demo")
    print(f"URL: {url}")
    print(f"Mode: {demo_mode}")
    print(f"Model: {model_name or 'Default (GPT-4)'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if demo_mode in ['--fact-check', 'fact-check']:
            await demo_fact_checking(url, model_name)
        elif demo_mode in ['--content', 'content']:
            await demo_content_analysis(url, model_name)
        elif demo_mode in ['--source', 'source']:
            await demo_source_verification(url, model_name)
        elif demo_mode in ['--comprehensive', 'comprehensive']:
            await demo_comprehensive_analysis(url, model_name)
        elif demo_mode in ['--convenience', 'convenience']:
            await demo_convenience_functions(url, model_name)
        elif demo_mode in ['--all', 'all']:
            await demo_fact_checking(url, model_name)
            await demo_content_analysis(url, model_name)
            await demo_source_verification(url, model_name)
            await demo_comprehensive_analysis(url, model_name)
            await demo_convenience_functions(url, model_name)
        else:
            print(f"❌ Unknown demo mode: {demo_mode}")
            print_usage()
            return
        
        print(f"\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main()) 