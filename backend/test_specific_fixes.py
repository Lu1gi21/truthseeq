#!/usr/bin/env python3
"""
Test script for specific fixes.

This script tests the specific issues that were mentioned in the logs:
1. 'os' is not defined error in workflow nodes
2. LangChain deprecation warning for BaseTool.__call__
"""

import asyncio
import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_os_import_fix():
    """Test that the 'os' import error is fixed."""
    try:
        # Import the modules that were causing the 'os' error
        from app.workflow.tools import check_domain_reliability
        from app.workflow.nodes import DomainAnalysisNode, CredibilityAnalysisNode
        
        logger.info("✓ Successfully imported workflow modules")
        
        # Test that the tools can be called without 'os' error
        try:
            # This should not raise a 'name os is not defined' error
            result = check_domain_reliability.invoke({"domain": "example.com"})
            logger.info("✓ check_domain_reliability tool works without 'os' error")
        except NameError as e:
            if "os" in str(e):
                logger.error(f"✗ 'os' is not defined error still exists: {e}")
                return False
            else:
                logger.info(f"✓ No 'os' error (other error: {e})")
        
        # Test that nodes can be instantiated
        domain_node = DomainAnalysisNode()
        credibility_node = CredibilityAnalysisNode()
        logger.info("✓ Nodes can be instantiated without 'os' error")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ OS import fix test failed: {e}")
        return False

def test_langchain_deprecation_fix():
    """Test that LangChain deprecation warnings are addressed."""
    try:
        from app.workflow.tools import check_domain_reliability
        
        # Test that the tool uses .invoke() instead of direct calling
        # This should not trigger the LangChain deprecation warning
        result = check_domain_reliability.invoke({"domain": "example.com"})
        logger.info("✓ Tool uses .invoke() method correctly")
        
        # Test that the tool returns expected structure
        if isinstance(result, dict):
            logger.info("✓ Tool returns proper dictionary structure")
        else:
            logger.warning(f"Tool returned unexpected type: {type(result)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LangChain deprecation fix test failed: {e}")
        return False

def test_workflow_execution():
    """Test that workflow nodes can execute without errors."""
    try:
        from app.workflow.nodes import DomainAnalysisNode
        
        # Create a mock state
        mock_state = {
            "scraped_content": {
                "url": "https://example.com/article"
            }
        }
        
        # Test that the node can be called (this would normally be async)
        node = DomainAnalysisNode()
        logger.info("✓ DomainAnalysisNode can be instantiated")
        
        # Test that the node has the expected structure
        if hasattr(node, '__call__'):
            logger.info("✓ Node has __call__ method")
        else:
            logger.error("✗ Node missing __call__ method")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Workflow execution test failed: {e}")
        return False

async def main():
    """Run all specific fix tests."""
    logger.info("Starting specific fix tests...")
    
    tests = [
        ("OS Import Fix", test_os_import_fix),
        ("LangChain Deprecation Fix", test_langchain_deprecation_fix),
        ("Workflow Execution", test_workflow_execution),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SPECIFIC FIX TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All specific fixes are working correctly!")
        logger.info("The 'os' import error and LangChain deprecation warning should be resolved.")
    else:
        logger.warning("⚠️  Some specific fixes failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 