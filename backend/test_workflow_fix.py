#!/usr/bin/env python3
"""
Test script to verify workflow fixes work correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.workflow.workflows import FactCheckingWorkflow
from app.workflow.state import create_fact_check_state

async def test_workflow():
    """Test the fact-checking workflow with a simple URL."""
    
    print("Testing fact-checking workflow...")
    
    # Create workflow instance
    workflow = FactCheckingWorkflow("gpt-4")
    
    # Test URL
    test_url = "https://example.com"
    
    try:
        # Execute workflow
        result = await workflow.execute(test_url, workflow_id="test_workflow")
        
        print(f"Workflow completed with status: {result.get('status')}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        
        if result.get('status') == 'completed':
            print("✅ Workflow executed successfully!")
        elif result.get('status') == 'error':
            print(f"❌ Workflow failed: {result.get('error_message')}")
        else:
            print(f"⚠️  Workflow status: {result.get('status')}")
            
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow()) 