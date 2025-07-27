#!/usr/bin/env python3
"""
Test script for model configuration.

This script tests that the AI model configuration is working correctly
with the updated model name gpt-4.1-mini-2025-04-14.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from app.config import settings
from app.workflow.nodes import BaseNode
from app.workflow.workflows import FactCheckingWorkflow, ContentAnalysisWorkflow, SourceVerificationWorkflow
from app.services.ai_service import ModelManager


def test_model_configuration():
    """Test that the model configuration is working correctly."""
    logger.info("Testing model configuration...")
    
    # Test settings
    logger.info(f"OpenAI API Key configured: {bool(settings.ai.OPENAI_API_KEY)}")
    logger.info(f"Default OpenAI model: {settings.ai.OPENAI_MODEL}")
    
    # Test ModelManager
    model_manager = ModelManager()
    logger.info(f"Available models: {model_manager.get_available_models()}")
    logger.info(f"Default model: {model_manager.default_model}")
    
    # Test BaseNode
    base_node = BaseNode()
    logger.info(f"BaseNode model name: {base_node.ai_model_name}")
    
    # Test workflow classes
    fact_checking = FactCheckingWorkflow()
    logger.info(f"FactCheckingWorkflow model: {fact_checking.ai_model_name}")
    
    content_analysis = ContentAnalysisWorkflow()
    logger.info(f"ContentAnalysisWorkflow model: {content_analysis.ai_model_name}")
    
    source_verification = SourceVerificationWorkflow()
    logger.info(f"SourceVerificationWorkflow model: {source_verification.ai_model_name}")
    
    # Test model initialization
    if settings.ai.OPENAI_API_KEY:
        try:
            model = base_node.model
            if model:
                logger.info("✅ Model initialization successful")
            else:
                logger.error("❌ Model initialization failed")
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
    else:
        logger.warning("⚠️ No OpenAI API key configured - skipping model test")


def test_model_names():
    """Test that all model names are correctly set."""
    logger.info("Testing model names...")
    
    expected_model = "gpt-4.1-mini-2025-04-14"
    
    # Check settings
    assert settings.ai.OPENAI_MODEL == expected_model, f"Settings model mismatch: {settings.ai.OPENAI_MODEL}"
    logger.info("✅ Settings model name correct")
    
    # Check workflow classes
    workflows = [
        FactCheckingWorkflow(),
        ContentAnalysisWorkflow(),
        SourceVerificationWorkflow()
    ]
    
    for i, workflow in enumerate(workflows):
        assert workflow.ai_model_name == expected_model, f"Workflow {i} model mismatch: {workflow.ai_model_name}"
    
    logger.info("✅ All workflow model names correct")
    
    # Check base node
    base_node = BaseNode()
    assert base_node.ai_model_name == expected_model, f"BaseNode model mismatch: {base_node.ai_model_name}"
    logger.info("✅ BaseNode model name correct")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Model Configuration Test")
    logger.info("=" * 60)
    
    try:
        test_model_configuration()
        print()
        test_model_names()
        print()
        logger.info("✅ All model configuration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1) 