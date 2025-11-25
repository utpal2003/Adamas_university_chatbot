"""
University Chatbot Package
--------------------------
A modular chatbot system for university queries with intelligent content extraction.
"""

__version__ = "1.0.0"
__author__ = "Adamas University"
__description__ = "AI-powered university chatbot with knowledge base integration"

# Import main classes for easier access
from .extractors import SmartContentExtractor, ExpertContentExtractor, HybridContentExtractor
from .utils import ResponseFormatter, IntentRefiner, TelecallerEscalation

# Define what gets imported with "from chatbot import *"
__all__ = [
    'SmartContentExtractor',
    'ExpertContentExtractor', 
    'HybridContentExtractor',
    'ResponseFormatter',
    'IntentRefiner',
    'TelecallerEscalation'
]

# Package initialization
print(f"🤖 University Chatbot Package v{__version__} initialized")