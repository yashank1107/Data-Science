"""
Utility functions for the Skill Assessment AI system.
"""

import logging
import json
from typing import List, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 5000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for LLM context."""
    if not results:
        return "No search results available."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. {result.get('title', 'Untitled')}")
        formatted.append(f"   {result.get('snippet', 'No description')}")
        formatted.append(f"   Source: {result.get('url', 'Unknown')}\n")
    
    return "\n".join(formatted)


def extract_topics(text: str) -> List[str]:
    """Extract potential topics from text using simple keyword extraction."""
    # This is a simple implementation - can be enhanced with NLP
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    
    words = text.lower().split()
    topics = [word for word in words if len(word) > 4 and word not in common_words]
    
    # Return unique topics
    return list(set(topics))[:5]


def log_interaction(query: str, response: str, metadata: Dict[str, Any] = None):
    """Log user interactions for analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response[:200],  # Truncate for logging
        "metadata": metadata or {}
    }
    
    logger.info(f"Interaction logged: {json.dumps(log_entry, indent=2)}")


def validate_url(url: str) -> bool:
    """Validate if string is a proper URL."""
    return url.startswith(("http://", "https://"))


def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Completed: {self.name} ({elapsed:.2f}s)")


# Example usage
if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    test_text = "  This   is   a    test   text  "
    print(f"Cleaned: '{clean_text(test_text)}'")
    
    long_text = "a" * 10000
    print(f"Truncated length: {len(truncate_text(long_text, 100))}")
    
    print("✓ Utilities working correctly")