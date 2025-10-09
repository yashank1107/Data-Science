"""
Web Tools module using Serper API for search and requests for scraping.
All SSL issues bypassed - no urllib/urlopen usage.
"""

import logging
import time
import ssl
import urllib3
import os
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse, quote_plus
import re

# Disable SSL warnings and verification GLOBALLY
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

import requests
from bs4 import BeautifulSoup

import config
from src.utils import clean_text, truncate_text, validate_url, Timer

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tool for searching the web using Serper API (bypasses SSL issues)."""

    def __init__(self, enabled: bool = False, max_results: int = None):
        self.enabled = enabled
        self.max_results = max_results or config.MAX_SEARCH_RESULTS
        self.api_key = config.SERPER_API_KEY
        self.api_url = "https://google.serper.dev/search"

        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        if not self.api_key:
            logger.warning("Serper API key not configured - search will be limited")

        logger.info(f"Initialized WebSearchTool with Serper API (enabled={self.enabled}, max_results={self.max_results})")

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """Search using Serper API only if enabled."""
        if not self.enabled:
            logger.info("Serper search is disabled for this feature/model.")
            return []
        if not self.api_key:
            logger.error("Serper API key not configured")
            return []
        num_results = max_results or self.max_results
        try:
            payload = {
                "q": query,
                "num": num_results,
                "gl": "us",
                "hl": "en"
            }
            logger.info(f"Searching Serper for: {query}")
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=15,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            organic_results = data.get('organic', [])
            results = []
            for result in organic_results[:num_results]:
                title = result.get('title', 'No title')
                link = result.get('link', '')
                snippet = result.get('snippet', 'No description available')
                if title and link:
                    results.append({
                        "title": clean_text(title),
                        "url": link,
                        "snippet": clean_text(snippet),
                        "source": urlparse(link).netloc if link else "Unknown"
                    })
            logger.info(f"✓ Found {len(results)} results from Serper")
            return results
        except Exception as e:
            logger.error(f"Serper API request failed: {str(e)}")
            return []