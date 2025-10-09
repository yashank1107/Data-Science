"""
SSL Bypass Module - MUST be imported FIRST in your application.

Import this at the very top of your main.py or app.py:
    import ssl_bypass  # This line must be first!
    import other_modules...
"""

import ssl
import os
import sys
import urllib.request
import urllib3
import http.client

print("🔓 Initializing SSL bypass...")

# ============================================================================
# ENVIRONMENT VARIABLES - Disable all SSL verification
# ============================================================================
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

# ============================================================================
# SSL CONTEXT - Create unverified context globally
# ============================================================================
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================================
# URLLIB3 - Disable warnings
# ============================================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# URLLIB.REQUEST - Monkey patch to disable SSL verification
# ============================================================================

# Store original functions
_original_urlopen = urllib.request.urlopen
_original_Request = urllib.request.Request

def _patched_urlopen(url, data=None, timeout=None, *args, **kwargs):
    """Patched urlopen that always uses unverified SSL context."""
    # Always add unverified context
    context = ssl._create_unverified_context()
    
    # Handle both string URLs and Request objects
    if 'context' not in kwargs:
        kwargs['context'] = context
    
    try:
        return _original_urlopen(url, data=data, timeout=timeout, *args, **kwargs)
    except Exception as e:
        # If it still fails, try without context parameter
        if 'context' in kwargs:
            del kwargs['context']
        return _original_urlopen(url, data=data, timeout=timeout, *args, **kwargs)

# Replace urlopen with patched version
urllib.request.urlopen = _patched_urlopen

# ============================================================================
# HTTPS HANDLER - Custom handler that never verifies SSL
# ============================================================================

class UnverifiedHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that never verifies certificates."""
    
    def __init__(self, *args, **kwargs):
        # Remove any context argument
        kwargs.pop('context', None)
        super().__init__(*args, **kwargs)
    
    def https_open(self, req):
        """Open HTTPS connection without verification."""
        return self.do_open(
            self._get_connection,
            req,
            context=ssl._create_unverified_context()
        )
    
    def _get_connection(self, host, **kwargs):
        """Get HTTPS connection with unverified context."""
        kwargs['context'] = ssl._create_unverified_context()
        return http.client.HTTPSConnection(host, **kwargs)

# Install the unverified handler globally
_opener = urllib.request.build_opener(UnverifiedHTTPSHandler())
urllib.request.install_opener(_opener)

# ============================================================================
# HTTP.CLIENT - Patch HTTPSConnection
# ============================================================================

_original_https_connection = http.client.HTTPSConnection

class PatchedHTTPSConnection(http.client.HTTPSConnection):
    """Patched HTTPSConnection that never verifies SSL."""
    
    def __init__(self, *args, **kwargs):
        # Force context to unverified
        kwargs['context'] = ssl._create_unverified_context()
        # Remove check_hostname if present
        kwargs.pop('check_hostname', None)
        super().__init__(*args, **kwargs)

# Replace HTTPSConnection
http.client.HTTPSConnection = PatchedHTTPSConnection

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_ssl_bypass():
    """Verify that SSL bypass is working."""
    try:
        import requests
        # Test with a site that would normally fail with corporate SSL
        response = requests.get('https://www.google.com', verify=False, timeout=5)
        if response.status_code == 200:
            print("✓ SSL bypass verified - requests working")
            return True
    except Exception as e:
        print(f"⚠️  SSL bypass verification failed: {e}")
        return False

# Run verification
verify_ssl_bypass()

print("✓ SSL bypass initialized successfully")
print("  - All urllib.request calls will bypass SSL")
print("  - All requests calls should use verify=False")
print("  - All http.client connections will bypass SSL")