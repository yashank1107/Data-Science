"""
Apply network fixes for corporate environments
"""
import ssl
import os

print("Applying network fixes...")

# 1. Disable SSL verification globally (temporary)
ssl._create_default_https_context = ssl._create_unverified_context
print("✅ SSL verification disabled")

# 2. Set environment variables
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
print("✅ Environment variables set")

# 3. Test connections
print("\n📡 Testing connections...\n")

# Test Ollama
# print("1. Testing Ollama...")
# import requests
# try:
#     response = requests.get("http://localhost:11434/api/tags", timeout=5)
#     if response.status_code == 200:
#         print("   ✅ Ollama is responding")
#     else:
#         print(f"   ⚠️ Ollama returned status: {response.status_code}")
# except Exception as e:
#     print(f"   ❌ Ollama error: {e}")

# Test Groq
print("\n2. Testing Groq API...")
try:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"   ✅ Groq is working: {completion.choices[0].message.content}")
    else:
        print("   ⚠️ Groq API key not configured")
except Exception as e:
    print(f"   ❌ Groq error: {e}")

# Test Web Search
print("\n3. Testing Web Search...")
try:
    from duckduckgo_search import DDGS
    ddgs = DDGS()
    results = list(ddgs.text("python", max_results=1))
    if results:
        print(f"   ✅ Web search working: {results[0]['title']}")
    else:
        print("   ⚠️ No search results")
except Exception as e:
    print(f"   ❌ Web search error: {e}")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)