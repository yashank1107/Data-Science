"""Quick setup checker"""
import subprocess
import os
import sys

print("🔍 Checking your setup...\n")

# Check Python
print(f"✅ Python: {sys.version.split()[0]}")

# Check if in virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Virtual Environment: Active")
else:
    print("⚠️  Virtual Environment: Not active (run: venv\\Scripts\\activate)")

# Check Ollama
try:
    result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Ollama: Installed")
    else:
        print("❌ Ollama: Not installed")
except FileNotFoundError:
    print("❌ Ollama: Not installed")
    print("   → Download from: https://ollama.ai/download")

# Check .env file
if os.path.exists('.env'):
    print("✅ .env file: Found")
    with open('.env', 'r') as f:
        content = f.read()
        if 'GROQ_API_KEY' in content and 'gsk_' in content:
            print("✅ Groq API Key: Configured")
        else:
            print("⚠️  Groq API Key: Not configured (optional)")
else:
    print("❌ .env file: Not found")

# Check required packages
print("\n📦 Checking packages...")
required = ['streamlit', 'langchain', 'chromadb', 'detoxify']
for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}: Installed")
    except ImportError:
        print(f"❌ {pkg}: Not installed")

print("\n" + "="*50)
print("\n🎯 Recommendation:")
if os.path.exists('.env') and 'gsk_' in open('.env').read():
    print("✅ You can run the app with Groq!")
    print("   Run: streamlit run app.py")
else:
    print("❌ Please either:")
    print("   1. Install Ollama: https://ollama.ai/download")
    print("   2. Or add Groq API key to .env file")