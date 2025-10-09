"""
Download required models manually
"""
import ssl
import certifi
import os

# Fix SSL certificate issue
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

print("📥 Downloading sentence transformer model...")
print("This is a one-time download (~90MB)\n")

try:
    from sentence_transformers import SentenceTransformer
    
    # This will download the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model downloaded successfully!")
    print(f"Model saved to: {model._model_card_vars['model_name']}")
    
    # Test it
    test_embedding = model.encode("test")
    print(f"✅ Model works! Embedding dimension: {len(test_embedding)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative download method...")
    
    # Try with SSL verification disabled (temporary)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model downloaded successfully (with SSL workaround)!")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")