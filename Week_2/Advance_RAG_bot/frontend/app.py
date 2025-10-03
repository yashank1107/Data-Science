import streamlit as st
import requests
import base64
from PIL import Image
import io
import uuid
import os
import time

# Backend API URL
API_BASE = "http://localhost:8000"

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Enhanced Multi-modal RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Add some CSS for better styling
st.markdown("""
<style>
    .stMetric {
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4e8cff;
    }
    .stMetric > div > div > div {
        font-size: 1.2rem; /* Smaller font for metric value */
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = []
    if "last_config_update" not in st.session_state:
        st.session_state.last_config_update = 0
    if "last_document_update" not in st.session_state:
        st.session_state.last_document_update = 0
    if "current_config" not in st.session_state:
        st.session_state.current_config = {}
    if "refresh_trigger" not in st.session_state:
        st.session_state.refresh_trigger = 0

def trigger_refresh():
    """Trigger a refresh of the system status"""
    st.session_state.refresh_trigger += 1

def get_available_llms():
    try:
        response = requests.get(f"{API_BASE}/config/llms")
        if response.status_code == 200:
            return response.json()["llms"]
    except:
        pass
    return ["gemini:gemini-2.0-flash"]

def get_rag_variants():
    try:
        response = requests.get(f"{API_BASE}/config/rag-variants")
        if response.status_code == 200:
            return response.json()["variants"]
    except:
        pass
    return ["basic", "knowledge_graph", "hybrid"]

def get_uploaded_documents():
    try:
        response = requests.get(f"{API_BASE}/documents/list")
        if response.status_code == 200:
            return response.json()["documents"]
    except:
        pass
    return []

def get_current_config():
    """Get current configuration from backend with caching"""
    current_time = time.time()
    
    # Cache for 2 seconds to avoid too many API calls
    if (current_time - st.session_state.get('last_config_update', 0) < 2 and 
        st.session_state.get('current_config')):
        return st.session_state.current_config
    
    try:
        response = requests.get(f"{API_BASE}/config/current")
        if response.status_code == 200:
            config = response.json()
            st.session_state.current_config = config
            st.session_state.last_config_update = current_time
            return config
    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {e}")
    
    return {}

def update_config(selected_llm, selected_rag, selected_docs, enable_search):
    config = {
        "selected_llm": selected_llm,
        "selected_rag_variant": selected_rag,
        "selected_documents": selected_docs,
        "enable_internet_search": enable_search
    }
    try:
        response = requests.post(f"{API_BASE}/config/update", json=config)
        if response.status_code == 200:
            # Clear cache to force refresh
            st.session_state.last_config_update = 0
            trigger_refresh()
            return True
    except Exception as e:
        st.error(f"Configuration update failed: {e}")
    return False

def upload_document(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE}/documents/upload", files=files)
        if response.status_code == 200:
            # Clear cache to force refresh
            st.session_state.last_document_update = 0
            trigger_refresh()
            return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": False, "error": "Upload failed"}

def upload_multiple_documents(files):
    results = []
    for file in files:
        result = upload_document(file)
        results.append(result)
    return results

def send_message(message, images, document_ids):
    data = {
        "message": message,
        "images": images,
        "session_id": st.session_state.session_id,
        "document_ids": document_ids
    }
    try:
        response = requests.post(f"{API_BASE}/chat", json=data)
        if response.status_code == 200:
            trigger_refresh()  # Refresh status after chat
            return response.json()
    except Exception as e:
        return {"response": f"Error: {str(e)}", "sources": []}
    return {"response": "Error: Failed to get response", "sources": []}

def display_system_status():
    """Display system status with auto-refresh"""
    st.subheader("📊 System Status")
    
    # Get current configuration
    config = get_current_config()
    
    if config:
        # Display configuration in a nice format
        selected_llm = config.get('selected_llm', 'Not set')
        is_gemini = selected_llm.startswith("gemini:")
        
        # Determine web search status based on model type
        web_search_status = "✅ Enabled (Auto)" if not is_gemini else ("✅ On" if config.get('enable_internet_search') else "❌ Off")

        st.metric("LLM", selected_llm)
        st.metric("Internet Search", web_search_status)
        
        # Display backend connection status
        st.success("✅ Backend connected")
        
    else:
        st.error("❌ Cannot connect to backend")
        st.info("Make sure the backend server is running.")

def display_uploaded_documents():
    """Display uploaded documents with auto-refresh"""
    st.subheader("📂 Uploaded Documents")
    
    # Use cached documents or fetch new ones
    current_time = time.time()
    if (current_time - st.session_state.get('last_document_update', 0) > 5 or 
        not st.session_state.get('uploaded_documents')):
        st.session_state.uploaded_documents = get_uploaded_documents()
        st.session_state.last_document_update = current_time
    
    uploaded_docs = st.session_state.uploaded_documents
    
    if uploaded_docs:
        for doc in uploaded_docs:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 **{doc['name']}**")
                    st.caption(f"Type: {doc['type']} | Size: {doc.get('size', 0)} bytes")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                        try:
                            response = requests.delete(f"{API_BASE}/documents/{doc['id']}")
                            if response.status_code == 200:
                                st.success(f"Deleted {doc['name']}")
                                st.session_state.last_document_update = 0
                                trigger_refresh()
                                st.rerun()
                        except:
                            st.error("Failed to delete document")
                
                if doc.get('content_summary'):
                    with st.expander("Preview"):
                        st.text(doc['content_summary'][:200] + "..." if len(doc['content_summary']) > 200 else doc['content_summary'])
    else:
        st.info("No documents uploaded yet")
        
    # Refresh button for documents
    if st.button("🔄 Refresh Documents", key="refresh_docs"):
        st.session_state.last_document_update = 0
        trigger_refresh()
        st.rerun()

def display_conversation_memory():
    """Display conversation memory"""
    st.subheader("🧠 Memory Usage")
    
    # Current memory stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages in Memory", len(st.session_state.conversation))
    with col2:
        st.metric("Session ID", st.session_state.session_id[:8] + "...")
    
    # Clear memory button
    if st.button("🗑️ Clear Memory", key="clear_memory_main"):
        st.session_state.conversation = []
        trigger_refresh()
        st.success("Memory cleared!")
    
    # Conversation history
    if st.session_state.conversation:
        with st.expander("View Conversation History", expanded=False):
            for i, msg in enumerate(st.session_state.conversation[-10:]):
                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                with st.chat_message(msg["role"]):
                    st.write(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")
                    if "images" in msg and msg["images"]:
                        st.write("📷 *Image attached*")

def main():
    init_session()
    
    st.title("🚀 Enhanced Multi-modal Advanced RAG Chatbot")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Get current config to set defaults
        current_config_data = get_current_config()
        
        def config_change_callback():
            """Callback to update config when a sidebar widget changes."""
            selected_llm = st.session_state.llm_selector
            is_gemini = selected_llm.startswith("gemini:")
            enable_search = st.session_state.search_toggle if is_gemini else True

            update_config(
                selected_llm,
                st.session_state.rag_selector,
                st.session_state.selected_documents,
                enable_search
            )
            st.success("Configuration updated!")

        # LLM Selection
        available_llms = get_available_llms()
        llm_index = available_llms.index(current_config_data.get('selected_llm', available_llms[0])) if current_config_data.get('selected_llm') in available_llms else 0
        st.selectbox("Select LLM", available_llms, index=llm_index, key="llm_selector", on_change=config_change_callback)
        
        # RAG Variant Selection
        rag_variants = get_rag_variants()
        rag_index = rag_variants.index(current_config_data.get('selected_rag_variant', rag_variants[0])) if current_config_data.get('selected_rag_variant') in rag_variants else 0
        st.selectbox("RAG Variant", rag_variants, index=rag_index, key="rag_selector", on_change=config_change_callback)
        
        # Document Management
        st.subheader("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Documents (Multiple files supported)",
            type=['pdf', 'txt', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'pptx', 'xlsx'],
            accept_multiple_files=True,
            key="multi_file_uploader"
        )
        
        if uploaded_files and st.button("Upload Files", key="upload_files_btn"):
            with st.spinner("Uploading files..."):
                results = upload_multiple_documents(uploaded_files)
                
                success_count = 0
                for i, result in enumerate(results):
                    if result.get("success"):
                        success_count += 1
                        st.success(f"✅ {uploaded_files[i].name} uploaded successfully!")
                    else:
                        st.error(f"❌ Failed to upload {uploaded_files[i].name}: {result.get('error', 'Unknown error')}")
                
                if success_count > 0:
                    # Force refresh of documents list
                    st.session_state.last_document_update = 0
                    trigger_refresh()
        
        # Document selection
        uploaded_docs = get_uploaded_documents()
        if uploaded_docs:
            st.subheader("📄 Select Documents for Chat")
            doc_options = {doc['name']: doc['id'] for doc in uploaded_docs}
            default_selected = [name for name, doc_id in doc_options.items() if doc_id in current_config_data.get('selected_documents', [])]
            st.multiselect(
                "Choose documents to chat with:",
                options=list(doc_options.keys()),
                default=default_selected,
                key="doc_selector",
                on_change=lambda: setattr(st.session_state, 'selected_documents', [doc_options[name] for name in st.session_state.doc_selector]) and config_change_callback()
            )
        
        # Internet Search Toggle
        is_gemini_selected = st.session_state.get('llm_selector', '').startswith("gemini:")
        search_value = current_config_data.get('enable_internet_search', False) if is_gemini_selected else True
        st.checkbox(
            "🔍 Enable Internet Search", 
            value=search_value, 
            key="search_toggle", 
            on_change=config_change_callback,
            disabled=not is_gemini_selected)
        
        st.markdown("---")
        st.subheader("💬 Conversation Memory")
        if st.button("🗑️ Clear Memory", key="clear_memory_sidebar"):
            st.session_state.conversation = []
            trigger_refresh()
            st.success("Memory cleared!")
        
        st.markdown("---")
        st.subheader("🛡️ Features")
        st.markdown("""
        - **Multi-modal** (Text + Images)
        - **Advanced RAG** variants
        - **Internet search** integration
        - **Enhanced guardrails** with document relevance
        - **Multi-file upload** support
        - **Conversation memory** (10 messages)
        - **Observability** with Opik
        """)
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display conversation
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "images" in msg and msg["images"]:
                    for img_data in msg["images"]:
                        try:
                            img_bytes = base64.b64decode(img_data)
                            image = Image.open(io.BytesIO(img_bytes))
                            st.image(image, caption="Uploaded Image", width=200)
                        except:
                            st.write("📷 *Image attachment*")
        
        # Image upload
        uploaded_images = st.file_uploader(
            "📷 Upload Images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="image_uploader"
        )
        
        # Convert images to base64
        image_data_list = []
        if uploaded_images:
            for uploaded_image in uploaded_images:
                image_bytes = uploaded_image.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                image_data_list.append(image_b64)
                # Display uploaded image
                st.image(image_bytes, caption=uploaded_image.name, width=150)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to conversation
            st.session_state.conversation.append({
                "role": "user",
                "content": user_input,
                "images": image_data_list
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
                for img_data in image_data_list:
                    try:
                        img_bytes = base64.b64decode(img_data)
                        image = Image.open(io.BytesIO(img_bytes))
                        st.image(image, caption="Uploaded Image", width=200)
                    except:
                        st.write("📷 *Image attachment*")
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = send_message(user_input, image_data_list, st.session_state.selected_documents)
                    
                    if response.get("rejection_reason"):
                        st.warning(f"🚫 {response['response']}")
                    else:
                        st.write(response["response"])
                    
                    # Display sources if available
                    if response.get("sources"):
                        with st.expander("📚 Sources"):
                            for source in response["sources"]:
                                st.write(f"**Title:** {source.get('title', 'Unknown')}")
                                st.write(f"**Snippet:** {source.get('snippet', '')}")
                                if source.get('link'):
                                    st.write(f"**Link:** {source.get('link')}")
                                st.markdown("---")
            
            # Add assistant response to conversation
            if not response.get("rejection_reason"):
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response["response"]
                })
            
            # Force a small refresh to update the status panel
            trigger_refresh()
    
    with col2:
        # System Status Section with auto-refresh
        display_system_status()
        
        # Uploaded Documents Section
        display_uploaded_documents()
        
        # Conversation Memory Section
        display_conversation_memory()
        
        # Manual refresh button for the entire status panel
        if st.button("🔄 Refresh All Status", key="refresh_all"):
            st.session_state.last_config_update = 0
            st.session_state.last_document_update = 0
            trigger_refresh()
            st.rerun()

if __name__ == "__main__":
    main()