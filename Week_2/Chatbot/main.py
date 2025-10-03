import streamlit as st
import requests
import json
from typing import List, Dict
from datetime import datetime
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - API Keys
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SERPER_API_URL = "https://google.serper.dev/search"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# Story themes and joke categories
STORY_THEMES = [
    "Adventure in Space", "Mystery in a Haunted House", "Romance in Paris",
    "Sci-Fi Dystopia", "Fantasy Kingdom", "Time Travel Paradox",
    "Detective Noir", "Superhero Origin", "Underwater Civilization",
    "Post-Apocalyptic Survival"
]

JOKE_CATEGORIES = [
    "Programming", "Dad Jokes", "Science", "Animals", "Food",
    "Technology", "Wordplay", "One-liners", "Knock-Knock", "Puns"
]

class WebSearch:
    """Web search functionality"""
    
    @staticmethod
    def search(query: str, api_key: str = SERPER_API_KEY) -> str:
        """Search the web and return results"""
        try:
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": 5
            }
            
            response = requests.post(
                SERPER_API_URL,
                headers=headers,
                json=payload,
                timeout=10,
                verify=False
            )
            
            if response.status_code == 200:
                results = response.json()
                search_context = "Current web search results:\n\n"
                
                if "organic" in results:
                    for i, result in enumerate(results["organic"][:5], 1):
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        search_context += f"{i}. {title}\n{snippet}\n\n"
                
                return search_context
            else:
                return None
                
        except Exception as e:
            return None

class GroqClient:
    """Client for Groq API"""
    
    @staticmethod
    def generate(prompt: str, model: str = DEFAULT_GROQ_MODEL, max_tokens: int = 1024, 
                 system_message: str = None) -> str:
        """Generate response from Groq"""
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
            
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

class GeminiClient:
    """Client for Gemini API - Complete package with built-in search"""
    
    @staticmethod
    def generate(prompt: str, max_tokens: int = 1024, system_message: str = None) -> str:
        """Generate response from Gemini"""
        try:
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            
            # Combine system message with prompt if provided
            full_prompt = prompt
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.9
                }
            }
            
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No response generated"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

def needs_web_search(query: str) -> bool:
    """Determine if query needs current information"""
    current_keywords = [
        "today", "now", "current", "latest", "recent", "2024", "2025", 
        "this year", "this month", "news", "update", "happening",
        "stock price", "weather", "score", "election", "president"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in current_keywords)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'groq_model' not in st.session_state:
        st.session_state.groq_model = DEFAULT_GROQ_MODEL
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Gemini Chatbot"

def generate_story(theme: str) -> str:
    """Generate a creative story using Gemini"""
    prompt = f"""Write a captivating short story (300-500 words) based on the theme: "{theme}".
    
Make it engaging with:
- Interesting characters
- A clear beginning, middle, and end
- Vivid descriptions
- An unexpected twist or meaningful conclusion

Be creative and entertaining!"""
    
    return GeminiClient.generate(prompt, max_tokens=1500)

def generate_joke(category: str) -> str:
    """Generate jokes using Gemini"""
    prompt = f"""Generate 3 funny and creative {category} jokes. Make them clever and entertaining!

Format:
1. [First joke]
2. [Second joke]
3. [Third joke]

Make sure they're appropriate and genuinely funny!"""
    
    return GeminiClient.generate(prompt, max_tokens=800)

def gemini_chatbot_interface():
    """Gemini chatbot - Complete package"""
    st.title("🤖 Gemini AI Chatbot")
    st.caption("⚡ Powered by Gemini 2.0 Flash • Complete AI Package • Chat, Stories, Jokes & More!")
    
    # Feature tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📖 Story Generator", "😄 Joke Generator"])
    
    with tab1:
        st.success("🟢 Gemini Ready • Can answer current events!")
        st.divider()
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Ask me anything..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Enhanced system message for Gemini to handle any question
                    system_msg = f"""You are a highly capable AI assistant powered by Gemini 2.0 Flash. 
Today's date is {datetime.now().strftime('%B %d, %Y')}.

You have access to up-to-date information and can answer questions about:
- Current events, news, and happenings
- Recent developments in technology, politics, sports, entertainment
- Latest trends and updates
- Historical facts and general knowledge
- Technical topics and explanations
- Creative tasks like writing, brainstorming, coding
- Any other topic the user asks about

Provide accurate, helpful, and comprehensive answers. If you're unsure about very recent events (last few hours), mention that your information might be slightly delayed. Always be conversational and engaging."""
                    
                    response = GeminiClient.generate(prompt, max_tokens=1024, system_message=system_msg)
                    st.markdown(response)
            
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.subheader("📖 AI Story Generator")
        
        # Input section with better layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            custom_theme = st.text_input("Enter your own theme:", placeholder="e.g., A dragon who loves cooking")
        
        with col2:
            selected_theme = st.selectbox("Or select a quick theme:", [""] + STORY_THEMES)
        
        theme = custom_theme if custom_theme else selected_theme
        
        # Generate button centered
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_clicked = st.button("✨ Generate Story", type="primary", use_container_width=True)
        
        # Display story in full width container
        if generate_clicked:
            if theme:
                with st.spinner("📝 Crafting your story..."):
                    story = generate_story(theme)
                    
                    # Full width display with nice styling
                    st.divider()
                    st.markdown(f"### 📚 {theme}")
                    
                    # Use container for better formatting
                    with st.container():
                        st.markdown(story)
            else:
                st.warning("⚠️ Please enter or select a theme!")
        
        st.divider()
        st.caption("💡 Try: 'A robot learning to love', 'Time traveler's dilemma', 'Last human on Earth'")
    
    with tab3:
        st.subheader("😄 AI Joke Generator")
        
        # Input section with better layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            custom_category = st.text_input("Enter custom category:", placeholder="e.g., Pirates, Coffee, etc.")
        
        with col2:
            selected_category = st.selectbox("Or select a category:", [""] + JOKE_CATEGORIES)
        
        category = custom_category if custom_category else selected_category
        
        # Generate button centered
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_jokes_clicked = st.button("😂 Generate Jokes", type="primary", use_container_width=True)
        
        # Display jokes in full width container
        if generate_jokes_clicked:
            if category:
                with st.spinner("🎭 Cooking up some jokes..."):
                    jokes = generate_joke(category)
                    
                    # Full width display with nice styling
                    st.divider()
                    st.markdown(f"### 😄 {category} Jokes")
                    
                    # Use container for better formatting
                    with st.container():
                        st.markdown(jokes)
            else:
                st.warning("⚠️ Please enter or select a category!")
        
        st.divider()
        st.caption("💡 Try: 'Math', 'Cats', 'Cooking', 'Aliens', 'Social Media'")

def groq_serper_chatbot_interface():
    """Groq + Serper chatbot - Web search enhanced"""
    st.title("🤖 Groq AI Chatbot + Web Search")
    st.caption("⚡ Powered by Groq • 🌐 Real-time web search with Serper")
    
    # Feature tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📖 Story Generator", "😄 Joke Generator"])
    
    with tab1:
        # Show status
        col1, col2 = st.columns(2)
        with col1:
            st.success("🟢 Groq Ready")
        with col2:
            st.success("🌐 Web Search ON")
        
        st.divider()
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("used_search"):
                    st.caption("🔍 Used web search for current information")
        
        # Chat input
        if prompt := st.chat_input("💬 Ask me anything..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                used_search = False
                
                # Check if web search is needed
                if needs_web_search(prompt):
                    with st.spinner("🔍 Searching the web..."):
                        search_results = WebSearch.search(prompt)
                        
                        if search_results:
                            used_search = True
                            enhanced_prompt = f"{search_results}\n\nBased on the above current information, please answer: {prompt}"
                            system_msg = f"You are a helpful AI assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. Use the provided web search results to give accurate, up-to-date information."
                        else:
                            enhanced_prompt = prompt
                            system_msg = None
                else:
                    enhanced_prompt = prompt
                    system_msg = None
                
                with st.spinner("🤔 Thinking..."):
                    response = GroqClient.generate(
                        enhanced_prompt,
                        model=st.session_state.groq_model,
                        max_tokens=1024,
                        system_message=system_msg
                    )
                    
                    st.markdown(response)
                    if used_search:
                        st.caption("🔍 Used web search for current information")
            
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": response,
                "used_search": used_search
            })
    
    with tab2:
        st.subheader("📖 AI Story Generator")
        
        # Input section with better layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            custom_theme = st.text_input("Enter your own theme:", placeholder="e.g., A dragon who loves cooking", key="groq_story_theme")
        
        with col2:
            selected_theme = st.selectbox("Or select a quick theme:", [""] + STORY_THEMES, key="groq_story_select")
        
        theme = custom_theme if custom_theme else selected_theme
        
        # Generate button centered
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_story_clicked = st.button("✨ Generate Story", type="primary", use_container_width=True, key="groq_gen_story")
        
        # Display story in full width container
        if generate_story_clicked:
            if theme:
                with st.spinner("📝 Crafting your story..."):
                    story_prompt = f"""Write a captivating short story (300-500 words) based on the theme: "{theme}".
    
Make it engaging with:
- Interesting characters
- A clear beginning, middle, and end
- Vivid descriptions
- An unexpected twist or meaningful conclusion

Be creative and entertaining!"""
                    story = GroqClient.generate(story_prompt, model=st.session_state.groq_model, max_tokens=1500)
                    
                    # Full width display with nice styling
                    st.divider()
                    st.markdown(f"### 📚 {theme}")
                    
                    # Use container for better formatting
                    with st.container():
                        st.markdown(story)
            else:
                st.warning("⚠️ Please enter or select a theme!")
        
        st.divider()
        st.caption("💡 Try: 'A robot learning to love', 'Time traveler's dilemma', 'Last human on Earth'")
    
    with tab3:
        st.subheader("😄 AI Joke Generator")
        
        # Input section with better layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            custom_category = st.text_input("Enter custom category:", placeholder="e.g., Pirates, Coffee, etc.", key="groq_joke_cat")
        
        with col2:
            selected_category = st.selectbox("Or select a category:", [""] + JOKE_CATEGORIES, key="groq_joke_select")
        
        category = custom_category if custom_category else selected_category
        
        # Generate button centered
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_jokes_clicked = st.button("😂 Generate Jokes", type="primary", use_container_width=True, key="groq_gen_joke")
        
        # Display jokes in full width container
        if generate_jokes_clicked:
            if category:
                with st.spinner("🎭 Cooking up some jokes..."):
                    joke_prompt = f"""Generate 3 funny and creative {category} jokes. Make them clever and entertaining!

Format:
1. [First joke]
2. [Second joke]
3. [Third joke]

Make sure they're appropriate and genuinely funny!"""
                    jokes = GroqClient.generate(joke_prompt, model=st.session_state.groq_model, max_tokens=800)
                    
                    # Full width display with nice styling
                    st.divider()
                    st.markdown(f"### 😄 {category} Jokes")
                    
                    # Use container for better formatting
                    with st.container():
                        st.markdown(jokes)
            else:
                st.warning("⚠️ Please enter or select a category!")
        
        st.divider()
        st.caption("💡 Try: 'Math', 'Cats', 'Cooking', 'Aliens', 'Social Media'")

def main():
    st.set_page_config(
        page_title="AI Assistant Suite",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Mode Selection
        st.subheader("🤖 Choose AI Assistant")
        st.session_state.app_mode = st.radio(
            "Select mode:",
            ["Gemini Chatbot", "Groq + Serper"],
            help="Gemini: Complete package | Groq: Web search specialist"
        )
        
        st.divider()
        
        # Show info based on mode
        if st.session_state.app_mode == "Gemini Chatbot":
            st.success("✅ Gemini 2.0 Flash")
            st.info("""
**Complete AI Package:**
• 💬 Smart Chat
• 📖 Story Generator
• 😄 Joke Generator
• 🌍 Current Events
• 🧠 Creative Writing
            """)
        else:
            st.success("✅ Groq + Serper")
            st.info("""
**Web Search Specialist:**
• ⚡ Lightning Fast
• 🔍 Real-time Web Search
• 📰 Latest News
• 📊 Current Data
• 🌐 Live Information
• 📖 Story Generator
• 😄 Joke Generator
            """)
            
            # Groq Model selection
            available_models = [
                "llama-3.1-8b-instant",
                "gemma2-9b-it"
            ]
            
            model_descriptions = {
                "llama-3.1-8b-instant": "⚡ Fastest",
                "gemma2-9b-it": "🎯 Balanced"
            }
            
            st.divider()
            st.subheader("Model Selection")
            st.session_state.groq_model = st.selectbox(
                "Choose model:",
                available_models,
                format_func=lambda x: f"{x.split('-')[0].title()} - {model_descriptions[x]}"
            )
        
        st.divider()
        
        # Clear chat button
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.divider()
        
        # Comparison table
        st.subheader("📊 Comparison")
        comparison_data = """
| Feature | Gemini | Groq+Serper |
|---------|--------|-------------|
| Chat | ✅ | ✅ |
| Stories | ✅ | ✅ |
| Jokes | ✅ | ✅ |
| Current Info | Built-in | Web Search |
| Speed | Fast | Fastest |
| Specialty | All-rounder | Web Search |
        """
        st.markdown(comparison_data)
        
        st.divider()
        st.caption("💡 **Tip:** Use Gemini for all-in-one, use Groq for speed!")
    
    # Main content based on app mode
    if st.session_state.app_mode == "Gemini Chatbot":
        gemini_chatbot_interface()
    else:
        groq_serper_chatbot_interface()

if __name__ == "__main__":
    main()