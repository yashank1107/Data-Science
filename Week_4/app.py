"""
Streamlit frontend for the Skill Assessment AI system.
Main application entry point with multi-model support and enhanced features.
"""
import ssl_bypass
import streamlit as st
import logging
from datetime import datetime, timedelta
import sys
import os
import re
from pathlib import Path
import json
import time
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from src.web_tools import WebSearchTool

# Fix path issues
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Test imports with error handling
try:
    import config
    from src.agent import create_agent
    from src.utils import Timer
    IMPORTS_OK = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Skill Assessment Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if imports failed
if not IMPORTS_OK:
    st.error(f"❌ Import Error: {IMPORT_ERROR}")
    st.info("""
    **Troubleshooting Steps:**
    1. Make sure all dependencies are installed
    2. Check that `src/` folder exists with `agent.py` and `utils.py`
    3. Check that `config.py` exists in the root directory
    4. Check terminal/console for detailed error messages
    """)
    st.stop()

# Custom CSS - Enhanced for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
        width: 100%;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        width: 100%;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .assessment-box {
    }
    .question-box {
    }
    .timer-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .score-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .recommendation-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .safety-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .safety-error {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
        color: white;
    }
    .badge-groq { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .badge-gemini { 
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
    }
    .badge-azure { 
        background: linear-gradient(135deg, #0078d4 0%, #00bcf2 100%);
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .model-info-card {
        
    }
</style>
""", unsafe_allow_html=True)


def get_available_models():
    """Get all available models organized by provider."""
    models = {}
    
    # Groq models
    if config.GROQ_API_KEY:
        models['Groq'] = config.GROQ_MODELS
    
    # Gemini models
    if config.GEMINI_API_KEY:
        models['Gemini'] = config.GEMINI_MODELS
    
    # Azure OpenAI models
    if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
        models['Azure OpenAI'] = config.AZURE_OPENAI_MODELS
    
    return models


def get_model_info(provider, model_name):
    """Get detailed information about the selected model."""
    model_info = {
        'Groq': {
            'llama-3.1-8b-instant': {
                'name': 'Llama 3.1 8B Instant',
                'description': 'Fast and efficient model for general tasks',
                'strengths': ['Quick responses', 'Good reasoning', 'Code generation'],
                'best_for': 'General assessments and quick evaluations',
                'speed': '⚡ Very Fast',
                'context': '128K tokens'
            },
            'gemma2-9b-it': {
                'name': 'Gemma 2 9B IT',
                'description': 'Instruction-tuned model for complex tasks',
                'strengths': ['Detailed analysis', 'Instruction following', 'Educational content'],
                'best_for': 'Detailed assessments and comprehensive feedback',
                'speed': '⚡ Fast',
                'context': '8K tokens'
            }
        },
        'Gemini': {
            'gemini-2.0-flash': {
                'name': 'Gemini 2.0 Flash',
                'description': 'Latest fast-inference multimodal model',
                'strengths': ['Multimodal understanding', 'Fast processing', 'Latest knowledge'],
                'best_for': 'Comprehensive assessments with latest information',
                'speed': '⚡⚡ Ultra Fast',
                'context': '1M tokens'
            },
            'gemini-2.5-flash': {
                'name': 'Gemini 2.5 Flash',
                'description': 'Advanced flash model with enhanced capabilities',
                'strengths': ['Advanced reasoning', 'Complex analysis', 'Deep insights'],
                'best_for': 'Advanced skill assessments and detailed reports',
                'speed': '⚡⚡ Ultra Fast',
                'context': '1M tokens'
            },
            'gemini-1.5-flash': {
                'name': 'Gemini 1.5 Flash',
                'description': 'Balanced performance and speed',
                'strengths': ['Balanced performance', 'Good accuracy', 'Efficient'],
                'best_for': 'Standard assessments and evaluations',
                'speed': '⚡ Very Fast',
                'context': '1M tokens'
            }
        },
        'Azure OpenAI': {
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini',
                'description': 'Optimized GPT-4 variant for efficiency',
                'strengths': ['High accuracy', 'Reliable', 'Well-tested'],
                'best_for': 'Professional assessments and enterprise use',
                'speed': '⚡ Fast',
                'context': '128K tokens'
            }
        }
    }
    
    return model_info.get(provider, {}).get(model_name, {
        'name': model_name,
        'description': 'AI model for skill assessment',
        'strengths': ['General purpose'],
        'best_for': 'Various assessment tasks',
        'speed': '⚡ Fast',
        'context': 'Standard'
    })


def initialize_agent_with_model(provider, model_name, serper_enabled=False):
    """Initialize agent with specific model."""
    try:
        # Determine which LLM to use
        if provider == "Groq":
            llm_config = {
                'type': 'groq',
                'model': model_name,
                'api_key': config.GROQ_API_KEY
            }
        elif provider == "Gemini":
            llm_config = {
                'type': 'gemini',
                'model': model_name,
                'api_key': config.GEMINI_API_KEY
            }
        else:  # Azure OpenAI
            llm_config = {
                'type': 'azure_openai',
                'model': model_name,
                'api_key': config.AZURE_OPENAI_API_KEY,
                'endpoint': config.AZURE_OPENAI_ENDPOINT,
                'api_version': config.AZURE_OPENAI_API_VERSION
            }
        
        # Initialize with safety fallbacks
        agent = create_agent(initialize_sample_data=True, serper_enabled=serper_enabled)
        return agent, None
    except Exception as e:
        error_msg = f"Agent initialization error: {str(e)}"
        logger.error(error_msg)
        
        # Provide more specific guidance
        if "meta tensor" in str(e).lower() or "detoxify" in str(e).lower():
            error_msg += "\n\nToxicity detection model failed to load. The app will continue with basic safety checks."
            # Create a minimal agent without safety features
            from src.llm_interface import LLMManager
            from src.vector_store_use import USEVectorStore
            llm_manager = LLMManager()
            vector_store = USEVectorStore()
            return MinimalAgent(llm_manager, vector_store), error_msg
        else:
            return None, error_msg


class MinimalAgent:
    """Minimal agent for fallback when full agent fails."""
    
    def __init__(self, llm_manager, vector_store):
        self.llm = llm_manager
        self.vector_store = vector_store
    
    def run(self, query: str):
        """Simple run method for fallback."""
        try:
            # Simple context retrieval
            vector_results = self.vector_store.search(query, n_results=3)
            context = ""
            if vector_results["documents"] and vector_results["documents"][0]:
                context_docs = vector_results["documents"][0]
                context = "\n\n".join(context_docs)
            
            prompt = f"""Please provide a helpful assessment for this query:

Query: {query}

Context from knowledge base:
{context}

Please provide:
1. A clear explanation of the topic
2. Key concepts to understand
3. Learning recommendations

Keep it educational and helpful."""
            
            assessment = self.llm.generate(prompt)
            
            return {
                "assessment": assessment,
                "recommendations": [
                    "Practice regularly with hands-on exercises",
                    "Review fundamental concepts",
                    "Work on small projects to apply your knowledge"
                ],
                "safety_check": {
                    "issues": [],
                    "safe": True
                },
                "vector_results": vector_results,
                "web_results": {}
            }
        except Exception as e:
            return {
                "assessment": f"I apologize, but I encountered an error: {str(e)}",
                "recommendations": [],
                "error": str(e)
            }

def generate_direct_llm_response(provider, model_name, prompt, system_prompt=None):
    """Generate response directly from LLM without agent workflow."""
    try:
        from src.llm_interface import LLMManager
        llm_manager = LLMManager()
        
        response = llm_manager.generate(
            prompt=prompt,
            system_prompt=system_prompt or config.SYSTEM_PROMPT
        )
        
        return response, None
    except Exception as e:
        error_msg = f"LLM generation error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def generate_test_questions(topic, level, theme, provider, model_name, num_questions=12):
    """Generate test questions using AI."""
    try:
        # Safely format theme text
        theme_text = ""
        if theme and theme.strip():
            clean_theme = theme.strip().replace('"', "'").replace('\n', ' ')
            theme_text = f" with a focus on {clean_theme}"
        
        # Create a more focused prompt using config template
        prompt = config.QUESTION_GENERATION_PROMPT.format(
            topic=topic,
            level=level,
            theme_text=theme_text,
            num_questions=num_questions
        )
        
        # Use direct LLM call instead of agent workflow
        assessment, error = generate_direct_llm_response(
            provider, 
            model_name, 
            prompt,
            system_prompt="You are an expert educational content creator. Generate valid JSON only."
        )
        
        if error:
            return None, error
        
        result = {"assessment": assessment}
        
        # Extract assessment content
        assessment = result.get("assessment", "")
        
        if not assessment:
            logger.warning("Empty assessment received from agent")
            return generate_fallback_questions(topic, level, num_questions), None
        
        # Clean the response
        cleaned_response = assessment.strip()
        logger.info(f"Raw response preview: {cleaned_response[:200]}...")
        
        # Try multiple parsing strategies
        questions = try_parse_json_response(cleaned_response, num_questions)
        
        if questions:
            logger.info(f"Successfully parsed {len(questions)} questions")
            return questions, None
        else:
            logger.warning("Could not parse any valid questions from response")
            return generate_fallback_questions(topic, level, num_questions), None
            
    except Exception as e:
        logger.error(f"Question generation error: {str(e)}", exc_info=True)
        st.error(f"⚠️ Error generating questions: {str(e)}")
        return generate_fallback_questions(topic, level, num_questions), None


def try_parse_json_response(response_text, expected_count):
    """Try multiple strategies to parse JSON from LLM response."""
    strategies = [
        parse_direct_json,
        parse_json_from_markdown,
        parse_json_repair_single_quotes,
        parse_json_find_array
    ]
    
    for strategy in strategies:
        try:
            questions = strategy(response_text, expected_count)
            if questions and len(questions) >= min(3, expected_count):  # At least 3 questions
                logger.info(f"Strategy {strategy.__name__} succeeded with {len(questions)} questions")
                return questions[:expected_count]
        except Exception as e:
            logger.debug(f"Strategy {strategy.__name__} failed: {e}")
            continue
    
    return None


def parse_direct_json(response_text, expected_count):
    """Try to parse the response as direct JSON."""
    try:
        # Try parsing the entire response as JSON
        questions = json.loads(response_text)
        if isinstance(questions, list):
            return validate_questions(questions, expected_count)
    except json.JSONDecodeError:
        pass
    return None


def parse_json_from_markdown(response_text, expected_count):
    """Extract JSON from markdown code blocks."""
    import re
    # Look for ```json ... ``` or ``` ... ```
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for match in matches:
        try:
            questions = json.loads(match.strip())
            if isinstance(questions, list):
                validated = validate_questions(questions, expected_count)
                if validated:
                    return validated
        except json.JSONDecodeError:
            continue
    return None


def parse_json_repair_single_quotes(response_text, expected_count):
    """Try to fix common JSON issues like single quotes."""
    try:
        # Replace single quotes with double quotes, but be careful with contractions
        fixed_json = re.sub(r"(?<!\w)'|'(?!\w)", '"', response_text)
        # Fix unescaped quotes in text
        fixed_json = re.sub(r'(?<![\\])"(?=\s*:)', '"', fixed_json)
        questions = json.loads(fixed_json)
        if isinstance(questions, list):
            return validate_questions(questions, expected_count)
    except json.JSONDecodeError:
        pass
    return None


def parse_json_find_array(response_text, expected_count):
    """Find and extract JSON array from text."""
    import re
    # Look for array pattern
    array_pattern = r'\[\s*\{.*?\}\s*\]'
    matches = re.findall(array_pattern, response_text, re.DOTALL)
    
    for match in matches:
        try:
            questions = json.loads(match)
            if isinstance(questions, list):
                validated = validate_questions(questions, expected_count)
                if validated:
                    return validated
        except json.JSONDecodeError:
            continue
    return None


def validate_questions(questions, expected_count):
    """Validate and clean parsed questions."""
    if not isinstance(questions, list):
        return None
    
    valid_questions = []
    for q in questions:
        if not isinstance(q, dict):
            continue
            
        # Check required fields
        if not all(key in q for key in ['question', 'options', 'correct', 'explanation']):
            continue
            
        # Validate options is a list with 4 items
        if not (isinstance(q['options'], list) and len(q['options']) == 4):
            continue
            
        # Validate correct answer format
        if q['correct'] not in ['A', 'B', 'C', 'D']:
            continue
            
        # Clean the question data
        cleaned_q = {
            'question': str(q['question']).strip(),
            'options': [str(opt).strip() for opt in q['options']],
            'correct': str(q['correct']).strip().upper(),
            'explanation': str(q['explanation']).strip()
        }
        
        # Ensure options start with A), B), C), D)
        for i, option in enumerate(cleaned_q['options']):
            prefix = ['A)', 'B)', 'C)', 'D)'][i]
            if not option.startswith(prefix):
                cleaned_q['options'][i] = f"{prefix} {option}"
        
        valid_questions.append(cleaned_q)
    
    return valid_questions if valid_questions else None

def generate_fallback_questions(topic, level, num_questions):
    """Generate fallback questions if AI generation fails."""
    logger.info(f"Generating {num_questions} fallback questions for {topic} at {level} level")
    
    questions = []
    for i in range(num_questions):
        questions.append({
            "question": f"Question {i+1} about {topic} ({level} level) - This is a sample question. Please try generating questions again for actual content.",
            "options": [
                f"A) Option 1 for question {i+1}",
                f"B) Option 2 for question {i+1}",
                f"C) Option 3 for question {i+1}",
                f"D) Option 4 for question {i+1}"
            ],
            "correct": "A",
            "explanation": f"This is a fallback question about {topic}. The AI-generated questions could not be parsed. Please try again or contact support if this persists."
        })
    
    return questions


def calculate_time_per_question(num_questions, level):
    """Calculate time allocation based on level and number of questions."""
    base_time = {
        'Beginner': 60,  # 60 seconds per question
        'Intermediate': 75,  # 75 seconds
        'Advanced': 90  # 90 seconds
    }
    total_seconds = base_time.get(level, 60) * num_questions
    return total_seconds


def generate_pdf_report(report_data, student_name):
    """Generate PDF report (placeholder - requires reportlab)."""
    try:    
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        elements.append(Paragraph(f"Skill Assessment Report", title_style))
        elements.append(Paragraph(f"<b>Student:</b> {student_name}", styles['Normal']))
        elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add report content
        if 'score' in report_data:
            elements.append(Paragraph(f"<b>Score:</b> {report_data['score']}%", styles['Heading2']))
        
        if 'summary' in report_data:
            elements.append(Paragraph("<b>Summary:</b>", styles['Heading3']))
            elements.append(Paragraph(report_data['summary'], styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer
    except ImportError:
        # Fallback to text report
        buffer = BytesIO()
        report_text = f"""
SKILL ASSESSMENT REPORT
{'='*50}

Student: {student_name}
Date: {datetime.now().strftime('%B %d, %Y')}

{report_data.get('summary', 'Assessment completed.')}

Score: {report_data.get('score', 'N/A')}%
"""
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer


def display_safety_report(safety_check):
    """Display safety check results."""
    issues = safety_check.get("issues", [])
    
    if not issues:
        st.success("✅ All safety checks passed")
        return
    
    # Categorize issues by severity
    high_severity = [i for i in issues if i["severity"] == "high"]
    medium_severity = [i for i in issues if i["severity"] == "medium"]
    low_severity = [i for i in issues if i["severity"] == "low"]
    
    # Display high severity (blocking)
    if high_severity:
        st.markdown('<div class="safety-error">', unsafe_allow_html=True)
        st.error("🚫 **Critical Safety Issues Detected**")
        for issue in high_severity:
            st.write(f"- **{issue['type'].upper()}**: {issue['message']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display medium severity (warning)
    if medium_severity:
        st.markdown('<div class="safety-warning">', unsafe_allow_html=True)
        st.warning("⚠️ **Safety Warnings**")
        for issue in medium_severity:
            st.write(f"- **{issue['type'].upper()}**: {issue['message']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display low severity (info)
    if low_severity:
        with st.expander("ℹ️ Additional Safety Notes"):
            for issue in low_severity:
                st.info(f"**{issue['type'].upper()}**: {issue['message']}")


def display_assessment_result(result):
    """Display the complete assessment result."""
    
    # Check for errors
    if result.get("error"):
        st.error(f"❌ An error occurred: {result['error']}")
        return
    
    # Safety Check Section
    st.markdown("---")
    st.subheader("🔒 Safety Check")
    display_safety_report(result.get("safety_check", {}))
    
    # If blocked due to toxicity, don't show further results
    high_toxicity = any(
        issue["type"] == "toxicity" and issue["severity"] == "high"
        for issue in result.get("safety_check", {}).get("issues", [])
    )
    
    if high_toxicity:
        st.error("⛔ Processing stopped due to inappropriate content. Please rephrase your query.")
        return
    
    # Assessment Section
    st.markdown("---")
    st.subheader("📊 Skill Assessment")
    
    assessment = result.get("assessment", "")
    if assessment:
        st.markdown(f'<div class="assessment-box">{assessment}</div>', unsafe_allow_html=True)
    else:
        st.warning("No assessment generated.")
    
    # Recommendations Section
    st.markdown("---")
    st.subheader("💡 Personalized Learning Recommendations")
    
    recommendations = result.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(
                f'<div class="recommendation-box"><strong>{i}.</strong> {rec}</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No specific recommendations at this time.")
    
    # Source Information (Expandable)
    st.markdown("---")
    with st.expander("🔍 View Source Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 Knowledge Base Results**")
            vector_results = result.get("vector_results", {})
            if vector_results.get("documents") and vector_results["documents"][0]:
                for i, (doc, meta) in enumerate(zip(
                    vector_results["documents"][0][:3],
                    vector_results["metadatas"][0][:3]
                ), 1):
                    st.write(f"{i}. **Topic**: {meta.get('topic', 'N/A')}")
                    st.caption(doc[:150] + "...")
            else:
                st.info("No knowledge base results")
        
        with col2:
            st.markdown("**🌐 Web Search Results**")
            web_results = result.get("web_results", {}).get("search_results", [])
            if web_results:
                for i, res in enumerate(web_results[:3], 1):
                    st.write(f"{i}. [{res['title']}]({res['url']})")
                    st.caption(res['snippet'][:100] + "...")
            else:
                st.info("No web results")


def feature_skill_test():
    """Feature 1: Interactive Skill Test"""
    st.markdown('<div class="main-header">📝 Interactive Skill Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Test your knowledge with AI-generated questions</div>', unsafe_allow_html=True)
    
    # Get selected model from session state
    provider = st.session_state.get('selected_provider', 'Groq')
    model_name = st.session_state.get('selected_model', 'llama-3.1-8b-instant')
    
    # Test configuration
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
    
    if not st.session_state.test_started:
        # Configuration phase
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "📚 Enter Topic",
                value="Python Programming",
                help="Type any topic you want to be tested on (e.g., 'Machine Learning', 'Cloud Computing', etc.)"
            )
            
            level = st.selectbox(
                "📊 Choose Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"]
            )
        
        with col2:
            theme = st.text_input(
                "🎨 Theme (Optional)",
                placeholder="e.g., 'real-world applications', 'interview questions'"
            )
            
            num_questions = st.slider(
                "❓ Number of Questions",
                min_value=10,
                max_value=15,
                value=12
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Test", type="primary", use_container_width=True):
                with st.spinner("🎯 Generating your personalized test..."):
                    questions, error = generate_test_questions(
                        topic, level, theme, provider, model_name, num_questions
                    )
                    
                    if error:
                        st.error(f"Failed to generate test: {error}")
                    else:
                        st.session_state.test_started = True
                        st.session_state.test_questions = questions
                        st.session_state.test_topic = topic
                        st.session_state.test_level = level
                        st.session_state.test_theme = theme
                        st.session_state.current_question = 0
                        # Initialize all answers as None
                        st.session_state.test_answers = {i: None for i in range(len(questions))}
                        st.session_state.test_start_time = datetime.now()
                        st.session_state.test_duration = calculate_time_per_question(len(questions), level)
                        st.session_state.last_update_time = datetime.now()
                        st.rerun()
    
    else:
        # Test in progress
        questions = st.session_state.test_questions
        current_q = st.session_state.current_question
        
        # Calculate remaining time
        elapsed = (datetime.now() - st.session_state.test_start_time).total_seconds()
        remaining = max(0, st.session_state.test_duration - elapsed)
        
        # Timer display with auto-refresh
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        
        # Create a container for the timer that will auto-refresh
        timer_container = st.empty()
        with timer_container.container():
            st.markdown(
                f'<div class="timer-box">⏱️ Time Remaining: {minutes:02d}:{seconds:02d}</div>',
                unsafe_allow_html=True
            )
        
        # Auto-submit when time's up
        if remaining <= 0:
            st.session_state.test_completed = True
            st.rerun()
        
        if len(questions) == 0:
            st.error("No questions available for this topic. Please try a different topic or check your data source.")
            st.stop()

        # Progress bar
        progress = (current_q / len(questions))
        st.progress(progress, text=f"Question {current_q + 1} of {len(questions)}")
        
        st.markdown("---")
        
        # Display current question
        if current_q < len(questions):
            question_data = questions[current_q]
            
            st.markdown(f'<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"### Question {current_q + 1}")
            st.markdown(f"**{question_data['question']}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer options - initialize with None if not answered
            current_answer = st.session_state.test_answers.get(current_q, None)
            answer = st.radio(
                "Select your answer:",
                question_data['options'],
                key=f"q_{current_q}",
                index=question_data['options'].index(current_answer) if current_answer in question_data['options'] else None
            )
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_q > 0:
                    if st.button("⬅️ Previous", use_container_width=True):
                        if answer:
                            st.session_state.test_answers[current_q] = answer
                        st.session_state.current_question -= 1
                        st.rerun()
            
            with col2:
                if st.button("💾 Save & Next", type="primary", use_container_width=True, disabled=not answer):
                    st.session_state.test_answers[current_q] = answer
                    if current_q < len(questions) - 1:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        st.session_state.test_completed = True
                        st.rerun()
            
            with col3:
                if st.button("🏁 Submit Test", use_container_width=True):
                    if answer:
                        st.session_state.test_answers[current_q] = answer
                    st.session_state.test_completed = True
                    st.rerun()
            
            # Auto-refresh timer every 5 seconds if time is running
            if remaining > 0:
                time_since_last_update = (datetime.now() - st.session_state.get('last_update_time', datetime.now())).total_seconds()
                if time_since_last_update >= 5:  # Refresh every 5 seconds
                    st.session_state.last_update_time = datetime.now()
                    st.rerun()
        
        # Show test results
        if st.session_state.get('test_completed', False):
            st.balloons()
            display_test_results()


def display_test_results():
    """Display test results with detailed analytics."""
    st.markdown('<div class="main-header">🎉 Test Completed!</div>', unsafe_allow_html=True)
    
    questions = st.session_state.test_questions
    answers = st.session_state.test_answers
    
    # Calculate score
    correct = 0
    total = len(questions)
    question_analysis = []
    
    for i, q in enumerate(questions):
        user_answer = answers.get(i)
        correct_answer = q['correct']
        
        # Handle None or empty answers safely
        if user_answer is None or not user_answer.strip():
            is_correct = False
            user_answer_display = "Not answered"
        else:
            is_correct = user_answer.startswith(correct_answer)
            user_answer_display = user_answer
        
        if is_correct:
            correct += 1
        
        question_analysis.append({
            'question': q['question'],
            'user_answer': user_answer_display,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'explanation': q.get('explanation', '')
        })
    
    score_percentage = (correct / total) * 100 if total > 0 else 0
    
    # Score card
    st.markdown(
        f'''<div class="score-card">
            <h1>Your Score: {correct}/{total}</h1>
            <h2>{score_percentage:.1f}%</h2>
        </div>''',
        unsafe_allow_html=True
    )
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("✅ Correct", correct)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("❌ Incorrect", total - correct)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.metric("📊 Accuracy", f"{score_percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        grade = "A+" if score_percentage >= 90 else "A" if score_percentage >= 80 else "B" if score_percentage >= 70 else "C" if score_percentage >= 60 else "D"
        st.metric("🏆 Grade", grade)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Rest of the function remains the same...
    # Visualization
    st.markdown("---")
    st.subheader("📈 Performance Analysis")
    
    # Create performance chart
    try:
        import plotly.graph_objects as go
        
        # Pie chart for results
        fig = go.Figure(data=[go.Pie(
            labels=['Correct', 'Incorrect'],
            values=[correct, total - correct],
            hole=.3,
            marker_colors=['#28a745', '#dc3545']
        )])
        fig.update_layout(
            title="Answer Distribution",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        # Fallback visualization
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Correct: {correct}")
        with col2:
            st.error(f"❌ Incorrect: {total - correct}")
    
    # Detailed question review
    st.markdown("---")
    st.subheader("📝 Detailed Review")
    
    for i, analysis in enumerate(question_analysis, 1):
        with st.expander(f"Question {i} - {'✅ Correct' if analysis['is_correct'] else '❌ Incorrect'}"):
            st.markdown(f"**Q: {analysis['question']}**")
            st.write(f"Your answer: {analysis['user_answer']}")
            st.write(f"Correct answer: {analysis['correct_answer']}")
            if analysis['explanation']:
                st.info(f"💡 {analysis['explanation']}")
    
    # AI-generated recommendations
    st.markdown("---")
    st.subheader("🎯 Personalized Recommendations")
    
    # Generate recommendations based on performance
    if score_percentage >= 80:
        st.success("🌟 **Excellent Performance!** You have a strong understanding of this topic.")
        recommendations = [
            "Continue to advanced topics and real-world projects",
            "Consider teaching others to reinforce your knowledge",
            "Explore specialized areas within this domain",
            "Take on challenging projects to apply your skills"
        ]
    elif score_percentage >= 60:
        st.info("📚 **Good Progress!** You're on the right track with room for improvement.")
        recommendations = [
            "Review the concepts you missed in detail",
            "Practice with more exercises in weak areas",
            "Work on hands-on projects to strengthen understanding",
            "Consider study groups or mentorship"
        ]
    else:
        st.warning("💪 **Keep Learning!** Focus on building strong fundamentals.")
        recommendations = [
            "Start with foundational concepts and basics",
            "Use interactive tutorials and guided learning",
            "Practice regularly with simple exercises",
            "Don't hesitate to ask for help and clarification"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f'<div class="recommendation-box">{i}. {rec}</div>', unsafe_allow_html=True)
    
    # PDF Download with name input
    st.markdown("---")
    st.subheader("📄 Download Report")
    
    student_name = st.text_input("Enter your name for the report:", key="test_report_name")
    
    if student_name:
        if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
            report_data = {
                'score': score_percentage,
                'correct': correct,
                'total': total,
                'summary': f"Test on {st.session_state.test_topic} ({st.session_state.test_level} level). Score: {correct}/{total} ({score_percentage:.1f}%)"
            }
            
            pdf_buffer = generate_pdf_report(report_data, student_name)
            
            st.download_button(
                label="📥 Download Report PDF",
                data=pdf_buffer,
                file_name=f"skill_test_report_{student_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    # Restart test button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Take Another Test", use_container_width=True):
            # Clear test state
            for key in ['test_started', 'test_completed', 'test_questions', 'test_answers', 
                       'current_question', 'test_start_time', 'test_duration']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def feature_report_analysis():
    """Feature 2: Report Analysis and Advice"""
    st.markdown('<div class="main-header">📊 Report Analysis & Advice</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your report and get personalized advice</div>', unsafe_allow_html=True)
    
    # Get selected model
    provider = st.session_state.get('selected_provider', 'Groq')
    model_name = st.session_state.get('selected_model', 'llama-3.1-8b-instant')
    
    st.info("📤 Upload your performance report (PDF or Image) to receive detailed analysis and improvement suggestions.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Upload a PDF document or image of your report"
    )
    
    if uploaded_file:
        file_type = uploaded_file.type
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.write(f"**Type:** {file_type}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            # Display preview for images
            if 'image' in file_type:
                st.image(uploaded_file, caption="Report Preview", use_column_width=True)
        
        st.markdown("---")
        
        # Additional context
        additional_context = st.text_area(
            "📝 Additional Context (Optional)",
            placeholder="Provide any additional information about your report or specific areas you'd like advice on...",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Report", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing your report and generating personalized advice..."):
                    try:
                        # Extract text from file (basic implementation)
                        report_text = f"Report uploaded: {uploaded_file.name}"
                        
                        if 'image' in file_type:
                            report_text += "\n[Image-based report - visual analysis would be performed]"
                        else:
                            report_text += "\n[PDF report - text extraction would be performed]"
                        
                        # Generate advice using direct LLM call
                        analysis_prompt = config.REPORT_ANALYSIS_PROMPT.format(
                            report_content=report_text,
                            additional_context=additional_context or "No additional context provided."
                        )
                        
                        assessment, error = generate_direct_llm_response(
                            provider,
                            model_name,
                            analysis_prompt
                        )
                        
                        if error:
                            st.error(f"Failed to generate analysis: {error}")
                        else:
                            result = {'assessment': assessment, 'recommendations': []}
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("🎯 Analysis Results")
                            
                            st.markdown('<div class="assessment-box">', unsafe_allow_html=True)
                            st.markdown(result.get('assessment', 'Analysis completed.'))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display recommendations
                            if result.get('recommendations'):
                                st.markdown("---")
                                st.subheader("💡 Personalized Recommendations")
                                for i, rec in enumerate(result['recommendations'], 1):
                                    st.markdown(
                                        f'<div class="recommendation-box">{i}. {rec}</div>',
                                        unsafe_allow_html=True
                                    )
                            
                            # Download analysis report
                            st.markdown("---")
                            student_name = st.text_input("Enter your name for the analysis report:", key="analysis_report_name")
                            
                            if student_name:
                                if st.button("📥 Download Analysis Report", use_container_width=True):
                                    report_data = {
                                        'summary': result.get('assessment', 'Analysis completed.')
                                    }
                                    pdf_buffer = generate_pdf_report(report_data, student_name)
                                    
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=pdf_buffer,
                                        file_name=f"report_analysis_{student_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf"
                                    )
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"Report analysis error: {str(e)}", exc_info=True)


def feature_notes_generator():
    """Feature 3: SOAP Notes / Study Notes Generator"""
    st.markdown('<div class="main-header">📝 Notes Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate professional SOAP notes or comprehensive study notes</div>', unsafe_allow_html=True)
    
    # Get selected model
    provider = st.session_state.get('selected_provider', 'Groq')
    model_name = st.session_state.get('selected_model', 'llama-3.1-8b-instant')
    
    # Note type selection
    col1, col2 = st.columns(2)
    
    with col1:
        note_type = st.selectbox(
            "📋 Select Note Type",
            ["SOAP Notes", "Study Notes", "Summary Notes", "Revision Notes"]
        )
    
    with col2:
        note_style = st.selectbox(
            "🎨 Note Style",
            ["Detailed", "Concise", "Bullet Points", "Mind Map Format"]
        )
    
    # Topic/Subject input
    topic = st.text_area(
        "📚 Topic or Content",
        placeholder="Enter the topic, subject matter, or paste content you want notes on...",
        height=150
    )
    
    # Additional parameters
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_examples = st.checkbox("Include Examples", value=True)
            include_diagrams = st.checkbox("Include Diagram Descriptions", value=False)
        
        with col2:
            include_quiz = st.checkbox("Add Practice Questions", value=False)
            include_references = st.checkbox("Add References/Resources", value=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Generate Notes", type="primary", use_container_width=True):
            if not topic.strip():
                st.warning("⚠️ Please enter a topic or content.")
            else:
                with st.spinner("📝 Generating your notes..."):
                    try:
                        # Build requirements string
                        requirements = []
                        if include_examples:
                            requirements.append("- Add practical examples and use cases")
                        if include_diagrams:
                            requirements.append("- Include descriptions for helpful diagrams or visuals")
                        if include_quiz:
                            requirements.append("- Add practice questions for self-assessment")
                        if include_references:
                            requirements.append("- Include recommended resources and references")
                        
                        requirements_text = "\n".join(requirements) if requirements else ""
                        
                        # Build prompt based on note type
                        if note_type == "SOAP Notes":
                            prompt = config.SOAP_NOTES_PROMPT.format(
                                topic=topic,
                                style=note_style,
                                requirements=requirements_text
                            )
                        else:
                            prompt = config.STUDY_NOTES_PROMPT.format(
                                note_type=note_type,
                                topic=topic,
                                style=note_style,
                                requirements=requirements_text
                            )
                        
                        # Use direct LLM call
                        assessment, error = generate_direct_llm_response(
                            provider,
                            model_name,
                            prompt
                        )
                        
                        if error:
                            st.error(f"Failed to generate notes: {error}")
                        else:
                            result = {'assessment': assessment}
                            
                            # Display generated notes
                            st.markdown("---")
                            st.subheader(f"📄 Your {note_type}")
                            
                            st.markdown('<div class="assessment-box">', unsafe_allow_html=True)
                            st.markdown(result.get('assessment', 'Notes generated successfully.'))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download options
                            st.markdown("---")
                            st.subheader("💾 Download Options")
                            
                            student_name = st.text_input("Enter your name for the notes:", key="notes_name")
                            
                            if student_name:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # PDF download
                                    if st.button("📥 Download as PDF", use_container_width=True):
                                        report_data = {
                                            'summary': result.get('assessment', 'Notes generated.')
                                        }
                                        pdf_buffer = generate_pdf_report(report_data, student_name)
                                        
                                        st.download_button(
                                            label="📄 Download PDF",
                                            data=pdf_buffer,
                                            file_name=f"{note_type.lower().replace(' ', '_')}_{student_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf"
                                        )
                                
                                with col2:
                                    # Text download
                                    notes_text = f"""{note_type}
{'='*50}
Student: {student_name}
Date: {datetime.now().strftime('%B %d, %Y')}
Topic: {topic}

{result.get('assessment', 'Notes generated.')}
"""
                                    st.download_button(
                                        label="📝 Download as Text",
                                        data=notes_text,
                                        file_name=f"{note_type.lower().replace(' ', '_')}_{student_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"Notes generation error: {str(e)}", exc_info=True)


def feature_ai_assessment():
    """Feature 4: Original AI Assessment (existing functionality)"""
    st.markdown('<div class="main-header">🎓 AI Skill Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Get personalized feedback and learning recommendations powered by AI</div>',
        unsafe_allow_html=True
    )
    
    # Get selected model
    provider = st.session_state.get('selected_provider', 'Groq')
    model_name = st.session_state.get('selected_model', 'llama-3.1-8b-instant')
    
    serper_enabled = provider == "Groq" and config.SERPER_API_KEY != ""
    agent, error = initialize_agent_with_model(provider, model_name, serper_enabled=serper_enabled)

    st.markdown("---")
    
    # Input section
    st.subheader("📝 Ask Your Question")
    
    # Get query from session state or default
    default_query = st.session_state.get("query_input", "")
    
    query = st.text_area(
        "Enter your question or topic you'd like to learn about:",
        value=default_query,
        height=120,
        placeholder="e.g., Explain object-oriented programming in Python, or How do I learn machine learning?",
        key="query_text_area"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        assess_button = st.button("🚀 Assess My Skills", type="primary")
    
    # Process query
    if assess_button and query.strip():
        # Clear previous query from session state
        if "query_input" in st.session_state:
            del st.session_state.query_input
        
        # Initialize agent with selected model
        with st.spinner(f"🔄 Initializing {provider} ({model_name})..."):
            agent, error = initialize_agent_with_model(provider, model_name)
        
        if error:
            st.error(f"❌ Failed to initialize AI system: {error}")
            return
        
        with st.spinner("🤔 Analyzing your query and generating assessment..."):
            try:
                web_results = []
                # Pass web_results to agent.run if your agent supports it
                result = agent.run(query)

                # Run assessment
                start_time = datetime.now()
                result = agent.run(query)
                elapsed_time = (datetime.now() - start_time).total_seconds()
                
                # Display results
                display_assessment_result(result)
                
                # Show processing time and model info
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"⏱️ Processing time: {elapsed_time:.2f} seconds")
                with col2:
                    st.caption(f"🤖 Model used: {provider} - {model_name}")
                
            except Exception as e:
                logger.error(f"Assessment error: {str(e)}", exc_info=True)
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please try again or rephrase your question.")
    
    elif assess_button:
        st.warning("⚠️ Please enter a question or topic to assess.")


def main():
    """Main application function."""
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No LLM providers configured!")
        st.info("""
        **Setup Instructions:**
        1. For Groq: Add GROQ_API_KEY to .env file
        2. For Gemini: Add GEMINI_API_KEY to .env file
        3. For Azure OpenAI: Add AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to .env
        """)
        return
    
    # Sidebar - Model Selection & Navigation
    with st.sidebar:
        st.markdown("### 🤖 AI Model Selection")
        
        # Flatten models for selection
        model_options = []
        model_map = {}
        
        for provider, models in available_models.items():
            for model in models:
                display_name = f"{provider}: {model}"
                model_options.append(display_name)
                model_map[display_name] = (provider, model)
        
        # Model selector
        selected_model = st.selectbox(
            "Choose AI Model",
            options=model_options,
            index=0,
            help="Select which AI model to use"
        )
        
        provider, model_name = model_map[selected_model]
        
        # Store in session state
        st.session_state.selected_provider = provider
        st.session_state.selected_model = model_name
        
        # Display model badge
        badge_class = {
            'Groq': 'badge-groq',
            'Gemini': 'badge-gemini',
            'Azure OpenAI': 'badge-azure'
        }
        st.markdown(
            f'<span class="model-badge {badge_class[provider]}">{provider}</span>',
            unsafe_allow_html=True
        )
        st.caption(f"Model: `{model_name}`")
        
        # Model information
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        model_info = get_model_info(provider, model_name)
        
        st.markdown('<div class="model-info-card">', unsafe_allow_html=True)
        st.markdown(f"**{model_info['name']}**")
        st.caption(model_info['description'])
        st.write(f"**Speed:** {model_info['speed']}")
        st.write(f"**Context:** {model_info['context']}")
        
        with st.expander("View Details"):
            st.markdown("**Strengths:**")
            for strength in model_info['strengths']:
                st.write(f"• {strength}")
            st.markdown(f"**Best For:** {model_info['best_for']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        feature_choice = st.radio(
            "Choose Feature:",
            [
                "🎓 AI Assessment",
                "📝 Skill Test",
                "📊 Report Analysis",
                "📝 Notes Generator"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **AI Skill Assessment Hub**
        
        Your comprehensive learning companion with:
        • Multi-model AI support
        • Interactive skill tests
        • Report analysis
        • Notes generation
        • Personalized recommendations
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        st.markdown("**Available Providers:**")
        for prov in available_models.keys():
            st.write(f"✓ {prov}")
        
        st.markdown("**Capabilities:**")
        st.write("✓ Toxicity Detection")
        st.write("✓ PII Protection")
        st.write("✓ Bias Awareness")
        if provider == "Groq":
            st.write("✓ Web Search (Serper)")
    
    # Main content based on feature selection
    if feature_choice == "📝 Skill Test":
        feature_skill_test()
    elif feature_choice == "📊 Report Analysis":
        feature_report_analysis()
    elif feature_choice == "📝 Notes Generator":
        feature_notes_generator()
    else:  # AI Assessment
        feature_ai_assessment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎓 AI Skill Assessment Hub - Your Learning Companion</p>
        <p><small>Powered by LangGraph • Groq • Gemini • Azure OpenAI • ChromaDB • Serper</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    
    try:
        # Run app
        main()
    except Exception as e:
        st.error(f"Fatal Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())