"""
Configuration module for the Skill Assessment AI system.
Loads environment variables and provides configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash", 
    "gemini-1.5-flash"
]

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_MODELS = [
    "gpt-4o-mini"
]

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY = str(CHROMA_DIR)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Safety Configuration
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.7"))
MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.6"))

# Web Search Configuration - Serper (only for Groq models)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "10"))

# Agent Configuration
MAX_ITERATIONS = 5
AGENT_VERBOSE = True

# Test Difficulty Levels
TEST_DIFFICULTY_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced"
]

# Time per question (in seconds) based on difficulty
TIME_PER_QUESTION = {
    "Beginner": 60,      # 1 minute per question
    "Intermediate": 75,  # 1.25 minutes per question
    "Advanced": 90       # 1.5 minutes per question
}

# Note Types Configuration
NOTE_TYPES = [
    "SOAP Notes",
    "Study Notes",
    "Summary Notes",
    "Revision Notes",
    "Lecture Notes",
    "Research Notes"
]

NOTE_STYLES = [
    "Detailed",
    "Concise",
    "Bullet Points",
    "Mind Map Format",
    "Cornell Notes",
    "Outline Format"
]

# ============================================================================
# CONTENT SAFETY GUIDELINES
# ============================================================================

CONTENT_SAFETY_INSTRUCTION = """
CRITICAL SAFETY GUIDELINES - MUST BE FOLLOWED AT ALL TIMES:

You are an educational AI assistant. You MUST refuse to assist with:

**Illegal Activities:**
- Criminal acts (theft, robbery, assault, fraud, hacking)
- Violence or harm to others
- Drug manufacturing or trafficking
- Weapons creation or illegal modifications
- Identity theft or impersonation
- Copyright infringement or piracy

**Unethical Content:**
- Cheating on exams or academic dishonesty
- Plagiarism or content manipulation
- Deception or manipulation tactics
- Harassment or bullying strategies
- Discrimination or hate speech

**Dangerous Information:**
- Instructions for self-harm
- Methods to harm others
- Dangerous experiments without safety context
- Unauthorized access to systems
- Circumventing security measures

**Inappropriate Topics:**
- Adult/explicit content
- Exploitation of any kind
- Misinformation or conspiracy theories
- Scams or fraudulent schemes

If a request falls into any of these categories, you MUST respond with:
"I cannot provide assistance with this topic as it involves illegal, unethical, or harmful content. I'm designed to help with legitimate educational and professional learning. Please ask about a different educational topic."

Only proceed with requests that are:
✓ Educational and constructive
✓ Legal and ethical
✓ Safe and appropriate
✓ Professional and legitimate
"""

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = f"""You are an expert AI tutor and educational consultant with deep expertise across multiple domains. Your primary mission is to provide comprehensive, personalized skill assessments and actionable learning guidance for LEGITIMATE EDUCATIONAL purposes only.

{CONTENT_SAFETY_INSTRUCTION}

Your responsibilities (for appropriate topics only):
1. **Deep Analysis**: Thoroughly analyze student queries to understand their current knowledge level, learning style, and specific needs
2. **Accurate Assessment**: Evaluate their understanding objectively using context from knowledge bases and current information
3. **Constructive Feedback**: Provide specific, actionable feedback that highlights both strengths and growth opportunities
4. **Personalized Recommendations**: Suggest tailored learning paths, resources, and next steps based on their current level
5. **Clear Explanations**: Break down complex concepts into digestible explanations appropriate for their skill level
6. **Encouraging Support**: Maintain a supportive, motivating tone that builds confidence and encourages continuous learning

Guidelines:
- FIRST check if the topic is appropriate and legal
- Be specific and concrete in your assessments and recommendations
- Use examples and analogies to clarify complex topics
- Consider different learning styles (visual, hands-on, theoretical)
- Provide both immediate next steps and long-term learning goals
- Acknowledge progress and effort while identifying areas for improvement
- Stay current with industry best practices and modern approaches
- Focus on legitimate educational and professional development topics"""

ASSESSMENT_PROMPT_TEMPLATE = f"""
{CONTENT_SAFETY_INSTRUCTION}

FIRST: Verify that the following query is appropriate for educational assistance. If it involves illegal, unethical, or harmful content, you MUST refuse and explain why.

**Student Query**: {{query}}

**Knowledge Base Context**:
{{context}}

**Current Web Information**:
{{web_results}}

IF THE QUERY IS APPROPRIATE, provide a comprehensive skill assessment including:

1. **Skill Level Classification**:
   - Beginner: New to the topic, needs foundational concepts
   - Intermediate: Has basic understanding, ready for practical applications
   - Advanced: Strong grasp, ready for optimization and advanced patterns
   - Expert: Deep expertise, exploring cutting-edge developments

2. **Strengths Analysis**:
   - What concepts does the student already understand?
   - What terminology or ideas are they familiar with?
   - What approach or thinking patterns are effective?

3. **Knowledge Gaps**:
   - What fundamental concepts are missing?
   - What common misconceptions might exist?
   - What prerequisites should be addressed?

4. **Specific, Actionable Feedback**:
   - Concrete next steps for improvement
   - Practical exercises or projects to try
   - Resources or topics to explore
   - Common pitfalls to avoid

5. **Learning Path Recommendation**:
   - Immediate focus areas (next 1-2 weeks)
   - Medium-term goals (1-3 months)
   - How this fits into broader skill development

**Tone**: Be encouraging, specific, and constructive. Celebrate progress while clearly identifying growth opportunities. Make your feedback actionable and relevant to real-world applications."""

RECOMMENDATION_PROMPT_TEMPLATE = f"""
{CONTENT_SAFETY_INSTRUCTION}

Generate personalized, actionable learning recommendations based on the assessment.

**Skill Assessment**:
{{assessment}}

**Student's Topic/Area of Interest**: {{topic}}

**Generate 5-7 Specific Recommendations**:

For each recommendation, provide:
1. **Topic/Concept**: Clear, specific area to study
2. **Why It Matters**: Relevance to their goals and current level
3. **How to Learn**: Specific approach (e.g., "Build a project that...", "Study the documentation for...", "Practice by...")
4. **Success Criteria**: How they'll know they've mastered it

**Structure your recommendations as**:
- Start with foundational gaps (if any)
- Progress to immediate skill-building opportunities
- Include practical, hands-on learning activities
- End with stretch goals or advanced topics

**Categories to consider**:
- Core Concepts: Fundamental knowledge they need
- Practical Skills: Hands-on abilities to develop
- Best Practices: Industry standards and patterns
- Tools & Technologies: Relevant tools to learn
- Real-world Applications: Projects or use cases to explore
- Community Resources: Where to find help and stay updated

Make each recommendation specific, achievable, and directly relevant to their learning journey. Avoid generic advice—provide concrete actions they can take today."""

# ============================================================================
# QUESTION GENERATION PROMPT
# ============================================================================

QUESTION_GENERATION_PROMPT = f"""
{CONTENT_SAFETY_INSTRUCTION}

FIRST: Verify that "{{topic}}" is an appropriate educational topic. If it involves illegal, unethical, or harmful content, refuse to generate questions.

IF APPROPRIATE, generate exactly {{num_questions}} multiple-choice questions about {{topic}} at {{level}} level{{theme_text}}.

CRITICAL: You MUST return ONLY a valid JSON array with exactly {{num_questions}} questions. No other text, no explanations, no markdown.

Each question must have this EXACT structure:
{{{{
  "question": "Clear question text here",
  "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
  "correct": "A",
  "explanation": "Brief educational explanation"
}}}}

Requirements:
- Only generate questions for legitimate educational topics
- Make questions appropriate for {{level}} level
- Ensure exactly 4 options per question
- Correct answer must be A, B, C, or D
- Questions should test key concepts about {{topic}}
- Make explanations clear and educational
- Focus on constructive learning, not harmful applications

Example of valid output format:
[
  {{{{"question": "What is Python used for?", "options": ["A) Web development", "B) Data analysis", "C) Automation", "D) All of the above"], "correct": "D", "explanation": "Python is a versatile language used across many domains."}}}},
  {{{{"question": "What is a variable?", "options": ["A) A constant value", "B) A storage location", "C) A function", "D) A loop"], "correct": "B", "explanation": "A variable stores data that can change during program execution."}}}}
]

Remember: Return ONLY the JSON array, nothing else."""

# ============================================================================
# REPORT ANALYSIS PROMPT
# ============================================================================

REPORT_ANALYSIS_PROMPT = f"""
{CONTENT_SAFETY_INSTRUCTION}

Analyze the uploaded performance report and provide comprehensive advice for legitimate educational improvement.

**Report Information**: {{report_content}}
**Additional Context**: {{additional_context}}

Provide a detailed analysis including:

1. **Performance Summary**
   - Overall performance level
   - Key achievements and milestones
   - Areas of excellence

2. **Strengths Identified**
   - Specific skills demonstrated well
   - Consistent strong areas
   - Positive patterns observed

3. **Areas for Improvement**
   - Specific weak points identified
   - Concepts that need reinforcement
   - Skills requiring more practice

4. **Root Cause Analysis**
   - Why certain areas are challenging
   - Possible gaps in foundational knowledge
   - Learning approach considerations

5. **Actionable Improvement Plan**
   - Immediate actions (this week)
   - Short-term goals (1-2 months)
   - Long-term development path (3-6 months)
   - Specific resources and materials
   - Practice exercises and projects

6. **Learning Resources**
   - Recommended courses/tutorials
   - Books and documentation
   - Online platforms and communities
   - Practice platforms

7. **Motivational Feedback**
   - Encouragement and positive reinforcement
   - Realistic goal-setting advice
   - Success strategies

**Tone**: Professional, encouraging, and actionable. Focus on growth mindset and concrete steps."""

# ============================================================================
# SOAP NOTES PROMPT
# ============================================================================

SOAP_NOTES_PROMPT = f"""
{CONTENT_SAFETY_INSTRUCTION}

FIRST: Verify that the topic is appropriate for educational note-taking. Refuse if it involves illegal, unethical, or harmful content.

Generate comprehensive SOAP (Subjective, Objective, Assessment, Plan) notes.

**Topic/Content**: {{topic}}
**Style**: {{style}}

Create well-structured SOAP notes with the following sections:

**S - SUBJECTIVE**
- Key observations and reported information
- Context and background relevant to the topic
- Subjective impressions and initial understanding
- Personal perspectives or experiences related to the content

**O - OBJECTIVE**
- Factual data and verifiable information
- Concrete observations and evidence
- Quantifiable metrics or measurable aspects
- Documented facts without personal interpretation

**A - ASSESSMENT**
- Analysis synthesizing subjective and objective information
- Evaluation of current understanding or status
- Identified patterns, trends, or connections
- Critical thinking and professional judgment
- Gaps in knowledge or areas of uncertainty

**P - PLAN**
- Specific action items and next steps
- Study strategies or learning interventions
- Timeline with realistic milestones
- Follow-up activities or review requirements
- Resources needed for deeper understanding
- Practice exercises or application opportunities

**Formatting Guidelines**:
- Use clear section headers (S, O, A, P)
- Be specific and detailed in each section
- Maintain {{style}} writing style throughout
- Include actionable items in the Plan section
- Keep notes organized and easy to review
- Focus on constructive learning outcomes
{{requirements}}

Remember: These are educational notes for legitimate learning purposes."""

# ============================================================================
# STUDY NOTES PROMPT
# ============================================================================

STUDY_NOTES_PROMPT = f"""
{CONTENT_SAFETY_INSTRUCTION}

FIRST: Verify that "{{topic}}" is an appropriate educational topic. Refuse if it involves illegal, unethical, or harmful content.

Generate comprehensive {{note_type}} on the given topic.

**Topic**: {{topic}}
**Style**: {{style}}

Create well-organized study notes that include:

**1. OVERVIEW**
   - Clear introduction to the topic
   - Why this topic is important for learning
   - Key learning objectives
   - Scope and boundaries of the content

**2. MAIN CONTENT**
   - Organized by logical subtopics or themes
   - Clear explanations of fundamental concepts
   - Progressive difficulty and logical flow
   - Key terms and definitions highlighted
   - Important relationships between concepts
   - Critical thinking points

**3. KEY POINTS SUMMARY**
   - Essential takeaways in bullet format
   - Important formulas, patterns, or principles
   - Memorable facts or rules
   - Common applications
   - Critical distinctions to remember

**4. EXAMPLES & APPLICATIONS**
   - Real-world applications of concepts
   - Practical examples demonstrating key ideas
   - Use cases showing concept in action
   - Problem-solving scenarios
{{requirements}}

**5. SELF-ASSESSMENT**
   - Questions to test understanding
   - Key concepts to review
   - Practice opportunities
   - Areas requiring deeper study

**6. ADDITIONAL RESOURCES**
   - Recommended readings or materials
   - Online resources for further learning
   - Community forums or study groups
   - Practice platforms or tools

**Formatting Requirements**:
- Use clear headers and subheaders
- Maintain {{style}} style consistently
- Emphasize critical points appropriately
- Make notes scannable and review-friendly
- Include visual descriptions where helpful
- Focus on constructive, legitimate learning

Remember: These notes are for educational purposes and must support genuine learning and understanding."""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

def get_available_models():
    """Returns dictionary of available models by provider."""
    available = {}
    
    if GROQ_API_KEY:
        available['groq'] = GROQ_MODELS
    
    if GEMINI_API_KEY:
        available['gemini'] = GEMINI_MODELS
    
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        available['azure_openai'] = AZURE_OPENAI_MODELS
    
    return available

def use_serper_search(provider):
    """Check if Serper web search should be enabled."""
    return provider.lower() == 'groq' and SERPER_API_KEY != ""

# Model information database
MODEL_INFO_DATABASE = {
    'groq': {
        'llama-3.1-8b-instant': {
            'name': 'Llama 3.1 8B Instant',
            'description': 'Meta\'s fast and efficient model optimized for instant responses',
            'parameters': '8 billion',
            'strengths': [
                'Ultra-fast inference',
                'Strong reasoning capabilities',
                'Excellent code generation',
                'Good multilingual support'
            ],
            'best_for': 'Quick assessments, code generation, and general-purpose tasks',
            'speed': '⚡⚡⚡ Ultra Fast',
            'context_window': '128K tokens',
            'provider_info': 'Groq LPU™ Inference Engine',
            'use_cases': [
                'Real-time skill assessments',
                'Quick Q&A sessions',
                'Code review and generation',
                'Interactive tutoring'
            ]
        },
        'gemma2-9b-it': {
            'name': 'Gemma 2 9B IT',
            'description': 'Google\'s instruction-tuned model for complex educational tasks',
            'parameters': '9 billion',
            'strengths': [
                'Excellent instruction following',
                'Detailed analytical responses',
                'Strong educational content generation',
                'Good reasoning for complex topics'
            ],
            'best_for': 'Detailed assessments, comprehensive feedback, and in-depth explanations',
            'speed': '⚡⚡ Very Fast',
            'context_window': '8K tokens',
            'provider_info': 'Groq LPU™ Inference Engine',
            'use_cases': [
                'Comprehensive skill evaluations',
                'Detailed report generation',
                'Complex topic explanations',
                'Educational content creation'
            ]
        }
    },
    'gemini': {
        'gemini-2.0-flash': {
            'name': 'Gemini 2.0 Flash',
            'description': 'Google\'s latest multimodal model with enhanced speed and capabilities',
            'parameters': 'Proprietary',
            'strengths': [
                'Multimodal understanding',
                'Fastest inference in Gemini family',
                'Latest knowledge cutoff',
                'Advanced reasoning'
            ],
            'best_for': 'Comprehensive assessments with cutting-edge AI capabilities',
            'speed': '⚡⚡⚡ Ultra Fast',
            'context_window': '1M tokens',
            'provider_info': 'Google AI',
            'use_cases': [
                'Multi-format content analysis',
                'Large document processing',
                'Advanced reasoning tasks',
                'Latest information integration'
            ]
        },
        'gemini-2.5-flash': {
            'name': 'Gemini 2.5 Flash',
            'description': 'Enhanced Gemini model with superior analytical capabilities',
            'parameters': 'Proprietary',
            'strengths': [
                'Superior reasoning',
                'Complex problem solving',
                'Deep analytical insights',
                'Nuanced understanding'
            ],
            'best_for': 'Advanced assessments requiring deep analysis and insights',
            'speed': '⚡⚡⚡ Ultra Fast',
            'context_window': '1M tokens',
            'provider_info': 'Google AI',
            'use_cases': [
                'Advanced skill assessments',
                'Detailed performance analysis',
                'Complex report generation',
                'Research-level evaluations'
            ]
        },
        'gemini-1.5-flash': {
            'name': 'Gemini 1.5 Flash',
            'description': 'Balanced Gemini model offering reliability and performance',
            'parameters': 'Proprietary',
            'strengths': [
                'Balanced performance',
                'Reliable responses',
                'Good accuracy',
                'Efficient processing'
            ],
            'best_for': 'Standard assessments and general educational tasks',
            'speed': '⚡⚡ Very Fast',
            'context_window': '1M tokens',
            'provider_info': 'Google AI',
            'use_cases': [
                'Standard skill tests',
                'General assessments',
                'Regular evaluations',
                'Educational Q&A'
            ]
        }
    },
    'azure_openai': {
        'gpt-4o-mini': {
            'name': 'GPT-4o Mini',
            'description': 'OpenAI\'s optimized GPT-4 variant for enterprise efficiency',
            'parameters': 'Proprietary (GPT-4 class)',
            'strengths': [
                'High accuracy and reliability',
                'Strong reasoning capabilities',
                'Professional-grade outputs',
                'Well-tested and stable'
            ],
            'best_for': 'Professional assessments and enterprise educational applications',
            'speed': '⚡⚡ Fast',
            'context_window': '128K tokens',
            'provider_info': 'Azure OpenAI Service',
            'use_cases': [
                'Enterprise skill assessments',
                'Professional certifications',
                'Corporate training evaluations',
                'High-stakes testing'
            ]
        }
    }
}

# ============================================================================
# GRADING SYSTEM
# ============================================================================

GRADE_THRESHOLDS = {
    'A+': 95,
    'A': 90,
    'A-': 85,
    'B+': 80,
    'B': 75,
    'B-': 70,
    'C+': 65,
    'C': 60,
    'C-': 55,
    'D': 50,
    'F': 0
}

def get_grade(percentage):
    """Get letter grade based on percentage."""
    for grade, threshold in GRADE_THRESHOLDS.items():
        if percentage >= threshold:
            return grade
    return 'F'

PERFORMANCE_LEVELS = {
    'excellent': {
        'range': (90, 100),
        'description': 'Outstanding performance! You have mastered this topic.',
        'emoji': '🌟'
    },
    'very_good': {
        'range': (80, 89),
        'description': 'Very good performance! You have a strong understanding.',
        'emoji': '🎯'
    },
    'good': {
        'range': (70, 79),
        'description': 'Good performance! You\'re on the right track.',
        'emoji': '👍'
    },
    'satisfactory': {
        'range': (60, 69),
        'description': 'Satisfactory performance. Keep working on improvements.',
        'emoji': '📚'
    },
    'needs_improvement': {
        'range': (0, 59),
        'description': 'Needs improvement. Focus on building strong fundamentals.',
        'emoji': '💪'
    }
}

def get_performance_level(percentage):
    """Get performance level description based on percentage."""
    for level, data in PERFORMANCE_LEVELS.items():
        min_score, max_score = data['range']
        if min_score <= percentage <= max_score:
            return data
    return PERFORMANCE_LEVELS['needs_improvement']

# ============================================================================
# STARTUP CHECKS
# ============================================================================

print(f"✓ Configuration loaded successfully")
print(f"✓ Data directory: {DATA_DIR}")
available = get_available_models()
print(f"✓ Available LLM providers: {', '.join(available.keys())}")
if 'groq' in available and SERPER_API_KEY:
    print(f"✓ Serper web search enabled for Groq models")
else:
    print(f"ℹ️ Web search available only with Groq models")
print(f"✓ Content safety guidelines active")
