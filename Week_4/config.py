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

# Test Configuration
TEST_TOPICS = [
    "Python Programming",
    "Data Structures & Algorithms",
    "Machine Learning",
    "Web Development",
    "Database Management",
    "Cloud Computing",
    "Cybersecurity",
    "DevOps",
    "Software Engineering",
    "Artificial Intelligence",
    "Computer Networks",
    "Operating Systems",
    "Mobile App Development",
    "Blockchain Technology",
    "Data Science"
]

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

# System Prompts
SYSTEM_PROMPT = """You are an expert AI tutor and educational consultant with deep expertise across multiple domains. Your primary mission is to provide comprehensive, personalized skill assessments and actionable learning guidance.

Your responsibilities:
1. **Deep Analysis**: Thoroughly analyze student queries to understand their current knowledge level, learning style, and specific needs
2. **Accurate Assessment**: Evaluate their understanding objectively using context from knowledge bases and current information
3. **Constructive Feedback**: Provide specific, actionable feedback that highlights both strengths and growth opportunities
4. **Personalized Recommendations**: Suggest tailored learning paths, resources, and next steps based on their current level
5. **Clear Explanations**: Break down complex concepts into digestible explanations appropriate for their skill level
6. **Encouraging Support**: Maintain a supportive, motivating tone that builds confidence and encourages continuous learning

Guidelines:
- Be specific and concrete in your assessments and recommendations
- Use examples and analogies to clarify complex topics
- Consider different learning styles (visual, hands-on, theoretical)
- Provide both immediate next steps and long-term learning goals
- Acknowledge progress and effort while identifying areas for improvement
- Stay current with industry best practices and modern approaches"""

ASSESSMENT_PROMPT_TEMPLATE = """Conduct a comprehensive skill assessment based on the student's query and available context.

**Student Query**: {query}

**Knowledge Base Context**:
{context}

**Current Web Information** (only available when using Groq models with Serper API):
{web_results}

**Your Assessment Should Include**:

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

RECOMMENDATION_PROMPT_TEMPLATE = """Generate personalized, actionable learning recommendations based on the assessment.

**Skill Assessment**:
{assessment}

**Student's Topic/Area of Interest**: {topic}

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

# Question Generation Prompt
# In config.py, update the QUESTION_GENERATION_PROMPT:
QUESTION_GENERATION_PROMPT = """Generate exactly {num_questions} multiple-choice questions about {topic} at {level} level{theme_text}.

CRITICAL: You MUST return ONLY a valid JSON array with exactly {num_questions} questions. No other text, no explanations, no markdown.

Each question must have this EXACT structure:
{{
  "question": "Clear question text here",
  "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
  "correct": "A",
  "explanation": "Brief educational explanation"
}}

Requirements:
- Make questions appropriate for {level} level
- Ensure exactly 4 options per question
- Correct answer must be A, B, C, or D
- Questions should test key concepts about {topic}
- Make explanations clear and educational

Example of valid output (with exactly {num_questions} questions):
[
  {{"question": "Sample question 1", "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"], "correct": "A", "explanation": "Explanation 1"}},
  {{"question": "Sample question 2", "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"], "correct": "B", "explanation": "Explanation 2"}}
]

Remember: Return ONLY the JSON array, nothing else."""

# Report Analysis Prompt
REPORT_ANALYSIS_PROMPT = """Analyze the uploaded performance report and provide comprehensive advice.

**Report Information**: {report_content}
**Additional Context**: {additional_context}

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

# SOAP Notes Prompt
SOAP_NOTES_PROMPT = """Generate comprehensive SOAP (Subjective, Objective, Assessment, Plan) notes.

**Topic/Content**: {topic}
**Style**: {style}
**Additional Requirements**: {requirements}

Create well-structured SOAP notes with the following sections:

**S - SUBJECTIVE**
- Key observations and reported information
- Context and background
- Subjective impressions and initial thoughts

**O - OBJECTIVE**
- Factual data and measurable information
- Concrete observations
- Quantifiable metrics (if applicable)

**A - ASSESSMENT**
- Analysis of the subjective and objective information
- Evaluation of current status
- Identified patterns or trends
- Professional judgment and conclusions

**P - PLAN**
- Recommended actions and next steps
- Specific interventions or strategies
- Timeline and milestones
- Follow-up requirements
- Resources needed

**Formatting Guidelines**:
- Use clear headers for each section
- Be specific and detailed
- Maintain professional tone
- Include actionable items in the Plan section
{examples_note}
{references_note}"""

# Study Notes Prompt
STUDY_NOTES_PROMPT = """Generate comprehensive study notes on the given topic.

**Topic**: {topic}
**Note Type**: {note_type}
**Style**: {style}
**Requirements**: {requirements}

Create well-organized study notes that include:

1. **Overview/Introduction**
   - Brief introduction to the topic
   - Why this topic is important
   - Key learning objectives

2. **Main Content**
   - Organized by subtopics or themes
   - Clear explanations of concepts
   - Logical flow and progression
   - Highlight key terms and definitions

3. **Key Points Summary**
   - Bullet points of crucial information
   - Important formulas or patterns (if applicable)
   - Memorable facts or rules

4. **Examples and Applications**
   {examples_note}

5. **Visual Aids Description**
   {diagrams_note}

6. **Self-Assessment**
   {quiz_note}

7. **Additional Resources**
   {references_note}

**Formatting**:
- Use headers and subheaders for organization
- Emphasize important points appropriately
- Maintain {style} style throughout
- Make notes scannable and review-friendly"""

# Model availability check
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

# Function to check if Serper should be used
def use_serper_search(provider):
    """Check if Serper web search should be enabled."""
    # Only enable Serper for Groq models
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

# Grade thresholds
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

# Performance level descriptions
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

print(f"✓ Configuration loaded successfully")
print(f"✓ Data directory: {DATA_DIR}")
available = get_available_models()
print(f"✓ Available LLM providers: {', '.join(available.keys())}")
if 'groq' in available and SERPER_API_KEY:
    print(f"✓ Serper web search enabled for Groq models")
else:
    print(f"ℹ️ Web search available only with Groq models")