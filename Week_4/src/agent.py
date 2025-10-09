"""
Main Agent module using LangGraph for orchestration.
Implements the skill assessment workflow with all tools integrated.

CRITICAL: SSL certificate bypass MUST be first, before any other imports!
"""

# ============================================================================
# CRITICAL: SSL BYPASS MUST BE FIRST - DO NOT MOVE THIS SECTION
# ============================================================================
import ssl
import os

# Completely disable SSL verification globally
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['SSL_CERT_DIR'] = ''

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Patch urllib to use unverified context
import urllib.request
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey patch urllib.request.urlopen to disable SSL verification
_original_urlopen = urllib.request.urlopen

def _patched_urlopen(url, data=None, timeout=None, *args, **kwargs):
    """Patched urlopen that disables SSL verification."""
    if isinstance(url, str):
        # Create request with unverified context
        import ssl
        context = ssl._create_unverified_context()
        return _original_urlopen(url, data=data, timeout=timeout, context=context, *args, **kwargs)
    else:
        # If url is already a Request object
        import ssl
        context = ssl._create_unverified_context()
        kwargs['context'] = context
        return _original_urlopen(url, data=data, timeout=timeout, *args, **kwargs)

urllib.request.urlopen = _patched_urlopen

# Also patch HTTPSHandler
class NoVerifyHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = ssl._create_unverified_context()
    
    def https_open(self, req):
        return self.do_open(self._get_connection, req, context=self._context)
    
    def _get_connection(self, host, **kwargs):
        import http.client
        return http.client.HTTPSConnection(host, context=self._context, **kwargs)

# Install the no-verify handler
urllib.request.install_opener(
    urllib.request.build_opener(NoVerifyHTTPSHandler())
)

# ============================================================================
# END SSL BYPASS SECTION - Now safe to import other modules
# ============================================================================

import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

import config
from src.llm_interface import LLMManager
from src.vector_store_use import USEVectorStore as VectorStore
from src.web_tools import WebSearchTool
from src.responsible_ai import ResponsibleAIMonitor
from src.utils import format_search_results, extract_topics, log_interaction, Timer

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State object passed between agent nodes."""
    query: str
    safety_check: Dict[str, Any]
    vector_results: Dict[str, Any]
    web_results: Dict[str, Any]
    context: str
    assessment: str
    recommendations: List[str]
    metadata: Dict[str, Any]
    error: Optional[str]


class SkillAssessmentAgent:
    """
    Main agent for skill assessment workflow.
    Uses LangGraph for state management and orchestration.
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        vector_store: VectorStore,
        web_tools: WebSearchTool,
        safety_monitor: ResponsibleAIMonitor
    ):
        self.llm = llm_manager
        self.vector_store = vector_store
        self.web_tools = web_tools
        self.safety_monitor = safety_monitor
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("✓ SkillAssessmentAgent initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow using LangGraph."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes with unique names (different from state keys)
        workflow.add_node("check_safety", self.safety_check_node)
        workflow.add_node("get_context", self.retrieve_context_node)
        workflow.add_node("search_web", self.web_search_node)
        workflow.add_node("assess", self.assess_skill_node)
        workflow.add_node("recommend", self.generate_recommendations_node)
        
        # Define edges (flow between nodes)
        workflow.set_entry_point("check_safety")
        
        # Conditional edge after safety check
        workflow.add_conditional_edges(
            "check_safety",
            self._route_after_safety_check,
            {
                "continue": "get_context",
                "end": END
            }
        )
        
        workflow.add_edge("get_context", "search_web")
        workflow.add_edge("search_web", "assess")
        workflow.add_edge("assess", "recommend")
        workflow.add_edge("recommend", END)
        
        logger.info("✓ Workflow graph constructed")
        return workflow.compile()
    
    def _route_after_safety_check(self, state: AgentState) -> str:
        """Route after safety check based on results."""
        if state.get("error"):
            return "end"
        
        safety_result = state.get("safety_check", {})
        
        # If toxicity detected, stop processing
        toxicity_issues = [
            issue for issue in safety_result.get("issues", [])
            if issue["type"] == "toxicity" and issue["severity"] == "high"
        ]
        
        if toxicity_issues:
            logger.warning("Blocking request due to high toxicity")
            state["error"] = "Request blocked due to inappropriate content"
            return "end"
        
        return "continue"
    
    def safety_check_node(self, state: AgentState) -> AgentState:
        """Node 1: Check query for safety issues."""
        logger.info("Node 1: Safety Check")
        
        try:
            with Timer("Safety check"):
                safety_result = self.safety_monitor.check_safety(
                    state["query"],
                    check_toxicity=True,
                    check_pii=True,
                    check_bias=True,
                    anonymize_pii=True
                )
            
            state["safety_check"] = safety_result
            
            # Use anonymized text if PII was detected
            if safety_result["processed_text"] != safety_result["original_text"]:
                logger.info("Using anonymized query for processing")
                state["query"] = safety_result["processed_text"]
            
            # Log safety issues
            if safety_result["issues"]:
                for issue in safety_result["issues"]:
                    logger.warning(f"Safety issue: {issue['type']} - {issue['message']}")
        
        except Exception as e:
            logger.error(f"Safety check error: {str(e)}")
            state["error"] = f"Safety check failed: {str(e)}"
        
        return state
    
    def retrieve_context_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant context from vector database."""
        logger.info("Node 2: Retrieve Context")
        
        try:
            with Timer("Vector search"):
                # Search vector database for relevant content
                vector_results = self.vector_store.search(
                    query=state["query"],
                    n_results=5
                )
            
            state["vector_results"] = vector_results
            
            # Format context for LLM
            if vector_results["documents"] and vector_results["documents"][0]:
                context_docs = vector_results["documents"][0]
                metadatas = vector_results["metadatas"][0]
                
                context_parts = []
                for doc, meta in zip(context_docs, metadatas):
                    topic = meta.get("topic", "general")
                    difficulty = meta.get("difficulty", "unknown")
                    context_parts.append(f"[{topic} - {difficulty}] {doc}")
                
                state["context"] = "\n\n".join(context_parts)
                logger.info(f"Retrieved {len(context_docs)} relevant documents")
            else:
                state["context"] = "No relevant context found in knowledge base."
                logger.warning("No vector results found")
        
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            state["context"] = "Error retrieving context from knowledge base."
        
        return state
    
    def web_search_node(self, state: AgentState) -> AgentState:
        """Node 3: Search web for current information."""
        logger.info("Node 3: Web Search")
        
        try:
            if not self.web_tools.enabled:
                logger.info("Web search disabled for this run.")
                state["web_results"] = {
                    "search_results": [],
                    "formatted": "Web search not enabled for this model."
                }
                return state
            # Extract topic from query for better search
            topics = extract_topics(state["query"])
            search_query = state["query"]
            
            if topics:
                # Enhance query with educational context
                search_query = f"{state['query']} tutorial guide"
            
            with Timer("Web search"):
                # Perform web search
                search_results = self.web_tools.search_tool.search(
                    query=search_query,
                    max_results=config.MAX_SEARCH_RESULTS
                )
            
            state["web_results"] = {
                "search_results": search_results,
                "formatted": format_search_results(search_results)
            }
            
            logger.info(f"Found {len(search_results)} web results")
        
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            state["web_results"] = {
                "search_results": [],
                "formatted": "Error performing web search."
            }
        
        return state
    
    def assess_skill_node(self, state: AgentState) -> AgentState:
        """Node 4: Assess student's skill level."""
        logger.info("Node 4: Assess Skill")
        
        try:
            # Prepare prompt for assessment
            assessment_prompt = config.ASSESSMENT_PROMPT_TEMPLATE.format(
                query=state["query"],
                context=state.get("context", "No context available"),
                web_results=state["web_results"]["formatted"]
            )
            
            with Timer("LLM assessment generation"):
                # Generate assessment
                assessment = self.llm.generate(
                    prompt=assessment_prompt,
                    system_prompt=config.SYSTEM_PROMPT
                )
            
            state["assessment"] = assessment
            logger.info("Assessment generated successfully")
        
        except Exception as e:
            logger.error(f"Assessment generation error: {str(e)}")
            state["assessment"] = "Error generating skill assessment."
            state["error"] = str(e)
        
        return state
    
    def generate_recommendations_node(self, state: AgentState) -> AgentState:
        """Node 5: Generate learning recommendations."""
        logger.info("Node 5: Generate Recommendations")
        
        try:
            # Extract topic from query
            topics = extract_topics(state["query"])
            main_topic = topics[0] if topics else "the subject"
            
            # Prepare prompt for recommendations
            recommendation_prompt = config.RECOMMENDATION_PROMPT_TEMPLATE.format(
                assessment=state["assessment"],
                topic=main_topic
            )
            
            with Timer("LLM recommendation generation"):
                # Generate recommendations
                recommendations_text = self.llm.generate(
                    prompt=recommendation_prompt,
                    system_prompt=config.SYSTEM_PROMPT
                )
            
            # Parse recommendations into list
            recommendations = [
                line.strip().lstrip('•-*').strip()
                for line in recommendations_text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            
            # Filter empty and very short recommendations
            recommendations = [r for r in recommendations if len(r) > 10]
            
            state["recommendations"] = recommendations[:5]  # Max 5 recommendations
            logger.info(f"Generated {len(state['recommendations'])} recommendations")
        
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            state["recommendations"] = ["Error generating recommendations. Please try again."]
        
        return state
    
    def run(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete skill assessment workflow.
        
        Args:
            query: Student's query or question
            metadata: Optional metadata about the interaction
        
        Returns:
            Complete assessment results
        """
        logger.info(f"Starting skill assessment for query: '{query[:50]}...'")
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "safety_check": {},
            "vector_results": {},
            "web_results": {},
            "context": "",
            "assessment": "",
            "recommendations": [],
            "metadata": metadata or {},
            "error": None
        }
        
        try:
            # Execute workflow
            with Timer("Complete workflow"):
                final_state = self.workflow.invoke(initial_state)
            
            # Log interaction
            log_interaction(
                query=query,
                response=final_state.get("assessment", ""),
                metadata={
                    "success": final_state.get("error") is None,
                    "safety_issues": len(final_state.get("safety_check", {}).get("issues", [])),
                    "recommendations_count": len(final_state.get("recommendations", []))
                }
            )
            
            logger.info("✓ Skill assessment completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                **initial_state,
                "error": f"Workflow failed: {str(e)}",
                "assessment": "An error occurred during assessment. Please try again.",
                "recommendations": []
            }


# Factory function to create agent with all dependencies
def create_agent(
    initialize_sample_data: bool = True, serper_enabled: bool = False
) -> SkillAssessmentAgent:
    """
    Factory function to create a fully initialized agent.
    
    Args:
        initialize_sample_data: Whether to add sample data to vector store
    
    Returns:
        Initialized SkillAssessmentAgent
    """
    logger.info("Creating Skill Assessment Agent...")
    
    try:
        # Initialize LLM
        llm_manager = LLMManager()
        logger.info(f"LLM: {llm_manager.get_active_model()}")
        
        # Initialize Vector Store with USE
        from src.vector_store_use import initialize_sample_data as init_sample_data
        vector_store = VectorStore()
        
        # Add sample data if needed and vector store is empty
        if initialize_sample_data and vector_store.collection.count() == 0:
            logger.info("Adding sample educational content...")
            init_sample_data(vector_store)
        
        # Initialize Web Tools
        web_tools = WebSearchTool(enabled=serper_enabled)
        
        # Initialize Safety Monitor
        safety_monitor = ResponsibleAIMonitor()
        
        # Create agent
        agent = SkillAssessmentAgent(
            llm_manager=llm_manager,
            vector_store=vector_store,
            web_tools=web_tools,
            safety_monitor=safety_monitor
        )
        
        logger.info("✓ Agent created successfully")
        return agent
    
    except Exception as e:
        logger.error(f"Agent initialization error: {str(e)}")
        raise Exception(f"Agent initialization error: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Skill Assessment Agent...\n")
    
    try:
        # Create agent
        agent = create_agent(initialize_sample_data=True)
        
        # Test queries
        test_queries = [
            "What is object-oriented programming in Python?",
            "Explain machine learning to me",
            "How do sorting algorithms work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"Test {i}: {query}")
            print(f"{'='*70}\n")
            
            result = agent.run(query)
            
            # Display results
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Assessment completed\n")
                
                # Safety check summary
                safety_issues = result["safety_check"].get("issues", [])
                if safety_issues:
                    print(f"⚠️  Safety Issues: {len(safety_issues)}")
                    for issue in safety_issues:
                        print(f"   - {issue['type']}: {issue['message']}")
                    print()
                
                # Assessment
                print("📊 ASSESSMENT:")
                print(result["assessment"])
                print()
                
                # Recommendations
                print("💡 RECOMMENDATIONS:")
                for j, rec in enumerate(result["recommendations"], 1):
                    print(f"{j}. {rec}")
            
            print(f"\n{'='*70}\n")
        
        print("✓ Agent module ready!")
    
    except Exception as e:
        print(f"❌ Failed to initialize or run agent: {str(e)}")
        import traceback
        traceback.print_exc()