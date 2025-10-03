from typing import List, Dict, Any
from collections import deque
import json
import time
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, max_messages: int = 10):
        self.memories: Dict[str, deque] = {}
        self.max_messages = max_messages
    
    def add_message(self, session_id: str, role: str, content: str, document_context: List[str] = None):
        """Add a message to conversation memory"""
        if session_id not in self.memories:
            self.memories[session_id] = deque(maxlen=self.max_messages)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "document_context": document_context if document_context else []
        }
        
        self.memories[session_id].append(message)
        logger.info(f"Added {role} message to memory for session {session_id}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.memories:
            return []
        return list(self.memories[session_id])
    
    def get_context_string(self, session_id: str, max_messages: int = 5) -> str:
        """Get conversation context as a string for LLM prompting"""
        history = self.get_conversation_history(session_id)
        context_parts = []
        
        # Get last N messages for context
        for msg in history[-max_messages:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
            
            # Include document context if available
            if msg.get("document_context"):
                doc_context = msg["document_context"]
                if doc_context:
                    context_parts.append(f"Document References: {', '.join([doc[:50] + '...' for doc in doc_context])}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self, session_id: str):
        """Clear memory for a specific session"""
        if session_id in self.memories:
            del self.memories[session_id]
            logger.info(f"Cleared memory for session {session_id}")
    
    def get_session_count(self, session_id: str) -> int:
        """Get number of messages in a session"""
        if session_id in self.memories:
            return len(self.memories[session_id])
        return 0
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs"""
        return list(self.memories.keys())