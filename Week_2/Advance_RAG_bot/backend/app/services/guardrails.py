from transformers import pipeline
from typing import Dict, Any, List
import re

class EnhancedGuardrailsService:
    def __init__(self):
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            tokenizer="unitary/toxic-bert"
        )
        
        # Document type patterns for relevance checking
        self.document_categories = {
            "medical": [
                'medical', 'health', 'patient', 'diagnosis', 'treatment', 
                'prescription', 'symptoms', 'doctor', 'hospital', 'clinical',
                'medicine', 'healthcare', 'report', 'test results'
            ],
            "legal": [
                'legal', 'law', 'contract', 'agreement', 'lawsuit', 'court',
                'attorney', 'legal document', 'clause', 'jurisdiction'
            ],
            "financial": [
                'financial', 'bank', 'loan', 'investment', 'tax', 'revenue',
                'profit', 'loss', 'balance sheet', 'income statement'
            ],
            "technical": [
                'technical', 'code', 'programming', 'software', 'hardware',
                'system', 'network', 'database', 'algorithm'
            ]
        }
    
    def check_toxicity(self, text: str) -> Dict[str, Any]:
        """Check if text contains toxic content"""
        try:
            results = self.toxicity_classifier(text[:512])
            toxic_score = max([result['score'] for result in results if result['label'] in ['toxic', 'obscene', 'insult', 'threat']])
            
            return {
                "is_toxic": toxic_score > 0.7,
                "score": toxic_score,
                "safe": toxic_score <= 0.7
            }
        except Exception:
            return {"is_toxic": False, "score": 0.0, "safe": True}
    
    def check_nsfw_request(self, text: str, images: List[str] = None) -> bool:
        """Check for NSFW content requests"""
        nsfw_keywords = [
            'porn', 'nude', 'sexual', 'explicit', 'adult content',
            'nsfw', 'not safe for work', 'erotic', 'xxx', 'pornography'
        ]
        
        text_lower = text.lower()
        for keyword in nsfw_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def check_document_relevance(self, user_question: str, document_context: List[str]) -> Dict[str, Any]:
        """Check if user question is relevant to uploaded documents"""
        if not document_context:
            return {"is_relevant": True, "reason": "No documents uploaded"}
        
        # Extract document topics
        document_topics = self._extract_document_topics(document_context)
        question_topics = self._extract_question_topics(user_question)
        
        # Check relevance
        is_relevant = self._check_topic_relevance(question_topics, document_topics)
        
        return {
            "is_relevant": is_relevant,
            "document_topics": document_topics,
            "question_topics": question_topics,
            "reason": "Question is outside document scope" if not is_relevant else "Question is relevant to documents"
        }
    
    def _extract_document_topics(self, document_context: List[str]) -> List[str]:
        """Extract main topics from document context"""
        topics = []
        full_text = " ".join(document_context).lower()
        
        for category, keywords in self.document_categories.items():
            for keyword in keywords:
                if keyword in full_text:
                    topics.append(category)
                    break  # Add category only once
        
        return list(set(topics))
    
    def _extract_question_topics(self, question: str) -> List[str]:
        """Extract topics from user question"""
        topics = []
        question_lower = question.lower()
        
        for category, keywords in self.document_categories.items():
            for keyword in keywords:
                if keyword in question_lower:
                    topics.append(category)
                    break
        
        return topics
    
    def _check_topic_relevance(self, question_topics: List[str], document_topics: List[str]) -> bool:
        """Check if question topics are relevant to document topics"""
        if not document_topics:  # No specific topics detected in documents
            return True
        
        if not question_topics:  # No specific topics in question
            return True
        
        # Check if any question topic matches document topics
        return any(topic in document_topics for topic in question_topics)
    
    def validate_request(self, message: str, images: List[str] = None, document_context: List[str] = None) -> Dict[str, Any]:
        """Enhanced validation with document relevance checking"""
        # Basic safety checks
        toxicity_check = self.check_toxicity(message)
        nsfw_check = self.check_nsfw_request(message, images)
        
        # Document relevance check
        relevance_check = self.check_document_relevance(message, document_context or [])
        
        is_safe = toxicity_check["safe"] and not nsfw_check and relevance_check["is_relevant"]
        
        rejection_reason = None
        if not is_safe:
            if not toxicity_check["safe"]:
                rejection_reason = "Toxic content detected"
            elif nsfw_check:
                rejection_reason = "NSFW content requested"
            elif not relevance_check["is_relevant"]:
                rejection_reason = f"Question is not relevant to uploaded documents. Document topics: {', '.join(relevance_check['document_topics'])}"
        
        return {
            "safe": is_safe,
            "toxicity_score": toxicity_check["score"],
            "is_nsfw": nsfw_check,
            "is_relevant": relevance_check["is_relevant"],
            "document_topics": relevance_check["document_topics"],
            "question_topics": relevance_check["question_topics"],
            "rejection_reason": rejection_reason
        }