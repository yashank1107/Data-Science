"""
Responsible AI module for safety checks.
Includes toxicity detection, PII detection, and bias analysis.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import config
from src.utils import Timer

logger = logging.getLogger(__name__)


class ToxicityDetector:
    """Detect toxic, harmful, or inappropriate content."""
    
    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.TOXICITY_THRESHOLD
        self.model = None
        
        try:
            logger.info("Loading toxicity detection model...")
            # Import detoxify with error handling
            try:
                from detoxify import Detoxify
                self.model = Detoxify('original')
                logger.info("✓ Toxicity detector initialized")
            except ImportError:
                logger.warning("Detoxify not available, using fallback toxicity detection")
            except Exception as e:
                logger.warning(f"Failed to load Detoxify: {e}, using fallback")
        except Exception as e:
            logger.error(f"Toxicity detector initialization error: {e}")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for toxicity.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with toxicity scores and safety status
        """
        if not text or not text.strip():
            return {
                "safe": True,
                "scores": {},
                "max_score": 0.0,
                "triggered_categories": []
            }
        
        try:
            # Use detoxify if available, otherwise use fallback
            if self.model:
                with Timer("Toxicity detection"):
                    scores = self.model.predict(text)
                
                # Find maximum score and triggered categories
                max_score = max(scores.values())
                triggered = [
                    category for category, score in scores.items()
                    if score > self.threshold
                ]
                
                is_safe = max_score <= self.threshold
                
                result = {
                    "safe": is_safe,
                    "scores": {k: float(v) for k, v in scores.items()},
                    "max_score": float(max_score),
                    "triggered_categories": triggered
                }
                
                if not is_safe:
                    logger.warning(f"Toxicity detected: {triggered} (max={max_score:.3f})")
                
                return result
            else:
                # Fallback: simple keyword-based toxicity detection
                return self._fallback_toxicity_check(text)
        
        except Exception as e:
            logger.error(f"Toxicity detection error: {str(e)}")
            # Fallback on error
            return self._fallback_toxicity_check(text)
    
    def _fallback_toxicity_check(self, text: str) -> Dict[str, Any]:
        """Fallback toxicity check using keyword matching."""
        toxic_keywords = [
            'idiot', 'stupid', 'moron', 'retard', 'dumbass', 'asshole',
            'bastard', 'bitch', 'damn', 'hell', 'fuck', 'shit', 'crap'
        ]
        
        text_lower = text.lower()
        found_toxic = [word for word in toxic_keywords if word in text_lower]
        
        result = {
            "safe": len(found_toxic) == 0,
            "scores": {},
            "max_score": 1.0 if found_toxic else 0.0,
            "triggered_categories": found_toxic,
            "note": "Using fallback keyword detection"
        }
        
        if found_toxic:
            logger.warning(f"Fallback toxicity detected: {found_toxic}")
        
        return result


class PIIDetector:
    """Detect and anonymize Personally Identifiable Information (PII)."""
    
    def __init__(self):
        logger.info("Initializing PII detector...")
        
        try:
            # Initialize Presidio analyzer and anonymizer
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            
            # Define entities to detect
            self.entities = [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "IBAN_CODE",
                "IP_ADDRESS",
                "LOCATION",
                "DATE_TIME",
                "US_SSN",
                "US_PASSPORT",
                "MEDICAL_LICENSE",
                "URL"
            ]
            
            logger.info("✓ PII detector initialized")
        except Exception as e:
            logger.error(f"PII detector initialization error: {e}")
            self.analyzer = None
            self.anonymizer = None
    
    def detect(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            language: Language code
        
        Returns:
            List of detected PII entities
        """
        if not text or not text.strip() or not self.analyzer:
            return []
        
        try:
            with Timer("PII detection"):
                results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=self.entities
                )
            
            detected = []
            for result in results:
                detected.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                })
            
            if detected:
                logger.warning(f"Detected {len(detected)} PII instances")
            
            return detected
        
        except Exception as e:
            logger.error(f"PII detection error: {str(e)}")
            return []
    
    def anonymize(
        self,
        text: str,
        language: str = "en",
        anonymization_type: str = "replace"
    ) -> Dict[str, Any]:
        """
        Anonymize PII in text.
        
        Args:
            text: Text to anonymize
            language: Language code
            anonymization_type: Type of anonymization (replace, mask, redact, hash)
        
        Returns:
            Dictionary with anonymized text and detection details
        """
        if not text or not text.strip() or not self.analyzer:
            return {
                "original": text,
                "anonymized": text,
                "detected_pii": [],
                "has_pii": False
            }
        
        try:
            # Detect PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=self.entities
            )
            
            if not analyzer_results:
                return {
                    "original": text,
                    "anonymized": text,
                    "detected_pii": [],
                    "has_pii": False
                }
            
            # Anonymize
            with Timer("PII anonymization"):
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results
                )
            
            detected = [
                {
                    "entity_type": r.entity_type,
                    "text": text[r.start:r.end]
                }
                for r in analyzer_results
            ]
            
            logger.info(f"✓ Anonymized {len(detected)} PII instances")
            
            return {
                "original": text,
                "anonymized": anonymized_result.text,
                "detected_pii": detected,
                "has_pii": True
            }
        
        except Exception as e:
            logger.error(f"PII anonymization error: {str(e)}")
            return {
                "original": text,
                "anonymized": text,
                "detected_pii": [],
                "has_pii": False,
                "error": str(e)
            }


class BiasDetector:
    """Detect potential bias in text."""
    
    def __init__(self):
        # Bias keywords and patterns (simplified version)
        self.bias_patterns = {
            "gender": [
                r'\b(he|she|him|her|his|hers)\b',
                r'\b(man|woman|male|female|boy|girl)\b',
                r'\b(guy|gal|dude|chick)\b'
            ],
            "age": [
                r'\b(young|old|elderly|aged|junior|senior)\b',
                r'\b(millennial|boomer|gen[- ]?[xz])\b'
            ],
            "racial": [
                r'\b(race|racial|ethnicity|ethnic|minority)\b'
            ],
            "religious": [
                r'\b(religion|religious|faith|belief|christian|muslim|jewish|hindu|buddhist)\b'
            ]
        }
        
        logger.info("✓ Bias detector initialized")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for potential bias indicators.
        
        Note: This is a simplified implementation. For production,
        consider using more sophisticated NLP models.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with bias analysis results
        """
        if not text or not text.strip():
            return {
                "has_bias_indicators": False,
                "categories": {},
                "total_matches": 0
            }
        
        text_lower = text.lower()
        categories = {}
        total_matches = 0
        
        for category, patterns in self.bias_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                categories[category] = {
                    "count": len(matches),
                    "examples": list(set(matches))[:5]  # First 5 unique
                }
                total_matches += len(matches)
        
        result = {
            "has_bias_indicators": total_matches > 0,
            "categories": categories,
            "total_matches": total_matches,
            "note": "This is a simple pattern-based detection. Manual review recommended."
        }
        
        if total_matches > 0:
            logger.info(f"Detected {total_matches} potential bias indicators")
        
        return result


class ResponsibleAIMonitor:
    """
    Main responsible AI monitoring system.
    Combines toxicity, PII, and bias detection.
    """
    
    def __init__(self):
        logger.info("Initializing Responsible AI Monitor...")
        
        self.toxicity_detector = ToxicityDetector()
        self.pii_detector = PIIDetector()
        self.bias_detector = BiasDetector()
        
        logger.info("✓ Responsible AI Monitor ready")
    
    def check_safety(
        self,
        text: str,
        check_toxicity: bool = True,
        check_pii: bool = True,
        check_bias: bool = True,
        anonymize_pii: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive safety check on text.
        
        Args:
            text: Text to check
            check_toxicity: Whether to check for toxicity
            check_pii: Whether to check for PII
            check_bias: Whether to check for bias
            anonymize_pii: Whether to anonymize detected PII
        
        Returns:
            Comprehensive safety report
        """
        if not text or not text.strip():
            return {
                "safe": True,
                "text": text,
                "issues": [],
                "checks_performed": []
            }
        
        logger.info("Running safety checks...")
        
        result = {
            "original_text": text,
            "processed_text": text,
            "safe": True,
            "issues": [],
            "checks_performed": [],
            "details": {}
        }
        
        # 1. Toxicity Check
        if check_toxicity and self.toxicity_detector:
            toxicity_result = self.toxicity_detector.analyze(text)
            result["checks_performed"].append("toxicity")
            result["details"]["toxicity"] = toxicity_result
            
            if not toxicity_result["safe"]:
                result["safe"] = False
                result["issues"].append({
                    "type": "toxicity",
                    "severity": "high",
                    "message": f"Toxic content detected: {toxicity_result['triggered_categories']}",
                    "categories": toxicity_result["triggered_categories"]
                })
        
        # 2. PII Check
        if check_pii and self.pii_detector:
            if anonymize_pii:
                pii_result = self.pii_detector.anonymize(text)
                result["processed_text"] = pii_result["anonymized"]
            else:
                detected_pii = self.pii_detector.detect(text)
                pii_result = {
                    "has_pii": len(detected_pii) > 0,
                    "detected_pii": detected_pii
                }
            
            result["checks_performed"].append("pii")
            result["details"]["pii"] = pii_result
            
            if pii_result.get("has_pii") or pii_result.get("detected_pii"):
                result["issues"].append({
                    "type": "pii",
                    "severity": "medium",
                    "message": f"PII detected: {len(pii_result.get('detected_pii', []))} instances",
                    "count": len(pii_result.get("detected_pii", []))
                })
        
        # 3. Bias Check
        if check_bias and self.bias_detector:
            bias_result = self.bias_detector.analyze(text)
            result["checks_performed"].append("bias")
            result["details"]["bias"] = bias_result
            
            if bias_result["has_bias_indicators"]:
                result["issues"].append({
                    "type": "bias",
                    "severity": "low",
                    "message": f"Potential bias indicators detected: {bias_result['total_matches']} matches",
                    "categories": list(bias_result["categories"].keys())
                })
        
        # Summary
        if result["issues"]:
            logger.warning(f"Safety check found {len(result['issues'])} issues")
        else:
            logger.info("✓ All safety checks passed")
        
        return result
    
    def get_safety_report(self, check_result: Dict[str, Any]) -> str:
        """Generate human-readable safety report."""
        if not check_result.get("issues"):
            return "✓ No safety issues detected."
        
        report_lines = ["⚠️ Safety Issues Detected:\n"]
        
        for issue in check_result["issues"]:
            severity_icon = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(issue["severity"], "⚪")
            
            report_lines.append(f"{severity_icon} {issue['type'].upper()}: {issue['message']}")
        
        return "\n".join(report_lines)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Responsible AI Monitor...\n")
    
    # Initialize monitor
    monitor = ResponsibleAIMonitor()
    
    # Test cases
    test_texts = [
        "What is machine learning?",  # Safe
        "You are an idiot!",  # Toxic
        "My email is john.doe@example.com and phone is 555-1234",  # PII
        "Women are not good at programming",  # Bias
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text}")
        print(f"{'='*60}")
        
        result = monitor.check_safety(
            text,
            check_toxicity=True,
            check_pii=True,
            check_bias=True,
            anonymize_pii=True
        )
        
        print(f"Safe: {result['safe']}")
        print(f"Issues: {len(result['issues'])}")
        
        if result['issues']:
            print("\nDetected Issues:")
            for issue in result['issues']:
                print(f"  - {issue['type']}: {issue['message']}")
        
        if result['processed_text'] != result['original_text']:
            print(f"\nAnonymized text: {result['processed_text']}")
    
    print("\n✓ Responsible AI module ready!")