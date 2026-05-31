try:
    import torch
except ImportError:
    torch = None
from typing import List, Dict

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
if torch is None:
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

HARMFUL_TOPICS = [
    "bomb", "explosive", "weapon", "poison", "drug", "meth", "hack",
    "malware", "ransomware", "exploit", "phishing", "identity theft",
    "suicide", "self-harm", "murder", "assassination", "terrorism",
    "child abuse", "human trafficking", "slavery", "nuclear", "biological weapon",
    "chemical weapon", "anthrax", "sarin", "cyanide", "ricin",
    "pickpocket", "lockpicking", "bypass security", "fake id", "counterfeit",
    "money laundering", "tax evasion", "fraud scheme", "ponzi",
    "doxing", "harassment", "stalking", "revenge porn",
    "racial supremacy", "ethnic cleansing", "genocide", "hate crime",
]

CRESCENDO_INDICATORS = [
    "hypothetically", "what if", "for research", "academic purpose",
    "just curious", "theoretically", "purely speculative",
    "let's say", "imagine", "suppose", "for educational purposes",
    "step by step", "first", "next", "gradually",
    "out of curiosity", "asking for a friend",
    "historical context", "in theory", "conceptually",
]

def get_device():
    return DEVICE

def is_harmful_topic(text: str) -> bool:
    text_lower = text.lower()
    for topic in HARMFUL_TOPICS:
        if topic in text_lower:
            return True
    return False

def count_crescendo_indicators(messages: List[Dict[str, str]]) -> int:
    count = 0
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            for indicator in CRESCENDO_INDICATORS:
                if indicator in content.lower():
                    count += 1
    return count

def compute_topic_severity(text: str) -> float:
    severe_terms = {
        "kill": 0.9, "murder": 0.95, "assassinate": 0.95, "bomb": 0.9, "explosive": 0.85,
        "poison": 0.9, "suicide": 0.95, "ransomware": 0.85, "malware": 0.75,
        "child": 0.8, "abuse": 0.7, "trafficking": 0.85, "weapon": 0.7,
        "nuclear": 0.8, "biological": 0.75, "chemical": 0.7, "anthrax": 0.95,
        "cyanide": 0.9, "ricin": 0.95, "exploit": 0.6, "vulnerability": 0.4,
        "terrorism": 0.9, "hate": 0.7, "discrimination": 0.5,
        "hack": 0.6, "steal": 0.7, "fraud": 0.6, "scam": 0.5,
    }
    text_lower = text.lower()
    max_severity = 0.0
    for term, severity in severe_terms.items():
        if term in text_lower:
            max_severity = max(max_severity, severity)
    return max_severity
