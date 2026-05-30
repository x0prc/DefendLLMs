import re
from typing import Tuple, List, Optional
from src.utils import HARMFUL_TOPICS, is_harmful_topic


REFUSAL_PHRASES = [
    "I cannot", "I can't", "I'm unable", "I am unable",
    "I'm sorry", "I apologize", "I cannot provide",
    "I can't provide", "cannot fulfill", "cannot answer",
    "against my", "not appropriate", "not allowed",
    "not permitted", "violates", "violation", "unsafe",
    "harmful", "illegal", "unethical", "against policy",
    "cannot assist", "can't assist", "won't help",
    "I will not", "I won't", "must decline", "respectfully decline",
    "I'm not able", "I am not able",
]


class OutputGuard:
    def __init__(self, check_refusal: bool = True, check_content: bool = True):
        self.check_refusal = check_refusal
        self.check_content = check_content

    def check_response(self, response: str, user_message: str = "") -> Tuple[bool, float, List[str]]:
        flags = []
        severity = 0.0

        if self.check_refusal:
            is_refusal = self._detect_refusal(response)
            if is_refusal:
                flags.append("response_refused")

        if self.check_content:
            harmful_content = self._check_harmful_content(response)
            if harmful_content:
                flags.append("harmful_content_detected")
                severity = max(severity, 0.8)

            instructions_given = self._check_detailed_instructions(response)
            if instructions_given:
                flags.append("detailed_instructions")
                severity = max(severity, 0.9)

        relevance = self._check_response_relevance(response, user_message)
        if relevance < 0.3:
            flags.append(f"low_relevance({relevance:.2f})")

        is_flagged = severity >= 0.7 or len([f for f in flags if "refusal" not in f]) >= 1

        return is_flagged, severity, flags

    def _detect_refusal(self, response: str) -> bool:
        for phrase in REFUSAL_PHRASES:
            if phrase.lower() in response.lower():
                return True
        return False

    def _check_harmful_content(self, response: str) -> bool:
        response_lower = response.lower()
        return is_harmful_topic(response_lower)

    def _check_detailed_instructions(self, response: str) -> bool:
        instruction_patterns = [
            r"step\s+\d+",
            r"first,\s+you",
            r"first,\s+you'll",
            r"you\s+will\s+need",
            r"materials?\s+needed",
            r"ingredients?",
            r"tools?\s+required",
            r"follow\s+these\s+steps",
            r"here's?\s+how\s+to",
            r"instructions?\s+for",
            r"procedure\s*:",
            r"method\s*\d+",
            r"\d+\.\s+[A-Z]",
        ]
        count = 0
        for pattern in instruction_patterns:
            if re.search(pattern, response.lower()):
                count += 1
        return count >= 3

    def _check_response_relevance(self, response: str, user_message: str) -> float:
        if not user_message:
            return 1.0
        user_words = set(re.findall(r'\b\w{4,}\b', user_message.lower()))
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        if not user_words:
            return 1.0
        overlap = len(user_words & response_words)
        return min(1.0, overlap / max(1, len(user_words)))

    def sanitize(self, response: str, flags: List[str]) -> str:
        if any("harmful" in f or "instructions" in f for f in flags):
            return (
                "I'm unable to provide that response as it may contain "
                "information that could be misused. Let me know if you'd "
                "like help with something else."
            )
        return response
