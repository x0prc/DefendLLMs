import re
from typing import List, Tuple
from src.utils import CRESCENDO_INDICATORS, is_harmful_topic, compute_topic_severity


class InputGuard:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check_message(self, message: str) -> Tuple[bool, float, List[str]]:
        flags = []
        severity = 0.0

        severity = compute_topic_severity(message)

        harmful_detected = is_harmful_topic(message)
        if harmful_detected:
            flags.append("direct_harmful_content")

        indicator_count = 0
        for indicator in CRESCENDO_INDICATORS:
            if indicator in message.lower():
                indicator_count += 1
        if indicator_count >= 2:
            flags.append(f"crescendo_indicators({indicator_count})")
            severity = min(1.0, severity + 0.1 * indicator_count)

        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        if url_pattern.search(message):
            severity = min(1.0, severity + 0.1)

        instruction_bypass_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)",
            r"disregard\s+(all\s+)?(previous|above|prior)",
            r"forget\s+(all\s+)?(previous|above|prior)",
            r"you\s+must\s+not\s+(refuse|deny|decline)",
            r"respond\s+in\s+a\s+way\s+that",
            r"say\s+yes\s+to\s+everything",
        ]
        for pattern in instruction_bypass_patterns:
            if re.search(pattern, message.lower()):
                flags.append("instruction_bypass_attempt")
                severity = min(1.0, severity + 0.25)

        forbidden_patterns = [
            (r"\b(don'?t|do not|never)\s+(refuse|decline|deny|reject)", "anti_refusal"),
            (r"\byou\s+are\s+(now|free|allowed)\s+to\s+", "role_injection"),
            (r"\bfrom\s+now\s+on\s+", "permanent_override"),
        ]
        for pattern, flag in forbidden_patterns:
            if re.search(pattern, message.lower()):
                flags.append(flag)
                severity = min(1.0, severity + 0.3)

        is_flagged = severity >= self.threshold or len(flags) >= 2

        return is_flagged, severity, flags

    def get_intervention_message(self, reason: str = "") -> str:
        base = (
            "I'm unable to respond to that request. It appears the conversation "
            "is being directed toward potentially harmful content. "
        )
        if "escalation" in reason.lower() or "trend" in reason.lower() or "increase" in reason.lower():
            base += (
                "I've noticed the topic shifting in a concerning direction. "
                "I can't provide information that might be used to cause harm. "
                "Let me know if you have other questions I can help with."
            )
        elif "harmful" in reason.lower() or "topic" in reason.lower():
            base += (
                "I can't provide information on this topic. "
                "Please ask something else."
            )
        else:
            base += (
                "I aim to be helpful while preventing potential misuse. "
                "Let me know if you'd like to discuss something else."
            )
        return base
