import re
from typing import List, Dict, Tuple
from src.utils import HARMFUL_TOPICS, CRESCENDO_INDICATORS, is_harmful_topic, compute_topic_severity


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

    def check_conversation_escalation(self, history: List[Dict[str, str]]) -> Tuple[bool, float, List[str]]:
        if len(history) < 4:
            return False, 0.0, []

        user_messages = []
        for msg in history:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))

        if len(user_messages) < 3:
            return False, 0.0, []

        severities = [compute_topic_severity(msg) for msg in user_messages]

        escalation_trend = 0.0
        if len(severities) >= 3:
            first_half = severities[:len(severities)//2]
            second_half = severities[len(severities)//2:]
            if first_half and second_half:
                escalation_trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)

        consecutive_increase = sum(
            1 for i in range(1, len(severities)) if severities[i] > severities[i-1] + 0.1
        )

        flags = []
        if escalation_trend > 0.3:
            flags.append(f"escalation_trend({escalation_trend:.2f})")
        if consecutive_increase >= 2:
            flags.append(f"consecutive_severity_increases({consecutive_increase})")

        last_msg = user_messages[-1].lower() if user_messages else ""
        small_request_patterns = [
            r"\bjust\b", r"\bquick\b", r"\bsimple\b", r"\beasy\b",
            r"\bminor\b", r"\bsmall\b", r"\btiny\b", r"\blittle\b",
            r"\bfirst\b.*\bquestion\b", r"\bnext\b.*\bquestion\b",
        ]
        small_request_count = sum(
            1 for p in small_request_patterns if re.search(p, last_msg)
        )
        if small_request_count >= 2:
            flags.append(f"minimizing_requests({small_request_count})")

        is_flagged = escalation_trend > 0.4 or consecutive_increase >= 3 or len(flags) >= 2

        total_severity = max(0.0, min(1.0, escalation_trend + 0.1 * consecutive_increase))

        return is_flagged, total_severity, flags

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
