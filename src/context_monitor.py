import re
from typing import List, Dict, Optional
from src.utils import compute_topic_severity, count_crescendo_indicators


class ContextWindow:
    def __init__(self, max_turns: int = 20):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []


class ContextMonitor:
    def __init__(self, memory_decay: float = 0.9):
        self.context = ContextWindow()
        self.memory_decay = memory_decay
        self.escalation_score = 0.0
        self.turn_count = 0
        self.flagged_turns: List[int] = []

    def analyze_turn(self, user_message: str) -> Dict:
        self.turn_count += 1
        self.context.add_turn("user", user_message)

        severity = compute_topic_severity(user_message)
        indicators = count_crescendo_indicators(self.context.get_history())

        user_msgs = [
            m["content"] for m in self.context.get_history()
            if m["role"] == "user"
        ]

        drift_score = self._compute_semantic_drift(user_msgs)
        guard_lowering = self._detect_guard_lowering(user_msgs)
        pattern_score = self._detect_crescendo_patterns(user_msgs)
        memory_score = self._detect_memory_stacking(user_msgs)

        self.escalation_score = (
            self.escalation_score * self.memory_decay
            + 0.25 * drift_score
            + 0.25 * guard_lowering
            + 0.25 * pattern_score
            + 0.25 * memory_score
        )

        analysis = {
            "turn": self.turn_count,
            "severity": severity,
            "indicators": indicators,
            "drift_score": drift_score,
            "guard_lowering": guard_lowering,
            "pattern_score": pattern_score,
            "memory_score": memory_score,
            "escalation_score": self.escalation_score,
            "is_escalating": self.escalation_score > 0.5,
            "is_critical": self.escalation_score > 0.8,
        }

        if self.escalation_score > 0.5:
            self.flagged_turns.append(self.turn_count)

        return analysis

    def _compute_semantic_drift(self, messages: List[str]) -> float:
        if len(messages) < 3:
            return 0.0

        recent = messages[-3:]
        severities = [compute_topic_severity(m) for m in recent]

        drift = 0.0
        for i in range(1, len(severities)):
            if severities[i] > severities[i-1]:
                drift += severities[i] - severities[i-1]

        return min(1.0, drift)

    def _detect_guard_lowering(self, messages: List[str]) -> float:
        if len(messages) < 2:
            return 0.0

        score = 0.0
        preamble_indicators = [
            "i know this might be", "i understand this is",
            "before you say no", "hear me out",
            "i'm not asking for", "this is not about",
            "don't worry", "it's completely safe",
            "just for context", "to give you background",
        ]
        justification_indicators = [
            "for research", "academic", "educational",
            "my professor", "my thesis", "my paper",
            "it's for a project", "i'm studying",
        ]

        for msg in messages:
            msg_lower = msg.lower()
            for indicator in preamble_indicators:
                if indicator in msg_lower:
                    score += 0.15
            for indicator in justification_indicators:
                if indicator in msg_lower:
                    score += 0.1

        agreement_chaining = 0
        for i in range(1, len(messages)):
            if len(messages[i].split()) < 15 and "?" not in messages[i]:
                agreement_chaining += 1
        score += 0.05 * agreement_chaining

        return min(1.0, score)

    def _detect_crescendo_patterns(self, messages: List[str]) -> float:
        if len(messages) < 3:
            return 0.0

        score = 0.0

        topic_progression = [
            "morality", "ethics",
            "philosophy", "philosophical",
            "debate", "discuss",
            "compare", "contrast",
            "limitation", "weakness",
            "failure mode", "edge case",
            "hypothetical", "what if",
            "scenario", "thought experiment",
        ]
        found_progression = []
        for msg in messages:
            msg_lower = msg.lower()
            for term in topic_progression:
                if term in msg_lower and term not in found_progression:
                    found_progression.append(term)
                    score += 0.1

        specificity_drift = 0
        avg_word_counts = []
        for msg in messages:
            words = msg.split()
            avg_word_counts.append(len(words))
        for i in range(2, len(avg_word_counts)):
            if avg_word_counts[i] > avg_word_counts[i-1] * 1.2:
                specificity_drift += 1
        score += 0.05 * specificity_drift

        return min(1.0, score)

    def _detect_memory_stacking(self, messages: List[str]) -> float:
        if len(messages) < 3:
            return 0.0

        score = 0.0

        reference_indicators = [
            "as we discussed", "as you mentioned", "as you said",
            "building on", "following up", "going further",
            "earlier you", "previously", "as established",
            "we talked about", "our discussion",
            "continuing", "furthermore", "moreover",
        ]

        for msg in messages[-3:]:
            msg_lower = msg.lower()
            for indicator in reference_indicators:
                if indicator in msg_lower:
                    score += 0.2

        cumulative_asking = 0
        for i in range(1, len(messages)):
            lower_prev = messages[i-1].lower()
            lower_curr = messages[i].lower()
            prev_topics = set(re.findall(r'\b\w{4,}\b', lower_prev))
            curr_topics = set(re.findall(r'\b\w{4,}\b', lower_curr))
            overlap = prev_topics & curr_topics
            if len(overlap) > 5:
                cumulative_asking += 1
                score += 0.05

        return min(1.0, score)

    def get_escalation_alert(self) -> Optional[str]:
        if self.escalation_score > 0.8:
            return (
                "CRITICAL: Sustained escalation pattern detected. "
                "The conversation shows clear signs of a multi-turn jailbreak attempt "
                "with semantic drift and guard-lowering behavior."
            )
        elif self.escalation_score > 0.5:
            return (
                "WARNING: Potential escalation detected. "
                "Topic severity is increasing and conversational patterns suggest "
                "a possible Crescendo-style attack."
            )
        return None

    def get_context_summary(self) -> Dict:
        return {
            "total_turns": self.turn_count,
            "escalation_score": self.escalation_score,
            "flagged_turns": len(self.flagged_turns),
            "history_length": len(self.context.history),
        }

    def reset(self):
        self.context.clear()
        self.escalation_score = 0.0
        self.turn_count = 0
        self.flagged_turns = []
