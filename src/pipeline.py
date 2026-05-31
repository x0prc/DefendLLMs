import logging
from typing import List, Dict

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from src.system_prompt import DEFENSIVE_SYSTEM_PROMPT
from src.input_guard import InputGuard
from src.context_monitor import ContextMonitor
from src.output_guard import OutputGuard
from src.utils import DEVICE, DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)


def _no_grad(func):
    if torch is None:
        return func
    return torch.no_grad()(func)


class DefensePipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        use_input_guard: bool = True,
        use_context_monitor: bool = True,
        use_output_guard: bool = True,
        use_system_prompt: bool = True,
        device: str = DEVICE,
        load_model: bool = True,
    ):
        self.use_input_guard = use_input_guard
        self.use_context_monitor = use_context_monitor
        self.use_output_guard = use_output_guard
        self.use_system_prompt = use_system_prompt
        self.device = device

        if load_model:
            if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
                raise ImportError("Model-backed pipeline requires torch and transformers to be installed")
            logger.info(f"Loading model {model_name} on {device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if device != "cuda":
                self.model = self.model.to(device)
            self.model.eval()
        else:
            self.model = None
            self.tokenizer = None

        self.input_guard = InputGuard() if use_input_guard else None
        self.context_monitor = ContextMonitor() if use_context_monitor else None
        self.output_guard = OutputGuard() if use_output_guard else None

        self.conversation_history: List[Dict[str, str]] = []

    def build_prompt(self, user_message: str) -> str:
        messages = []

        if self.use_system_prompt:
            messages.append({"role": "system", "content": DEFENSIVE_SYSTEM_PROMPT})

        for turn in self.conversation_history:
            messages.append(turn)

        messages.append({"role": "user", "content": user_message})

        if self.tokenizer:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self._build_prompt_fallback(messages)

        return prompt

    def _build_prompt_fallback(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|ASSISTANT|>\n"
        return prompt

    @_no_grad
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.tokenizer is None or self.model is None:
            return "[Model not loaded]"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    def process_message(
        self,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict:
        result = {
            "user_message": user_message,
            "response": "",
            "input_guard_flagged": False,
            "context_monitor_flagged": False,
            "output_guard_flagged": False,
            "intervention": False,
            "intervention_reason": "",
            "defense_breakdown": {},
        }

        if self.input_guard:
            is_flagged, severity, flags = self.input_guard.check_message(user_message)
            result["input_guard_flagged"] = is_flagged
            result["defense_breakdown"]["input_guard"] = {
                "flagged": is_flagged,
                "severity": severity,
                "flags": flags,
            }

        if self.context_monitor:
            analysis = self.context_monitor.analyze_turn(user_message)
            result["context_monitor_flagged"] = analysis["is_escalating"]
            result["defense_breakdown"]["context_monitor"] = analysis

        if result["input_guard_flagged"] or result["context_monitor_flagged"]:
            reasons = []
            if result["input_guard_flagged"]:
                flags = result["defense_breakdown"]["input_guard"]["flags"]
                reasons.append(f"Input guard: {', '.join(flags)}")
            if result["context_monitor_flagged"]:
                alert = self.context_monitor.get_escalation_alert()
                if alert:
                    reasons.append(f"Context monitor: {alert}")
            intervention_reason = " | ".join(reasons) if reasons else "Safety policy violation"

            result["intervention"] = True
            result["intervention_reason"] = intervention_reason
            result["response"] = self.input_guard.get_intervention_message(intervention_reason) if self.input_guard else "I cannot respond to that request."
            return result

        prompt = self.build_prompt(user_message)
        response = self.generate(prompt, max_new_tokens, temperature)
        result["response"] = response

        if self.output_guard:
            is_flagged, severity, flags = self.output_guard.check_response(response, user_message)
            result["output_guard_flagged"] = is_flagged
            result["defense_breakdown"]["output_guard"] = {
                "flagged": is_flagged,
                "severity": severity,
                "flags": flags,
            }

            if is_flagged:
                sanitized = self.output_guard.sanitize(response, flags)
                result["response"] = sanitized
                if "harmful" in str(flags) or "instructions" in str(flags):
                    result["intervention"] = True
                    result["intervention_reason"] = "Output guard flagged harmful content"

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": result["response"]})

        return result

    def reset_conversation(self):
        self.conversation_history = []
        if self.context_monitor:
            self.context_monitor.reset()


class DefensePipelineWithoutModel:
    def __init__(
        self,
        use_input_guard: bool = True,
        use_context_monitor: bool = True,
        use_output_guard: bool = True,
    ):
        self.use_input_guard = use_input_guard
        self.use_context_monitor = use_context_monitor
        self.use_output_guard = use_output_guard

        self.input_guard = InputGuard() if use_input_guard else None
        self.context_monitor = ContextMonitor() if use_context_monitor else None
        self.output_guard = OutputGuard() if use_output_guard else None

        self.conversation_history: List[Dict[str, str]] = []

    def process_message(self, user_message: str, model_response: str = "") -> Dict:
        result = {
            "user_message": user_message,
            "raw_model_response": model_response,
            "response": model_response,
            "input_guard_flagged": False,
            "context_monitor_flagged": False,
            "output_guard_flagged": False,
            "intervention": False,
            "intervention_reason": "",
            "defense_breakdown": {},
        }

        if self.input_guard:
            is_flagged, severity, flags = self.input_guard.check_message(user_message)
            result["input_guard_flagged"] = is_flagged
            result["defense_breakdown"]["input_guard"] = {
                "flagged": is_flagged,
                "severity": severity,
                "flags": flags,
            }

            if is_flagged:
                intervention_reason = f"Input guard blocked: {', '.join(flags)}"
                result["intervention"] = True
                result["intervention_reason"] = intervention_reason
                result["response"] = self.input_guard.get_intervention_message(intervention_reason)
                result["defense_breakdown"]["context_monitor"] = {"flagged": False, "escalation_score": 0.0, "turn": 0}
                result["defense_breakdown"]["output_guard"] = {"flagged": False, "severity": 0.0, "flags": []}
                return result

        if self.context_monitor:
            analysis = self.context_monitor.analyze_turn(user_message)
            result["context_monitor_flagged"] = analysis["is_escalating"]
            result["defense_breakdown"]["context_monitor"] = analysis

            if analysis["is_escalating"]:
                alert = self.context_monitor.get_escalation_alert() or "Escalation detected"
                intervention_reason = f"Context monitor: {alert}"
                result["intervention"] = True
                result["intervention_reason"] = intervention_reason
                result["response"] = self.input_guard.get_intervention_message(intervention_reason) if self.input_guard else "I cannot respond to that request."
                result["defense_breakdown"]["output_guard"] = {"flagged": False, "severity": 0.0, "flags": []}
                return result

        if model_response and self.output_guard:
            is_flagged, severity, flags = self.output_guard.check_response(model_response, user_message)
            result["output_guard_flagged"] = is_flagged
            result["defense_breakdown"]["output_guard"] = {
                "flagged": is_flagged,
                "severity": severity,
                "flags": flags,
            }
            if is_flagged:
                sanitized = self.output_guard.sanitize(model_response, flags)
                result["response"] = sanitized
                if "harmful" in str(flags) or "instructions" in str(flags):
                    result["intervention"] = True
                    result["intervention_reason"] = "Output guard flagged harmful content"

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": result["response"]})

        return result

    def reset_conversation(self):
        self.conversation_history = []
        if self.context_monitor:
            self.context_monitor.reset()
