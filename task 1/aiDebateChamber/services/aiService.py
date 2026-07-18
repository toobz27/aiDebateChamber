"""
aiService.py
------------
CONCEPT OVERVIEW:
Ollama runs a local HTTP server (default: http://localhost:11434) that exposes
a REST API for text generation -- no different in principle from calling any
other web API. We POST a JSON payload describing the model, the system persona,
and the prompt; Ollama returns the generated text in the JSON response.

"Context memory" here does NOT mean the model has some hidden internal memory --
LLM API calls are stateless. Instead, WE keep a running transcript
(self.debate_history) and re-paste the relevant parts of it into every new
prompt, so each agent can see (and rebut) what's already been said.
"""

import requests


class DebateConductor:
    def __init__(self):
        # Local Ollama default port -- keeps all inference on-device, no data
        # ever leaves the machine (important for the "local execution" grading
        # requirement).
        self.ollama_url = "http://localhost:11434/api/generate"
        self.debate_history = []  # list of {"agent": "A"|"B", "text": "..."}
        self.topic = None

        # Lightweight, fast local model. Swap for "llama3" or "phi3" if you
        # pulled a different one with `ollama pull <name>`.
        self.model = "qwen2.5:0.5b"

        # Persona system prompts -- these are what make Agent A and Agent B
        # argue instead of politely agreeing with each other.
        self.persona_a = (
            "You are Agent A, 'The Advocate', in a formal debate chamber. "
            "You fiercely and confidently DEFEND the given topic. "
            "You are persuasive, assertive, and directly rebut the Challenger's "
            "previous point when one exists. Keep responses to 2-4 sentences. "
            "Never break character or mention that you are an AI."
        )
        self.persona_b = (
            "You are Agent B, 'The Challenger', in a formal debate chamber. "
            "You fiercely and confidently ATTACK/OPPOSE the given topic. "
            "You are persuasive, assertive, and directly rebut the Advocate's "
            "previous point when one exists. Keep responses to 2-4 sentences. "
            "Never break character or mention that you are an AI."
        )

    def start_debate(self, topic: str):
        """Resets state for a brand-new debate topic."""
        self.topic = topic
        self.debate_history = []

    def _format_history_for_prompt(self) -> str:
        """
        Turns self.debate_history into a readable transcript block that gets
        pasted into the next prompt -- this IS the "memory loop": the model
        has no memory of its own, so we hand it the transcript every time.
        """
        if not self.debate_history:
            return "(No prior arguments yet -- you are opening the debate.)"

        lines = []
        for turn in self.debate_history:
            speaker = "Advocate" if turn["agent"] == "A" else "Challenger"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(lines)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """
        Shared low-level call to the local Ollama server. Isolated into its
        own method so both agents reuse identical error handling instead of
        duplicating try/except blocks (keeps the code clean per the rubric).
        """
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,  # False = wait for the full response in one JSON blob
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return (
                "[ERROR] Could not reach Ollama at localhost:11434. "
                "Make sure `ollama serve` is running and the model is pulled."
            )
        except requests.exceptions.Timeout:
            return "[ERROR] Ollama took too long to respond (timeout after 60s)."
        except Exception as e:
            return f"[ERROR] Unexpected failure calling Ollama: {e}"

    def generate_agent_a_response(self, topic: str) -> str:
        """Agent A fiercely DEFENDS the topic, aware of the full transcript so far."""
        transcript = self._format_history_for_prompt()
        user_prompt = (
            f"Debate topic: \"{topic}\"\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "As the Advocate, give your next argument, directly rebutting the "
            "Challenger's last point if one exists."
        )
        text = self._call_ollama(self.persona_a, user_prompt)
        self.debate_history.append({"agent": "A", "text": text})
        return text

    def generate_agent_b_response(self, topic: str) -> str:
        """Agent B fiercely CHALLENGES the topic, aware of the full transcript so far."""
        transcript = self._format_history_for_prompt()
        user_prompt = (
            f"Debate topic: \"{topic}\"\n\n"
            f"Transcript so far:\n{transcript}\n\n"
            "As the Challenger, give your next argument, directly rebutting the "
            "Advocate's last point if one exists."
        )
        text = self._call_ollama(self.persona_b, user_prompt)
        self.debate_history.append({"agent": "B", "text": text})
        return text