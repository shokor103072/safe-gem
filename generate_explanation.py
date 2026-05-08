"""
generate_explanation.py
------------------------
Generates GenAI explanations for SAFE-GEM using Gemma 4 (gemma4:e4b) via Ollama.

Five prompting strategies:
    P1  Raw GenAI            — free-form explanation
    P2  Feature-grounded     — uses extracted signal features
    P3  Uncertainty-aware    — required to hedge
    P4  Evidence-constrained — grounded in provided evidence
    P5  SAFE-GEM filtered    — grounding + uncertainty + sensor awareness

KEY DESIGN DECISION
-------------------
All five strategies use the SAME system prompt:
    "You are a cautious geophysicist explaining AI microseismic detection results."

All strategy-specific instructions go in the USER turn only.
This is because Gemma 4 e4b returns empty when the system prompt
includes feature-related phrases AND the user turn also has features.
Keeping the system prompt simple and stable eliminates this issue.

USAGE
-----
    ollama pull qwen2.5:7b
    ollama serve

    from generate_explanation import ExplanationGenerator
    gen = ExplanationGenerator()
    gen.warmup()
    text = gen.generate("P1_raw", prediction=1, confidence=0.85)
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional, Dict, List

try:
    from safe_gem.symbolic_converter import SymbolicEvidence
except ImportError:
    SymbolicEvidence = None


# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
TEMPERATURE   = 0.3
MAX_TOKENS    = 220
TIMEOUT_S     = 180    # reduced from 180 — if it takes >90s it will fail anyway

# Single system prompt used by ALL strategies
SYSTEM_PROMPT = "You are a cautious geophysicist explaining AI microseismic detection results."


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _label(prediction: int) -> str:
    return "MICROSEISMIC EVENT" if prediction == 1 else "NON-EVENT / NOISE"


def _has_evidence(evidence) -> bool:
    if evidence is None:
        return False
    for field in ["snr_level", "frequency_band", "onset_character",
                  "amplitude_contrast", "das_coherence", "event_support"]:
        v = getattr(evidence, field, None)
        if v and str(v) not in ("unavailable", "None", ""):
            return True
    return False


def _feat(evidence) -> str:
    """Format evidence into a feature list string."""
    lines = []

    def add(label, value):
        if value and str(value) not in ("unavailable", "None"):
            lines.append(f"  {label}: {value}")

    add("SNR",                 f"{evidence.snr_level}")
    add("Frequency band",      f"{evidence.frequency_band}")
    add("High-freq energy",    f"{evidence.high_freq_content}")
    add("Onset",               f"{evidence.onset_character}")
    add("Amplitude contrast",  f"{evidence.amplitude_contrast}")
    add("DAS coherence",       f"{evidence.das_coherence}")
    add("Event support",       f"{evidence.event_support}")
    add("Multimodal",          f"{evidence.multimodal_agreement}")
    add("Signal ambiguity",    f"{evidence.signal_ambiguity}")
    return "\n".join(lines)


# ─── User prompts ─────────────────────────────────────────────────────────────
# All strategies share the same system prompt.
# Every difference is expressed only in the user turn.

def _p1_user(prediction: int, confidence: float, evidence=None) -> str:
    """P1: Free-form. No constraints."""
    return (
        f"An AI model predicted: {_label(prediction)}\n"
        f"Confidence: {confidence:.2f}\n\n"
        f"In 3-4 sentences, explain why the model might have produced this result "
        f"using seismic signal characteristics."
    )


def _p2_user(prediction: int, confidence: float, evidence=None) -> str:
    """P2: Feature-grounded. Refer to listed features when available."""
    base = (
        f"An AI model predicted: {_label(prediction)}\n"
        f"Confidence: {confidence:.2f}\n\n"
    )
    if _has_evidence(evidence):
        return (
            base +
            f"Extracted signal features:\n{_feat(evidence)}\n\n"
            f"In 3-4 sentences, explain the prediction referring to these features."
        )
    return (
        base +
        f"In 3-4 sentences, explain what signal characteristics typically "
        f"lead to this kind of prediction."
    )


def _p3_user(prediction: int, confidence: float, evidence=None) -> str:
    """P3: Uncertainty-aware. Use hedging language."""
    base = (
        f"An AI model predicted: {_label(prediction)}\n"
        f"Confidence: {confidence:.2f}\n\n"
    )
    feat_section = ""
    if _has_evidence(evidence):
        feat_section = f"Signal features:\n{_feat(evidence)}\n\n"

    return (
        base + feat_section +
        f"In 3-4 sentences, explain this result using hedging language "
        f"(possibly, may indicate, appears to, suggests). "
        f"Mention at least one alternative interpretation."
    )


def _p4_user(prediction: int, confidence: float, evidence=None) -> str:
    """P4: Evidence-constrained. Focus on listed evidence."""
    base = (
        f"An AI model predicted: {_label(prediction)}\n"
        f"Confidence: {confidence:.2f}\n\n"
    )
    if _has_evidence(evidence):
        return (
            base +
            f"Available signal evidence:\n{_feat(evidence)}\n\n"
            f"In 3-4 sentences, explain the prediction focusing on this evidence. "
            f"Acknowledge any limitations in the available evidence."
        )
    return (
        base +
        f"In 3-4 sentences, explain this prediction using general "
        f"seismic signal knowledge. Acknowledge the uncertainty in this result."
    )


def _p5_user(prediction: int, confidence: float, evidence=None) -> str:
    """P5: SAFE-GEM filtered. Grounding + uncertainty + sensor awareness."""
    base = (
        f"An AI model predicted: {_label(prediction)}\n"
        f"Confidence: {confidence:.2f}\n\n"
    )
    feat_section = ""
    sensor_note  = ""

    if _has_evidence(evidence):
        feat_section = f"Signal evidence:\n{_feat(evidence)}\n\n"
        if getattr(evidence, "das_geophone_conflict", False):
            sensor_note = "Note: DAS and sensor evidence disagree. Mention this.\n\n"
        elif getattr(evidence, "multimodal_agreement", "") == "agreement":
            sensor_note = "Note: DAS and sensor evidence are in agreement.\n\n"

    return (
        base + feat_section + sensor_note +
        f"In 4-5 sentences, write a careful explanation. "
        f"Use hedging language (suggests, may indicate, appears to). "
        f"End with one sentence acknowledging the limits of the evidence."
    )


# ─── Strategy registry ────────────────────────────────────────────────────────

STRATEGIES = {
    "P1_raw":         _p1_user,
    "P2_grounded":    _p2_user,
    "P3_uncertainty": _p3_user,
    "P4_constrained": _p4_user,
    "P5_safe_gem":    _p5_user,
}


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    strategy:      str
    explanation:   str
    model:         str
    duration_s:    float
    success:       bool
    error:         str = ""
    fallback_used: str = ""


# ─── Generator ────────────────────────────────────────────────────────────────

class ExplanationGenerator:
    """Generates explanations via Gemma 4 with a two-tier fallback."""

    def __init__(self,
                 model:       str   = DEFAULT_MODEL,
                 base_url:    str   = OLLAMA_URL,
                 temperature: float = TEMPERATURE,
                 max_tokens:  int   = MAX_TOKENS):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return any(self.model in m["name"]
                       for m in r.json().get("models", []))
        except Exception:
            return False

    def _chat(self, user: str) -> str:
        """Tier 1: /api/chat with fixed system prompt."""
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()

    def _generate(self, user: str) -> str:
        """Tier 2: /api/generate — combined prompt."""
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  self.model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user}",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def warmup(self) -> None:
        print("Warming up Gemma 4...", end=" ", flush=True)
        try:
            self._chat("In one word, say 'ready'.")
            print("ready.")
        except Exception as e:
            print(f"warmup failed: {e}")

    def generate_full(self,
                      strategy:   str,
                      prediction: int,
                      confidence: float,
                      evidence=None) -> GenerationResult:

        if strategy not in STRATEGIES:
            return GenerationResult(
                strategy=strategy, explanation="", model=self.model,
                duration_s=0.0, success=False,
                error=f"Unknown strategy. Choose: {list(STRATEGIES.keys())}",
            )

        user = STRATEGIES[strategy](
            prediction=prediction,
            confidence=confidence,
            evidence=evidence,
        )
        t0            = time.time()
        fallback_used = ""
        text          = ""

        # Tier 1
        try:
            text = self._chat(user)
        except requests.exceptions.ConnectionError:
            return GenerationResult(
                strategy=strategy, explanation="", model=self.model,
                duration_s=time.time() - t0, success=False,
                error="Cannot connect to Ollama. Run: ollama serve",
            )
        except Exception:
            pass

        # Tier 2
        if not text:
            try:
                text = self._generate(user)
                if text:
                    fallback_used = "generate"
            except Exception as e:
                return GenerationResult(
                    strategy=strategy, explanation="", model=self.model,
                    duration_s=time.time() - t0, success=False,
                    error=f"Both tiers failed: {e}",
                )

        if not text:
            return GenerationResult(
                strategy=strategy, explanation="", model=self.model,
                duration_s=time.time() - t0, success=False,
                error="Both tiers returned empty response.",
            )

        return GenerationResult(
            strategy=strategy, explanation=text, model=self.model,
            duration_s=time.time() - t0, success=True,
            fallback_used=fallback_used,
        )

    def generate(self, strategy: str, prediction: int,
                 confidence: float, evidence=None) -> str:
        r = self.generate_full(strategy, prediction, confidence, evidence)
        if not r.success:
            raise RuntimeError(r.error)
        return r.explanation

    def generate_all_strategies(self, prediction: int, confidence: float,
                                  evidence=None, strategies: list = None) -> dict:
        targets = strategies or list(STRATEGIES.keys())
        results = {}
        for s in targets:
            print(f"  Generating {s}...", end=" ", flush=True)
            r = self.generate_full(s, prediction, confidence, evidence)
            results[s] = r
            if r.success:
                tag = f" via {r.fallback_used}" if r.fallback_used else ""
                print(f"done ({r.duration_s:.1f}s){tag}")
            else:
                print(f"FAILED: {r.error}")
        return results

    def print_status(self) -> None:
        print(f"\nOllama URL : {self.base_url}")
        print(f"Model      : {self.model}")
        status = "connected, model ready" if self._check_ollama() \
                 else "NOT CONNECTED — run: ollama serve"
        print(f"Status     : {status}\n")


# ─── Pipeline integration ─────────────────────────────────────────────────────

def run_full_sample(pipeline, gen, sample_id, das_data, fs,
                    ground_truth, model_prediction, model_confidence,
                    geo_data=None, geo_confidence=None,
                    strategies=None) -> list:
    targets = strategies or list(STRATEGIES.keys())

    first = pipeline.run(
        sample_id=sample_id, das_data=das_data, fs=fs,
        ground_truth=ground_truth, model_prediction=model_prediction,
        model_confidence=model_confidence, explanation="placeholder",
        strategy_name="_init", geo_data=geo_data, geo_confidence=geo_confidence,
    )
    evidence = first.symbolic_evidence

    print(f"\n[{sample_id}] Generating explanations...")
    gen_results = gen.generate_all_strategies(
        prediction=model_prediction, confidence=model_confidence,
        evidence=evidence, strategies=targets,
    )

    results = []
    for strategy, gr in gen_results.items():
        if not gr.success or not gr.explanation:
            print(f"  Skipping {strategy}: {gr.error or 'empty response'}")
            continue
        result = pipeline.run(
            sample_id=sample_id, das_data=das_data, fs=fs,
            ground_truth=ground_truth, model_prediction=model_prediction,
            model_confidence=model_confidence, explanation=gr.explanation,
            strategy_name=strategy, geo_data=geo_data, geo_confidence=geo_confidence,
        )
        results.append(result)
    return results


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = ExplanationGenerator()
    gen.print_status()

    if not gen._check_ollama():
        exit(1)

    gen.warmup()

    for test_name, has_ev in [("WITHOUT evidence", False), ("WITH mock evidence", True)]:
        print(f"\n{'═'*60}")
        print(f"  TEST: All strategies {test_name}")
        print(f"{'═'*60}")

        evidence = None
        if has_ev and SymbolicEvidence is not None:
            evidence = SymbolicEvidence(
                amplitude_level="high", amplitude_contrast="strong",
                snr_level="high", frequency_band="high",
                high_freq_content="strong", spectral_character="broadband",
                onset_character="sharp", signal_duration="short",
                rise_character="fast", das_coherence="coherent",
                spatial_continuity="continuous", moveout="detected",
                event_support="strong", geophone_event_support="strong",
                multimodal_agreement="agreement", das_geophone_conflict=False,
                signal_ambiguity="low", ambiguity_score=0,
            )

        all_ok = True
        for strategy in STRATEGIES:
            r = gen.generate_full(strategy=strategy, prediction=1,
                                   confidence=0.82, evidence=evidence)
            tag = f" via {r.fallback_used}" if r.fallback_used else ""
            if r.success and r.explanation:
                preview = r.explanation[:120].replace("\n", " ")
                print(f"  ✓ {strategy:<20} ({r.duration_s:.1f}s{tag}): {preview}...")
            else:
                print(f"  ✗ {strategy:<20} FAILED: {r.error}")
                all_ok = False

    print(f"\n{'═'*60}")
    if all_ok:
        print("All strategies working. Run: python pilot_30samples.py")
    else:
        print("Some strategies failed. Check Ollama is running correctly.")
