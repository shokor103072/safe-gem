"""
safe_gem/claim_extractor.py
-----------------------------
Deterministic, rule-based claim extraction from GenAI explanation text.
Uses a controlled vocabulary of phrase patterns — NO secondary LLM involved.

The method:
  1. Match phrases in the explanation against the controlled vocabulary.
  2. Each matched phrase maps to a (feature_group, polarity) tuple.
  3. The polarity is compared against the SymbolicEvidence label for that group.
  4. Result: supported | contradicted | unsupported (no signal evidence available)

This approach is:
  - Reproducible       (same input → same output, always)
  - Auditable          (vocabulary table can be inspected and extended)
  - Reviewer-safe      (no "LLM judging LLM" circularity)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from .symbolic_converter import SymbolicEvidence


VOCABULARY: Dict[str, Dict[str, List[str]]] = {

    "snr": {
        "positive": [
            "high snr", "high signal-to-noise", "strong signal-to-noise",
            "clear signal above noise", "good signal quality",
            "low noise contamination", "signal well above noise",
            "signal-to-noise ratio is high", "favorable signal-to-noise",
            "good signal-to-noise", "adequate snr", "sufficient snr",
            "strong signal relative to noise",
            "signal-to-noise ratio was high",
            "high signal-to-noise ratio",
            "exceptional signal-to-noise",
            "exceptionally high snr",
            "exceeding the noise floor",
            "above the noise floor",
            "well above background noise",
            "signal-to-noise ratio was exceptionally",
            "signal quality is good",
            "high data quality",
            "clear signal quality",
            "strong signal quality",
            "excellent snr",
            "signal is clear and distinct",
        ],
        "negative": [
            "low snr", "low signal-to-noise", "weak signal-to-noise",
            "noisy signal", "signal buried in noise", "poor snr",
            "noise-dominated", "high noise level", "noisy background", "poor signal-to-noise",
            "unfavorable signal-to-noise", "low signal-to-noise ratio",
            "insufficient snr", "elevated noise", "noise-contaminated",
            "noise floor", "buried in noise",
            "poor signal-to-noise ratio",
            "degraded signal-to-noise",
            "signal-to-noise ratio is low",
            "signal-to-noise ratio was poor",
            "unfavorable signal-to-noise ratio",
            "approaching the noise floor",
            "near the noise floor",
            "buried within the noise",
            "submerged in noise",
            "poor signal quality",
            "low data quality",
            "degraded signal quality",
            "signal quality is poor",
        ],
    },

    "amplitude": {
        "positive": [
            "high amplitude", "large amplitude", "strong amplitude",
            "clear amplitude burst", "high energy", "strong energy",
            "significant amplitude response", "strong signal",
            "high amplitude signal", "elevated amplitude", "clear amplitude",
            "prominent amplitude", "distinct amplitude",
            "high-amplitude waveform",
            "amplitude burst",
            "energy burst",
            "localized energy burst",
            "strong localized energy",
            "high-amplitude signal",
            "significant energy",
            "elevated energy levels",
            "strong waveform",
            "strong energy burst",
            "high energy signal",
            "significant energy burst",
            "strong signal energy",
        ],
        "negative": [
            "low amplitude", "weak amplitude", "small amplitude",
            "low energy", "weak energy", "minimal amplitude",
            "weak signal", "low amplitude signal", "poor amplitude",
            "weak amplitude response", "low energy signal",
            "insufficient amplitude", "subdued amplitude",
            "random fluctuations",
            "low amplitude fluctuations",
            "weak amplitude signal",
            "minimal energy",
        ],
    },

    "frequency": {
        "positive": [
            "high-frequency energy", "high frequency component",
            "broadband energy", "broadband signal", "high dominant frequency",
            "clear spectral energy", "rich spectral content",
            "spectral characteristics consistent with",
            "high-frequency content", "frequency content",
            "high frequency energy", "rich frequency content",
            "favorable spectral", "spectral energy",
            "energy distribution consistent", "frequency signature",
            "high-frequency signature",
            "unique frequency patterns",
            "specific frequency patterns",
            "frequency patterns",
            "spectral signature",
            "distinct frequency",
            "rich frequency",
        ],
        "negative": [
            "low-frequency noise", "low frequency dominated",
            "narrowband noise", "narrowband signal",
            "weak spectral energy", "low frequency content",
            "unfavorable spectral characteristics",
            "poor spectral characteristics",
            "spectral characteristics inconsistent",
            "low-frequency characteristics",
            "frequency characteristics of noise",
            "lacks high-frequency", "absence of high-frequency",
            "low spectral energy",
            "low-frequency vibrations",
            "low frequency vibrations",
            "low-frequency signal",
            "low frequency characteristics",
            "low frequency dominated signal",
            "dominated by low frequency",
        ],
    },

    "onset": {
        "positive": [
            "sharp onset", "impulsive onset", "abrupt onset",
            "sudden energy increase", "clear arrival onset",
            "impulsive arrival", "sharp first motion",
            "impulsive signal", "impulsive character",
            "clear impulsive", "distinct onset", "well-defined onset",
            "rapid onset", "abrupt increase",
            "p-wave arrival",
            "s-wave arrival",
            "p-wave and s-wave",
            "seismic arrivals",
            "wave arrivals",
            "clear arrivals",
            "p-wave onset",
            "first arrival",
            "phase arrival",
            "temporal occurrences",
            "temporal patterns",
            "distinct temporal",
            "transient signal",
            "transient event",
            "clear temporal",
            "impulsive waveform",
        ],
        "negative": [
            "gradual onset", "gradual arrival", "diffuse onset",
            "unclear onset", "slow onset", "emergent onset",
            "no clear onset", "lacks impulsive", "no impulsive",
            "lacks the sharp", "no distinct onset",
            "gradual increase", "no abrupt", "slow rise",
            "emergent signal",
            "random temporal",
            "irregular temporal",
            "no distinct temporal",
            "diffuse temporal",
        ],
    },

    "duration": {
        "positive": [
            "short-duration", "short duration", "compact signal",
            "brief impulse", "brief event", "short-lived",
        ],
        "negative": [
            "long-duration", "long duration", "extended signal",
            "prolonged disturbance", "extended noise", "long-lived",
        ],
    },

    "das_coherence": {
        "positive": [
            "coherent das response", "das channel coherence",
            "spatially coherent", "consistent das channels",
            "coherent across channels", "channel-consistent signal",
            "spatially consistent das", "coherent signal",
            "good coherence", "high coherence", "coherent across",
            "strong coherence", "spatially consistent",
            "coherent waveform",
            "coherent energy",
            "spatially coherent energy",
            "consistent waveform",
            "coherent across the array",
            "array coherence",
            "coherent waveforms",
            "consistent waveforms",
            "coherent signal pattern",
            "spatially coherent waveform",
            "coherent seismic waveform",
            "waveform consistency",
        ],
        "negative": [
            "incoherent das", "incoherent channel response",
            "fragmented das pattern", "inconsistent channels",
            "poor das coherence", "inconsistent das",
            "poor coherence", "poor signal coherence",
            "lack of coherence", "low coherence",
            "lacks coherence", "incoherent signal",
            "absence of coherence", "inconsistent signal",
            "incoherent waveform",
            "lack of spatial coherence",
            "no spatial coherence",
            "spatially incoherent",
            "random waveforms",
            "inconsistent waveforms",
            "waveform variability",
            "waveform incoherence",
        ],
    },

    "moveout": {
        "positive": [
            "visible moveout", "clear moveout pattern",
            "sloping arrival", "propagation pattern",
            "wave propagation", "consistent moveout",
        ],
        "negative": [
            "no moveout", "absent moveout", "unclear moveout",
            "no propagation pattern", "no sloping arrival",
        ],
    },

    "geophone_support": {
        "positive": [
            "geophone confirms", "geophone confirmation",
            "clear geophone arrival", "strong geophone support",
            "geophone detects", "visible on geophone",
            "geophone-confirmed", "independent confirmation",
            "sensor confirms", "corroborated by",
            "supported by sensor", "sensor data supports",
            "sensor evidence supports",
            "sensor support",
            "sensor data confirms",
            "supported by sensor data",
            "corroborated by sensor",
        ],
        "negative": [
            "weak geophone support", "no geophone confirmation",
            "geophone does not confirm", "not visible on geophone",
            "absent on geophone", "geophone shows nothing",
            "no independent confirmation", "sensor does not confirm",
            "not corroborated", "sensor data does not support",
            "no sensor confirmation",
        ],
    },

    "multimodal_agreement": {
        "positive": [
            "das and geophone agree", "das-geophone agreement",
            "both sensors support", "multimodal agreement",
            "consistent across sensors", "both das and geophone",
            "supported by both sensors",
        ],
        "negative": [
            "das-geophone disagreement", "conflicting sensor evidence",
            "only das supports", "only geophone supports",
            "sensor disagreement", "das and geophone conflict",
            "inconsistent sensor evidence",
        ],
    },

    "event_claim": {
        "positive": [
            "microseismic event", "seismic event", "event-like signal",
            "likely event", "probable microseismic", "fracture event",
            "induced seismicity", "seismic origin", "seismic source",
            "tectonic event", "earthquake signal", "genuine event",
            "real event",
            "p-wave",
            "s-wave",
            "natural source",
            "tectonic origin",
            "subsurface event",
            "reservoir event",
            "geothermal event",
            "seismic signature",
            "microseismic signature",
            "seismic activity",
            "microseismic activity",
            "seismic occurrence",
            "seismic source signature",
            "genuine seismic",
            "true seismic event",
        ],
        "negative": [
            "non-event", "noise signal", "artefact", "false alarm",
            "operational noise", "cultural noise", "surface noise",
            "background noise", "not a seismic event", "not microseismic",
            "likely noise", "classified as noise", "noise classification",
            "non-seismic", "instrumental noise", "anthropogenic noise",
            "does not meet the criteria",
            "does not represent an event",
            "not consistent with an event",
            "identified as noise",
            "noise or non-event",
            "no event detected",
            "not indicative of an event",
        ],
    },

    "uncertainty": {
        "positive": [
            "possibly", "likely", "may indicate", "might suggest",
            "appears to", "could be", "uncertain", "ambiguous",
            "limited evidence", "requires review", "cannot rule out",
            "tentatively", "we note that", "it is possible",
            "it appears", "it seems", "may be", "could indicate",
            "potential", "probable", "not certain",
            "further analysis", "further review", "warrants review",
            "suggests",
            "consistent with",
            "indicative of",
            "characteristic of",
            "typical of",
            "associated with",
            "we cannot rule out",
            "further investigation",
            "should be verified",
            "needs verification",
        ],
        "negative": [
            "definitely", "clearly", "certainly", "confirms",
            "unambiguously", "undoubtedly", "is confirmed",
            "is definitely", "is clearly", "is certainly",
            "without doubt", "it is clear", "it is evident",
            "it is definitive", "unquestionably",
            "there is no doubt", "it is obvious",
            "with high confidence",
            "with certainty",
            "strong evidence",
            "clear evidence",
            "definitive evidence",
            "conclusive",
            "unmistakable",
        ],
    },
}


EVIDENCE_MAP: Dict[str, Dict] = {
    "snr": {
        "field": "snr_level",
        "positive_labels": {"high"},
        "negative_labels": {"low"},
    },
    "amplitude": {
        "field": "amplitude_contrast",
        "positive_labels": {"strong"},
        "negative_labels": {"weak"},
    },
    "frequency": {
        "field": "high_freq_content",
        "positive_labels": {"strong", "moderate"},
        "negative_labels": {"weak"},
    },
    "onset": {
        "field": "onset_character",
        "positive_labels": {"sharp"},
        "negative_labels": {"gradual"},
    },
    "duration": {
        "field": "signal_duration",
        "positive_labels": {"short"},
        "negative_labels": {"long"},
    },
    "das_coherence": {
        "field": "das_coherence",
        "positive_labels": {"coherent"},
        "negative_labels": {"incoherent"},
    },
    "moveout": {
        "field": "moveout",
        "positive_labels": {"detected"},
        "negative_labels": {"absent"},
    },
    "geophone_support": {
        "field": "geophone_event_support",
        "positive_labels": {"strong", "moderate"},
        "negative_labels": {"weak"},
    },
    "multimodal_agreement": {
        "field": "multimodal_agreement",
        "positive_labels": {"agreement"},
        "negative_labels": {"disagreement"},
    },
    "event_claim": {
        "field": "event_support",
        "positive_labels": {"strong", "moderate"},
        "negative_labels": {"weak"},
    },
    "uncertainty": {
        "field": None,
        "positive_labels": set(),
        "negative_labels": set(),
    },
}


@dataclass
class ExtractedClaim:
    """One claim extracted from a GenAI explanation."""
    raw_phrase: str
    feature_group: str
    polarity: str


@dataclass
class ClaimCheckResult:
    """Result of checking one extracted claim against symbolic evidence."""
    claim: ExtractedClaim
    evidence_field: Optional[str]
    evidence_value: Optional[str]
    result: str


@dataclass
class GroundingReport:
    """Full grounding analysis for one explanation."""
    total_claims: int
    supported: int
    contradicted: int
    unavailable: int
    signal_grounding_score: float
    unsupported_claim_rate: float
    claim_details: List[ClaimCheckResult] = field(default_factory=list)

    @property
    def supported_claim_rate(self) -> float:
        return self.supported / max(self.total_claims, 1)


class ClaimExtractor:
    """
    Extracts signal-related claims from a GenAI explanation using the
    controlled vocabulary, then checks each claim against SymbolicEvidence.
    No LLM is used. All matching is deterministic string/regex matching.
    """

    def __init__(self, vocabulary: Dict = None):
        self.vocab = vocabulary or VOCABULARY
        self._patterns: Dict[str, Dict[str, List[re.Pattern]]] = {}
        for group, polarities in self.vocab.items():
            self._patterns[group] = {}
            for polarity, phrases in polarities.items():
                self._patterns[group][polarity] = [
                    re.compile(re.escape(p), re.IGNORECASE)
                    for p in phrases
                ]

    def _positive_match_is_negated(self, text: str, start: int) -> bool:
        prefix = text[max(0, start - 90):start]
        last_boundary = max(prefix.rfind(ch) for ch in [".", ";", ":", "!", "?", "\n"])
        if last_boundary != -1:
            prefix = prefix[last_boundary + 1:]
        negation_pattern = re.compile(
            r"(does\s+not|do\s+not|did\s+not|not|no|without|"
            r"lacks?|lack(?:ing)?|absence\s+of)"
            r"(?:\W+\w+){0,6}\W*$",
            re.IGNORECASE,
        )
        return bool(negation_pattern.search(prefix))

    def extract_claims(self, explanation: str) -> List[ExtractedClaim]:
        text = explanation.lower()
        candidates = []

        for group, polarities in self._patterns.items():
            for polarity, patterns in polarities.items():
                for pat in patterns:
                    for m in pat.finditer(text):
                        start, end = m.start(), m.end()
                        final_polarity = polarity
                        if (
                            polarity == "positive"
                            and group not in {"uncertainty"}
                            and self._positive_match_is_negated(text, start)
                        ):
                            final_polarity = "negative"

                        candidates.append((
                            start, end, end - start,
                            ExtractedClaim(
                                raw_phrase    = m.group(),
                                feature_group = group,
                                polarity      = final_polarity,
                            )
                        ))

        candidates.sort(key=lambda x: (x[2], -x[0]), reverse=True)
        selected = []
        selected_spans: List[Tuple[int, int]] = []
        for start, end, _length, claim in candidates:
            overlaps = any(not (end <= s or start >= e) for s, e in selected_spans)
            if overlaps:
                continue
            selected.append((start, claim))
            selected_spans.append((start, end))

        selected.sort(key=lambda x: x[0])
        return [claim for _, claim in selected]

    def check_claim(self,
                    claim: ExtractedClaim,
                    evidence: SymbolicEvidence) -> ClaimCheckResult:
        mapping = EVIDENCE_MAP.get(claim.feature_group, {})
        field_name = mapping.get("field")

        if field_name is None:
            return ClaimCheckResult(
                claim          = claim,
                evidence_field = None,
                evidence_value = None,
                result         = "unavailable",
            )

        evidence_value = getattr(evidence, field_name, None)

        if evidence_value in ("unavailable", None):
            return ClaimCheckResult(
                claim          = claim,
                evidence_field = field_name,
                evidence_value = evidence_value,
                result         = "unavailable",
            )

        if claim.polarity == "positive":
            compatible = mapping.get("positive_labels", set())
        else:
            compatible = mapping.get("negative_labels", set())

        result = "supported" if evidence_value in compatible else "contradicted"

        return ClaimCheckResult(
            claim          = claim,
            evidence_field = field_name,
            evidence_value = evidence_value,
            result         = result,
        )

    def analyse(self,
                explanation: str,
                evidence: SymbolicEvidence) -> GroundingReport:
        claims  = self.extract_claims(explanation)
        results = [self.check_claim(c, evidence) for c in claims]

        supported    = sum(1 for r in results if r.result == "supported")
        contradicted = sum(1 for r in results if r.result == "contradicted")
        unavailable  = sum(1 for r in results if r.result == "unavailable")
        total        = len(results)

        checkable = supported + contradicted
        sgs  = supported / checkable if checkable > 0 else 1.0
        ucr  = contradicted / total  if total > 0 else 0.0

        return GroundingReport(
            total_claims           = total,
            supported              = supported,
            contradicted           = contradicted,
            unavailable            = unavailable,
            signal_grounding_score = sgs,
            unsupported_claim_rate = ucr,
            claim_details          = results,
        )

    def print_report(self, report: GroundingReport) -> None:
        print(f"\n{'─'*60}")
        print(f"Grounding report: {report.total_claims} claims found")
        print(f"  Supported   : {report.supported}")
        print(f"  Contradicted: {report.contradicted}")
        print(f"  Unavailable : {report.unavailable}")
        print(f"  SGS (Signal Grounding Score) : {report.signal_grounding_score:.3f}")
        print(f"  UCR (Unsupported Claim Rate) : {report.unsupported_claim_rate:.3f}")
        print(f"{'─'*60}")
        for r in report.claim_details:
            icon = {"supported": "✓", "contradicted": "✗", "unavailable": "?"}.get(r.result, "?")
            print(f"  {icon} [{r.claim.feature_group}/{r.claim.polarity}] "
                  f'"{r.claim.raw_phrase}"')
            if r.evidence_field:
                print(f"      Evidence: {r.evidence_field} = {r.evidence_value} → {r.result}")
        print()
