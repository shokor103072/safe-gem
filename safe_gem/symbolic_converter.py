"""
safe_gem/symbolic_converter.py
-------------------------------
Converts raw SignalFeatures into human-readable symbolic evidence labels.
These labels are the ground truth that GenAI explanation claims are checked against.

All thresholds are configurable via ThresholdConfig.
Defaults are set for EGS microseismic monitoring (Utah FORGE range).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from .feature_extraction import SignalFeatures, MultimodalFeatures


# ─── Threshold configuration ─────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """
    All threshold values in one place.
    Override for your specific dataset.
    Defaults are tuned for Utah FORGE DAS/geophone data.
    """
    # SNR (dB)
    snr_low: float   = 2.0
    snr_high: float  = 8.0

    # Amplitude ratio (event / noise)
    amp_ratio_low:  float = 1.5
    amp_ratio_high: float = 4.0

    # Dominant frequency (Hz)
    freq_low:  float = 20.0
    freq_high: float = 60.0

    # High-frequency energy ratio
    hf_ratio_low:  float = 0.20
    hf_ratio_high: float = 0.45

    # Onset sharpness (STA/LTA peak)
    onset_low:  float = 3.0
    onset_high: float = 8.0

    # Duration (seconds)
    dur_short: float = 0.3
    dur_long:  float = 1.5

    # Envelope rise time (seconds)
    rise_fast: float = 0.05
    rise_slow: float = 0.20

    # DAS channel coherence
    coherence_low:  float = 0.40
    coherence_high: float = 0.70

    # Spatial continuity (fraction of active channels)
    spatial_low:  float = 0.30
    spatial_high: float = 0.65

    # Event probability from classifier
    prob_weak:    float = 0.40
    prob_strong:  float = 0.70

    # Signal ambiguity composite (see compute_signal_ambiguity)
    ambiguity_low:  int = 1   # sum of ambiguity flags
    ambiguity_high: int = 3


DEFAULT_THRESHOLDS = ThresholdConfig()


# ─── Symbolic evidence dictionary ────────────────────────────────────────────

@dataclass
class SymbolicEvidence:
    """
    Complete set of symbolic labels for one signal window.
    Each field is a string label from a fixed, closed vocabulary.
    These are matched against GenAI explanation claims.
    """
    # Amplitude
    amplitude_level: str             # "low" | "moderate" | "high"
    amplitude_contrast: str          # "weak" | "moderate" | "strong" contrast vs noise

    # SNR
    snr_level: str                   # "low" | "moderate" | "high" SNR

    # Frequency
    frequency_band: str              # "low" | "mid" | "high" dominant frequency
    high_freq_content: str           # "weak" | "moderate" | "strong" high-frequency energy
    spectral_character: str          # "narrowband" | "broadband"

    # Temporal
    onset_character: str             # "gradual" | "moderate" | "sharp" onset
    signal_duration: str             # "short" | "medium" | "long" duration
    rise_character: str              # "slow" | "moderate" | "fast" rise

    # DAS spatial (may be "unavailable" if single-channel)
    das_coherence: str               # "incoherent" | "partially coherent" | "coherent"
    spatial_continuity: str          # "fragmented" | "partial" | "continuous"
    moveout: str                     # "absent" | "detected" | "unavailable"

    # Event probability
    event_support: str               # DAS/model P(event): "weak" | "moderate" | "strong" support
    geophone_event_support: str      # geophone P(event): "weak" | "moderate" | "strong" | "unavailable"

    # Multimodal
    multimodal_agreement: str        # "agreement" | "partial" | "disagreement"
    das_geophone_conflict: bool      # True if sensors clearly disagree

    # Composite
    signal_ambiguity: str            # "low" | "moderate" | "high" ambiguity
    ambiguity_score: int             # 0–6 raw ambiguity flag count


# ─── Converter ───────────────────────────────────────────────────────────────

class SymbolicConverter:
    """
    Converts SignalFeatures + MultimodalFeatures → SymbolicEvidence.

    Usage
    -----
    converter = SymbolicConverter()                    # default thresholds
    converter = SymbolicConverter(my_thresholds)       # custom thresholds
    evidence  = converter.convert(das_feat, mm_feat)
    """

    def __init__(self, thresholds: ThresholdConfig = None):
        self.t = thresholds or DEFAULT_THRESHOLDS

    # ── Private helpers ─────────────────────────────────────────────────────

    def _three_level(self, value: float,
                     low_thresh: float, high_thresh: float,
                     labels=("low", "moderate", "high")) -> str:
        if value < low_thresh:
            return labels[0]
        elif value <= high_thresh:
            return labels[1]
        return labels[2]

    # ── Public method ───────────────────────────────────────────────────────

    def convert(self,
                feat: SignalFeatures,
                mm: Optional[MultimodalFeatures] = None) -> SymbolicEvidence:
        """
        Convert a SignalFeatures object (plus optional multimodal features)
        to SymbolicEvidence labels.
        """
        t = self.t

        # ── Amplitude
        amp_level    = self._three_level(feat.peak_amplitude,
                                          feat.pre_event_noise_rms * 3,
                                          feat.pre_event_noise_rms * 10)
        amp_contrast = self._three_level(feat.amplitude_ratio,
                                          t.amp_ratio_low, t.amp_ratio_high,
                                          ("weak", "moderate", "strong"))

        # ── SNR
        snr_level = self._three_level(feat.snr_db,
                                       t.snr_low, t.snr_high)

        # ── Frequency
        freq_band = self._three_level(feat.dominant_freq_hz,
                                       t.freq_low, t.freq_high,
                                       ("low", "mid", "high"))
        hf_content = self._three_level(feat.high_freq_energy_ratio,
                                        t.hf_ratio_low, t.hf_ratio_high,
                                        ("weak", "moderate", "strong"))
        spectral = ("broadband" if feat.spectral_bandwidth_hz > 30.0
                    else "narrowband")

        # ── Temporal
        onset = self._three_level(feat.onset_sharpness,
                                   t.onset_low, t.onset_high,
                                   ("gradual", "moderate", "sharp"))
        duration = self._three_level(feat.duration_s,
                                      t.dur_short, t.dur_long,
                                      ("short", "medium", "long"))
        rise = self._three_level(feat.envelope_rise_time_s,
                                  t.rise_fast, t.rise_slow,
                                  ("fast", "moderate", "slow"))
        # Invert: fast rise = sharp onset, so remap
        rise_char = {"fast": "fast", "moderate": "moderate", "slow": "slow"}[rise]

        # ── DAS spatial
        if feat.channel_coherence is not None:
            das_coh = self._three_level(feat.channel_coherence,
                                         t.coherence_low, t.coherence_high,
                                         ("incoherent", "partially coherent", "coherent"))
        else:
            das_coh = "unavailable"

        if feat.spatial_continuity is not None:
            sp_cont = self._three_level(feat.spatial_continuity,
                                         t.spatial_low, t.spatial_high,
                                         ("fragmented", "partial", "continuous"))
        else:
            sp_cont = "unavailable"

        if feat.moveout_detected is not None:
            moveout = "detected" if feat.moveout_detected else "absent"
        else:
            moveout = "unavailable"

        # ── Event support
        ev_support = self._three_level(feat.event_probability,
                                        t.prob_weak, t.prob_strong,
                                        ("weak", "moderate", "strong"))

        # ── Multimodal
        if mm is not None:
            mm_agree   = mm.agreement_level
            mm_conflict = mm.conflict_detected
            geo_support = self._three_level(mm.geophone_event_probability,
                                            t.prob_weak, t.prob_strong,
                                            ("weak", "moderate", "strong"))
        else:
            mm_agree   = "unavailable"
            mm_conflict = False
            geo_support = "unavailable"

        # ── Signal ambiguity composite
        # Count ambiguity flags (low SNR, low coherence, borderline probability,
        # weak amplitude, gradual onset, multimodal conflict)
        flags = [
            feat.snr_db < t.snr_low,
            feat.channel_coherence is not None and feat.channel_coherence < t.coherence_low,
            t.prob_weak < feat.event_probability < t.prob_strong,   # borderline
            feat.amplitude_ratio < t.amp_ratio_low,
            feat.onset_sharpness < t.onset_low,
            mm is not None and mm.conflict_detected,
        ]
        ambiguity_count = sum(flags)
        ambiguity_str   = self._three_level(ambiguity_count,
                                             t.ambiguity_low, t.ambiguity_high,
                                             ("low", "moderate", "high"))

        return SymbolicEvidence(
            amplitude_level     = amp_level,
            amplitude_contrast  = amp_contrast,
            snr_level           = snr_level,
            frequency_band      = freq_band,
            high_freq_content   = hf_content,
            spectral_character  = spectral,
            onset_character     = onset,
            signal_duration     = duration,
            rise_character      = rise_char,
            das_coherence       = das_coh,
            spatial_continuity  = sp_cont,
            moveout             = moveout,
            event_support       = ev_support,
            geophone_event_support = geo_support,
            multimodal_agreement = mm_agree,
            das_geophone_conflict = mm_conflict,
            signal_ambiguity    = ambiguity_str,
            ambiguity_score     = ambiguity_count,
        )

    def to_dict(self, evidence: SymbolicEvidence) -> Dict[str, str]:
        """Return symbolic evidence as a plain dictionary for logging."""
        return {k: str(v) for k, v in evidence.__dict__.items()}
