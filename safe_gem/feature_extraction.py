"""
safe_gem/feature_extraction.py
-------------------------------
Numerical feature extraction from DAS and geophone signals.
All features are computed deterministically from raw signal arrays.
No LLM involvement at this stage.

Inputs : numpy arrays (n_channels x n_samples for DAS, n_samples for geophone)
Outputs: SignalFeatures dataclass with all scalar feature values
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─── Data containers ────────────────────────────────────────────────────────

@dataclass
class SignalFeatures:
    """
    All numerical features extracted from one DAS or geophone signal window.
    Used downstream by SymbolicConverter to produce evidence labels.
    """
    # ── Amplitude
    peak_amplitude: float          # max absolute value in event window
    rms_amplitude: float           # RMS over event window
    amplitude_ratio: float         # event-window RMS / noise-window RMS

    # ── Noise / SNR
    snr_db: float                  # 20*log10(event_rms / noise_rms)
    pre_event_noise_rms: float     # RMS of pre-event window

    # ── Frequency
    dominant_freq_hz: float        # frequency of peak spectral power
    spectral_bandwidth_hz: float   # RMS spectral bandwidth
    high_freq_energy_ratio: float  # energy above HF_THRESH / total energy

    # ── Temporal
    onset_sharpness: float         # peak STA/LTA ratio at arrival
    duration_s: float              # estimated event duration in seconds
    envelope_rise_time_s: float    # time from 10% to 90% of peak envelope

    # ── DAS-specific (None when processing a single geophone trace)
    channel_coherence: Optional[float] = None   # mean adjacent-channel corr
    spatial_continuity: Optional[float] = None  # fraction of active channels
    moveout_detected: Optional[bool] = None     # linear moveout across channels

    # ── Classifier output (from your URDI or other model)
    event_probability: float = 0.5


@dataclass
class MultimodalFeatures:
    """
    Cross-sensor features derived from DAS + geophone SignalFeatures together.
    Used for Multimodal Inconsistency (MI) component of EIRRS.
    """
    das_event_probability: float
    geophone_event_probability: float
    probability_delta: float           # |P_DAS - P_geo|
    agreement_level: str               # "agreement" / "partial" / "disagreement"
    conflict_detected: bool            # True if delta > CONFLICT_THRESHOLD


# ─── Constants (adjust per dataset) ─────────────────────────────────────────

HF_THRESH_HZ     = 40.0    # high-frequency boundary for energy ratio
STA_S            = 0.01    # STA window length (seconds)
LTA_S            = 0.10    # LTA window length (seconds)
DURATION_THRESH  = 0.10    # envelope fraction threshold for duration estimate
CONFLICT_THRESH  = 0.30    # |P_DAS - P_geo| threshold for conflict detection
COHERENCE_PAIRS  = 20      # max adjacent-channel pairs for coherence computation


# ─── Core signal processing functions ────────────────────────────────────────

def _compute_snr(trace: np.ndarray,
                 pre_event_samples: int) -> Tuple[float, float]:
    """
    SNR in dB using a pre-event noise window.
    Returns (snr_db, noise_rms).
    """
    noise = trace[:pre_event_samples]
    event = trace[pre_event_samples:]

    noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-10
    event_rms = float(np.sqrt(np.mean(event ** 2)))
    snr_db    = 20.0 * np.log10(event_rms / noise_rms)
    return snr_db, noise_rms


def _compute_frequency_features(trace: np.ndarray,
                                 fs: float,
                                 fmin: float = 1.0,
                                 fmax: float = 300.0) -> Tuple[float, float, float]:
    """
    Returns (dominant_freq_hz, spectral_bandwidth_hz, high_freq_energy_ratio).
    """
    window  = np.hanning(len(trace))
    freqs   = np.fft.rfftfreq(len(trace), d=1.0 / fs)
    power   = np.abs(np.fft.rfft(trace * window)) ** 2

    # Restrict to valid band
    mask     = (freqs >= fmin) & (freqs <= fmax)
    f_valid  = freqs[mask]
    p_valid  = power[mask]

    if p_valid.sum() < 1e-30:
        return 0.0, 0.0, 0.0

    # Dominant frequency
    dom_freq = float(f_valid[np.argmax(p_valid)])

    # RMS spectral bandwidth
    p_norm    = p_valid / p_valid.sum()
    mean_f    = float(np.sum(f_valid * p_norm))
    bandwidth = float(np.sqrt(np.sum((f_valid - mean_f) ** 2 * p_norm)))

    # High-frequency energy ratio (relative to full spectrum)
    total_power = power.sum() + 1e-30
    hf_power    = power[freqs >= HF_THRESH_HZ].sum()
    hf_ratio    = float(hf_power / total_power)

    return dom_freq, bandwidth, hf_ratio


def _compute_onset_sharpness(trace: np.ndarray, fs: float) -> float:
    """
    Peak STA/LTA ratio — proxy for onset impulsiveness.
    Higher value → sharper / more event-like onset.
    """
    sta_n = max(1, int(STA_S * fs))
    lta_n = max(sta_n + 1, int(LTA_S * fs))

    cf  = trace ** 2
    sta = np.convolve(cf, np.ones(sta_n) / sta_n, mode='same')
    lta = np.convolve(cf, np.ones(lta_n) / lta_n, mode='same')

    ratio = sta / (lta + 1e-10)
    return float(np.max(ratio))


def _compute_envelope_rise_time(trace: np.ndarray, fs: float) -> float:
    """
    Time (seconds) for signal envelope to rise from 10% to 90% of its peak.
    Faster rise → more impulsive onset.
    """
    envelope = np.abs(scipy_signal.hilbert(trace))
    peak_val = np.max(envelope)
    peak_idx = int(np.argmax(envelope))

    pre = envelope[:peak_idx]
    if len(pre) == 0:
        return 0.0

    t10_idx = np.where(pre >= 0.10 * peak_val)[0]
    t90_idx = np.where(pre >= 0.90 * peak_val)[0]

    t10 = float(t10_idx[0])  / fs if len(t10_idx) else 0.0
    t90 = float(t90_idx[0])  / fs if len(t90_idx) else float(peak_idx) / fs

    return max(0.0, t90 - t10)


def _compute_duration(trace: np.ndarray, fs: float) -> float:
    """
    Estimated event duration: time above DURATION_THRESH × peak envelope.
    """
    envelope = np.abs(scipy_signal.hilbert(trace))
    peak_val = np.max(envelope)
    above    = np.where(envelope >= DURATION_THRESH * peak_val)[0]

    if len(above) < 2:
        return len(trace) / fs
    return float(above[-1] - above[0]) / fs


def _compute_channel_coherence(das_mc: np.ndarray) -> float:
    """
    Mean absolute Pearson correlation between adjacent DAS channels.
    das_mc: (n_channels, n_samples)
    """
    n_ch   = das_mc.shape[0]
    pairs  = min(n_ch - 1, COHERENCE_PAIRS)
    corrs  = []

    for i in range(pairs):
        c1, c2 = das_mc[i], das_mc[i + 1]
        s1, s2 = np.std(c1), np.std(c2)
        if s1 < 1e-10 or s2 < 1e-10:
            continue
        r = float(np.corrcoef(c1, c2)[0, 1])
        corrs.append(abs(r))

    return float(np.mean(corrs)) if corrs else 0.5


def _compute_spatial_continuity(das_mc: np.ndarray,
                                 fs: float,
                                 pre_event_s: float,
                                 snr_thresh_db: float = 3.0) -> float:
    """
    Fraction of DAS channels with SNR ≥ snr_thresh_db.
    """
    n_ch = das_mc.shape[0]
    pre  = int(pre_event_s * fs)
    active = sum(
        1 for ch in das_mc
        if _compute_snr(ch, pre)[0] >= snr_thresh_db
    )
    return active / n_ch


def _detect_moveout(das_mc: np.ndarray, fs: float,
                    min_r2: float = 0.70) -> bool:
    """
    Detect linear moveout across DAS channels using envelope peak times.
    Returns True if R² of linear fit to peak times exceeds min_r2.
    """
    n_ch = das_mc.shape[0]
    if n_ch < 5:
        return False

    peak_times = np.array([
        np.argmax(np.abs(scipy_signal.hilbert(ch))) / fs
        for ch in das_mc
    ])
    channels = np.arange(n_ch, dtype=float)

    # Linear regression
    A   = np.vstack([channels, np.ones(n_ch)]).T
    res = np.linalg.lstsq(A, peak_times, rcond=None)
    fit = A @ res[0]
    ss_res = np.sum((peak_times - fit) ** 2)
    ss_tot = np.sum((peak_times - peak_times.mean()) ** 2) + 1e-10
    r2 = 1.0 - ss_res / ss_tot

    return bool(r2 >= min_r2)


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_das_features(das_data: np.ndarray,
                          fs: float,
                          pre_event_s: float = 0.5,
                          event_probability: float = 0.5) -> SignalFeatures:
    """
    Extract all features from a DAS signal window.

    Parameters
    ----------
    das_data        : np.ndarray, shape (n_channels, n_samples) or (n_samples,)
    fs              : float, sampling rate in Hz
    pre_event_s     : float, length of pre-event noise window in seconds
    event_probability : float, AI classifier output probability [0, 1]

    Returns
    -------
    SignalFeatures
    """
    pre_samples = int(pre_event_s * fs)

    if das_data.ndim == 2:
        trace              = np.mean(das_data, axis=0)
        channel_coherence  = _compute_channel_coherence(das_data)
        spatial_continuity = _compute_spatial_continuity(das_data, fs, pre_event_s)
        moveout_detected   = _detect_moveout(das_data, fs)
    else:
        trace              = das_data
        channel_coherence  = None
        spatial_continuity = None
        moveout_detected   = None

    snr_db, noise_rms       = _compute_snr(trace, pre_samples)
    dom_f, bandwidth, hf_r  = _compute_frequency_features(trace, fs)
    onset_sharp             = _compute_onset_sharpness(trace, fs)
    duration                = _compute_duration(trace, fs)
    rise_time               = _compute_envelope_rise_time(trace, fs)

    peak_amp  = float(np.max(np.abs(trace)))
    rms_amp   = float(np.sqrt(np.mean(trace ** 2)))
    noise_rms_window = float(np.sqrt(np.mean(trace[:pre_samples] ** 2))) + 1e-10
    amp_ratio = rms_amp / noise_rms_window

    return SignalFeatures(
        peak_amplitude      = peak_amp,
        rms_amplitude       = rms_amp,
        amplitude_ratio     = amp_ratio,
        snr_db              = snr_db,
        pre_event_noise_rms = noise_rms,
        dominant_freq_hz    = dom_f,
        spectral_bandwidth_hz = bandwidth,
        high_freq_energy_ratio = hf_r,
        onset_sharpness     = onset_sharp,
        duration_s          = duration,
        envelope_rise_time_s = rise_time,
        channel_coherence   = channel_coherence,
        spatial_continuity  = spatial_continuity,
        moveout_detected    = moveout_detected,
        event_probability   = event_probability,
    )


def extract_geophone_features(geo_data: np.ndarray,
                               fs: float,
                               pre_event_s: float = 0.5,
                               event_probability: float = 0.5) -> SignalFeatures:
    """
    Extract features from a single geophone trace.
    Wrapper around extract_das_features with DAS spatial features disabled.
    """
    return extract_das_features(
        geo_data.ravel(), fs, pre_event_s, event_probability
    )


def compute_multimodal_features(das_feat: SignalFeatures,
                                 geo_feat: SignalFeatures) -> MultimodalFeatures:
    """
    Compute cross-sensor (DAS vs geophone) agreement features.
    """
    p_das   = das_feat.event_probability
    p_geo   = geo_feat.event_probability
    delta   = abs(p_das - p_geo)

    if delta <= 0.15:
        agreement = "agreement"
    elif delta <= CONFLICT_THRESH:
        agreement = "partial"
    else:
        agreement = "disagreement"

    return MultimodalFeatures(
        das_event_probability     = p_das,
        geophone_event_probability = p_geo,
        probability_delta         = delta,
        agreement_level           = agreement,
        conflict_detected         = delta > CONFLICT_THRESH,
    )
