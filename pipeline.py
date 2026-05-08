"""
safe_gem/pipeline.py
---------------------
Main SAFE-GEM pipeline orchestrator.

End-to-end flow for one sample:
    Raw signals → Features → Symbolic evidence → Claim extraction
    → Grounding check → EIRRS score → Risk decision

Also provides SafeGEMBatch for running the full dataset and
producing the summary table needed for the paper's Results section.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .feature_extraction import (
    SignalFeatures, MultimodalFeatures,
    extract_das_features, extract_geophone_features, compute_multimodal_features,
)
from .symbolic_converter import SymbolicConverter, SymbolicEvidence, ThresholdConfig
from .claim_extractor    import ClaimExtractor, GroundingReport
from .eirrs_scorer       import EIRRSScorer, EIRRSResult


# ─── Per-sample result ───────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Complete SAFE-GEM output for one signal sample."""
    sample_id:           str
    ground_truth:        int           # 1 = event, 0 = non-event
    model_prediction:    int
    model_confidence:    float
    model_correct:       bool

    explanation_strategy: str          # e.g. "P1_raw", "P2_grounded", ...
    explanation_text:    str

    das_features:        SignalFeatures
    geo_features:        Optional[SignalFeatures]
    mm_features:         Optional[MultimodalFeatures]
    symbolic_evidence:   SymbolicEvidence
    grounding_report:    GroundingReport
    eirrs_result:        EIRRSResult


@dataclass
class BatchSummary:
    """Aggregate metrics for one (dataset × strategy) combination."""
    strategy:             str
    n_samples:            int

    # Accuracy
    model_accuracy:       float

    # Grounding
    mean_sgs:             float        # Signal Grounding Score
    mean_ucr:             float        # Unsupported Claim Rate

    # EIRRS
    mean_eirrs_equal:     float
    mean_eirrs_error:     float
    mean_eirrs_hf:        float

    # Risk tier distribution
    pct_low_risk:         float
    pct_medium_risk:      float
    pct_high_risk:        float

    # Breakdown by signal ambiguity
    mean_eirrs_low_ambiguity:  float
    mean_eirrs_high_ambiguity: float


# ─── SAFE-GEM pipeline ───────────────────────────────────────────────────────

class SafeGEMPipeline:
    """
    Single-sample SAFE-GEM pipeline.

    Usage
    -----
    pipeline = SafeGEMPipeline()
    result   = pipeline.run(
        sample_id         = "FORGE_001",
        das_data          = das_array,          # (n_channels, n_samples) or (n_samples,)
        geo_data          = geo_array,          # (n_samples,) or None
        fs                = 500.0,
        ground_truth      = 1,
        model_prediction  = 1,
        model_confidence  = 0.87,
        explanation       = "...",
        strategy_name     = "P1_raw",
    )
    pipeline.print_result(result)
    """

    def __init__(self,
                 thresholds: ThresholdConfig = None,
                 pre_event_s: float = 0.5):
        self.converter  = SymbolicConverter(thresholds)
        self.extractor  = ClaimExtractor()
        self.scorer     = EIRRSScorer()
        self.pre_event_s = pre_event_s

    def run(self,
            sample_id:         str,
            das_data:          np.ndarray,
            fs:                float,
            ground_truth:      int,
            model_prediction:  int,
            model_confidence:  float,
            explanation:       str,
            strategy_name:     str,
            geo_data:          Optional[np.ndarray] = None,
            geo_confidence:    Optional[float] = None) -> SampleResult:
        """Run the full pipeline for one sample."""

        # ── Feature extraction
        # IMPORTANT: event_probability must mean P(event), not generic confidence.
        # If the model predicts non-event with confidence 0.83, P(event)=0.17.
        model_event_probability = (
            float(model_confidence) if int(model_prediction) == 1
            else 1.0 - float(model_confidence)
        )

        das_feat = extract_das_features(
            das_data, fs, self.pre_event_s, model_event_probability
        )

        geo_feat = None
        mm_feat  = None
        if geo_data is not None:
            # geo_confidence is treated as P(event) for the geophone branch.
            # If absent, fall back to the fused/model P(event).
            geo_prob = geo_confidence if geo_confidence is not None else model_event_probability
            geo_feat = extract_geophone_features(geo_data, fs, self.pre_event_s, float(geo_prob))
            mm_feat  = compute_multimodal_features(das_feat, geo_feat)

        # ── Symbolic evidence
        evidence = self.converter.convert(das_feat, mm_feat)

        # ── Claim grounding
        grounding = self.extractor.analyse(explanation, evidence)

        # ── EIRRS
        model_correct = (model_prediction == ground_truth)
        eirrs_result  = self.scorer.score(
            explanation      = explanation,
            evidence         = evidence,
            grounding        = grounding,
            model_correct    = model_correct,
            model_confidence = model_confidence,
        )

        return SampleResult(
            sample_id          = sample_id,
            ground_truth       = ground_truth,
            model_prediction   = model_prediction,
            model_confidence   = model_confidence,
            model_correct      = model_correct,
            explanation_strategy = strategy_name,
            explanation_text   = explanation,
            das_features       = das_feat,
            geo_features       = geo_feat,
            mm_features        = mm_feat,
            symbolic_evidence  = evidence,
            grounding_report   = grounding,
            eirrs_result       = eirrs_result,
        )

    def print_result(self, r: SampleResult) -> None:
        """Print a human-readable summary of one SampleResult."""
        print(f"\n{'━'*60}")
        print(f"  Sample: {r.sample_id}  |  Strategy: {r.explanation_strategy}")
        print(f"  Ground truth: {'EVENT' if r.ground_truth else 'NOISE'}  "
              f"|  Prediction: {'EVENT' if r.model_prediction else 'NOISE'}  "
              f"({'CORRECT' if r.model_correct else 'WRONG'})  "
              f"conf={r.model_confidence:.2f}")
        print(f"\n  Explanation:\n    {r.explanation_text[:180]}...")
        print(f"\n  Signal ambiguity: {r.symbolic_evidence.signal_ambiguity}")
        print(f"  Multimodal agreement: {r.symbolic_evidence.multimodal_agreement}")
        self.extractor.print_report(r.grounding_report)
        self.scorer.print_result(r.eirrs_result)


# ─── Batch runner ────────────────────────────────────────────────────────────

class SafeGEMBatch:
    """
    Runs the full pipeline over a list of samples and strategies,
    then produces the summary tables needed for the paper's Results section.
    """

    def __init__(self, pipeline: SafeGEMPipeline = None):
        self.pipeline = pipeline or SafeGEMPipeline()
        self.results: List[SampleResult] = []

    def add_result(self, result: SampleResult) -> None:
        self.results.append(result)

    def summarise_by_strategy(self) -> Dict[str, BatchSummary]:
        """Aggregate metrics per explanation strategy."""
        from collections import defaultdict
        groups: Dict[str, List[SampleResult]] = defaultdict(list)
        for r in self.results:
            groups[r.explanation_strategy].append(r)

        summaries = {}
        for strategy, items in groups.items():
            summaries[strategy] = self._summarise(strategy, items)
        return summaries

    def _summarise(self, strategy: str, items: List[SampleResult]) -> BatchSummary:
        n = len(items)
        if n == 0:
            raise ValueError("Empty group")

        acc   = sum(r.model_correct for r in items) / n
        m_sgs = sum(r.grounding_report.signal_grounding_score for r in items) / n
        m_ucr = sum(r.grounding_report.unsupported_claim_rate for r in items) / n

        m_eq  = sum(r.eirrs_result.eirrs_equal for r in items) / n
        m_eh  = sum(r.eirrs_result.eirrs_error_heavy for r in items) / n
        m_hf  = sum(r.eirrs_result.eirrs_hf_heavy for r in items) / n

        tiers = [r.eirrs_result.risk_tier for r in items]
        p_low  = tiers.count("low")    / n * 100
        p_med  = tiers.count("medium") / n * 100
        p_high = tiers.count("high")   / n * 100

        low_amb  = [r for r in items if r.symbolic_evidence.signal_ambiguity == "low"]
        high_amb = [r for r in items if r.symbolic_evidence.signal_ambiguity == "high"]

        m_eirrs_la = (sum(r.eirrs_result.eirrs_equal for r in low_amb)  / len(low_amb)
                      if low_amb else float("nan"))
        m_eirrs_ha = (sum(r.eirrs_result.eirrs_equal for r in high_amb) / len(high_amb)
                      if high_amb else float("nan"))

        return BatchSummary(
            strategy             = strategy,
            n_samples            = n,
            model_accuracy       = acc,
            mean_sgs             = m_sgs,
            mean_ucr             = m_ucr,
            mean_eirrs_equal     = m_eq,
            mean_eirrs_error     = m_eh,
            mean_eirrs_hf        = m_hf,
            pct_low_risk         = p_low,
            pct_medium_risk      = p_med,
            pct_high_risk        = p_high,
            mean_eirrs_low_ambiguity  = m_eirrs_la,
            mean_eirrs_high_ambiguity = m_eirrs_ha,
        )

    def print_summary_table(self) -> None:
        """Print the main results table for the paper."""
        summaries = self.summarise_by_strategy()
        w = 22
        print(f"\n{'─'*100}")
        print(f"{'Strategy':<{w}} {'EIRRS (eq)':>10} {'EIRRS (err)':>11} "
              f"{'SGS':>8} {'UCR':>8} {'Low%':>7} {'High%':>7}")
        print(f"{'─'*100}")
        for strategy, s in sorted(summaries.items()):
            print(f"{strategy:<{w}} {s.mean_eirrs_equal:>10.3f} {s.mean_eirrs_error:>11.3f} "
                  f"{s.mean_sgs:>8.3f} {s.mean_ucr:>8.3f} "
                  f"{s.pct_low_risk:>7.1f} {s.pct_high_risk:>7.1f}")
        print(f"{'─'*100}\n")
