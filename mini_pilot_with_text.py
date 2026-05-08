"""
mini_pilot_with_text.py
------------------------
Runs 3 samples × 5 strategies = 15 calls.
Saves full explanation texts to explanations_full.txt
so you can read what Gemma 4 actually writes and calibrate vocabulary.

Run from your safe_gem folder:
    python mini_pilot_with_text.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Dataset"
RESULTS_DIR = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Results"
OUTPUT_TXT  = "explanations_full.txt"

SAMPLING_RATE = 500.0
PRE_EVENT_S   = 0.5
STRATEGIES    = ["P1_raw", "P2_grounded", "P3_uncertainty",
                 "P4_constrained", "P5_safe_gem"]

# Use 3 samples: 1 clear event, 1 clear noise, 1 ambiguous
N_SAMPLES = 3


def main():
    from safe_gem import SafeGEMPipeline, SafeGEMBatch
    from generate_explanation import ExplanationGenerator, run_full_sample

    # Load data
    X    = np.load(os.path.join(DATA_DIR, "X_forge.npy"), mmap_mode='r')
    y    = np.load(os.path.join(DATA_DIR, "y_forge.npy"))
    pred = np.load(os.path.join(DATA_DIR, "forge_predictions.npy"))
    conf = np.load(os.path.join(DATA_DIR, "forge_confidences.npy"))

    # Pick 1 clear event, 1 clear noise, 1 ambiguous
    indices   = np.arange(len(y))
    ev_idx    = indices[(y==1) & (pred==1)][0]   # first correct event
    no_idx    = indices[(y==0) & (pred==1)][0]   # first wrong noise (high risk)
    amb_idx   = indices[np.abs(conf - 0.5) < 0.1][0]  # most ambiguous

    samples = [
        ("E", ev_idx,  "CLEAR EVENT — correct prediction"),
        ("N", no_idx,  "CLEAR NOISE — wrong prediction (high AER risk)"),
        ("A", amb_idx, "AMBIGUOUS — borderline confidence"),
    ]

    pipeline = SafeGEMPipeline(pre_event_s=PRE_EVENT_S)
    gen      = ExplanationGenerator()

    all_lines = []
    all_lines.append("=" * 80)
    all_lines.append("FULL EXPLANATION TEXTS — VOCABULARY CALIBRATION")
    all_lines.append("Read these carefully. Any phrase that SHOULD be caught")
    all_lines.append("but isn't in the vocabulary → add it to claim_extractor.py")
    all_lines.append("=" * 80)

    for tag, idx, description in samples:
        sample_id = f"FORGE_{tag}_{idx:04d}"
        signal    = X[idx, 0]                        # (361, 2400)
        geo_data  = signal.mean(axis=0)
        gt        = int(y[idx])
        p         = int(pred[idx])
        c         = float(conf[idx])
        geo_conf  = c if p == 1 else 1.0 - c

        all_lines.append(f"\n{'─'*80}")
        all_lines.append(f"SAMPLE: {sample_id}  ({description})")
        all_lines.append(f"GT={gt}  Pred={p}  Conf={c:.3f}  "
                         f"{'CORRECT' if p==gt else 'WRONG'}")
        all_lines.append(f"{'─'*80}")

        print(f"\n[{sample_id}] {description}")
        print(f"  GT={gt} Pred={p} Conf={c:.3f}")

        # Get symbolic evidence first
        first = pipeline.run(
            sample_id=sample_id, das_data=signal, fs=SAMPLING_RATE,
            ground_truth=gt, model_prediction=p, model_confidence=c,
            explanation="placeholder", strategy_name="_init",
            geo_data=geo_data, geo_confidence=geo_conf,
        )
        evidence = first.symbolic_evidence

        all_lines.append(f"\nSYMBOLIC EVIDENCE:")
        all_lines.append(f"  SNR level         : {evidence.snr_level}")
        all_lines.append(f"  Amplitude contrast: {evidence.amplitude_contrast}")
        all_lines.append(f"  Frequency band    : {evidence.frequency_band}")
        all_lines.append(f"  High-freq content : {evidence.high_freq_content}")
        all_lines.append(f"  Onset character   : {evidence.onset_character}")
        all_lines.append(f"  DAS coherence     : {evidence.das_coherence}")
        all_lines.append(f"  Event support     : {evidence.event_support}")
        all_lines.append(f"  Signal ambiguity  : {evidence.signal_ambiguity}")
        all_lines.append(f"  Multimodal        : {evidence.multimodal_agreement}")

        for strategy in STRATEGIES:
            print(f"  Generating {strategy}...", end=" ", flush=True)

            gen_result = gen.generate_full(
                strategy   = strategy,
                prediction = p,
                confidence = c,
                evidence   = evidence,
            )

            if not gen_result.success:
                print(f"FAILED: {gen_result.error}")
                all_lines.append(f"\n  [{strategy}] FAILED: {gen_result.error}")
                continue

            print(f"done ({gen_result.duration_s:.1f}s)")

            result = pipeline.run(
                sample_id=sample_id, das_data=signal, fs=SAMPLING_RATE,
                ground_truth=gt, model_prediction=p, model_confidence=c,
                explanation=gen_result.explanation,
                strategy_name=strategy,
                geo_data=geo_data, geo_confidence=geo_conf,
            )

            all_lines.append(f"\n  [{strategy}]")
            all_lines.append(f"  EIRRS={result.eirrs_result.eirrs_equal:.3f}  "
                             f"Risk={result.eirrs_result.risk_tier.upper()}  "
                             f"SGS={result.grounding_report.signal_grounding_score:.3f}  "
                             f"Claims={result.grounding_report.total_claims}")
            all_lines.append(f"\n  FULL EXPLANATION TEXT:")
            all_lines.append(f"  {gen_result.explanation}")

            if result.grounding_report.total_claims > 0:
                all_lines.append(f"\n  EXTRACTED CLAIMS:")
                for detail in result.grounding_report.claim_details:
                    icon = {"supported":"✓","contradicted":"✗","unavailable":"?"}.get(
                        detail.result, "?")
                    all_lines.append(
                        f"    {icon} [{detail.claim.feature_group}/"
                        f"{detail.claim.polarity}] "
                        f'"{detail.claim.raw_phrase}" → {detail.result}'
                    )
            else:
                all_lines.append(f"\n  NO CLAIMS EXTRACTED — vocabulary gap!")
                all_lines.append(f"  → Read explanation above and add matching phrases")

    # Write output
    output = "\n".join(all_lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n{'='*60}")
    print(f"Full texts saved to: {OUTPUT_TXT}")
    print("Open this file and read what Gemma 4 actually writes.")
    print("Add any unmatched signal phrases to safe_gem/claim_extractor.py")


if __name__ == "__main__":
    main()
