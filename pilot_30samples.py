"""
pilot_30samples.py
-------------------
30-sample SAFE-GEM pilot using Utah FORGE numpy dataset.

Folder structure expected:
    E:/Events_14/Data/
        Dataset/
            X_forge.npy      ← FORGE signals
            y_forge.npy      ← FORGE labels (0=noise, 1=event)
            forge_meta       ← metadata (sampling rate etc.)
        Model/               ← trained model files (optional)

Run from E:/Events_14/Data/:
    python pilot_30samples.py

Or set DATA_DIR below to your actual path.
"""

import numpy as np
import os
import json
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Path configuration ───────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Dataset"
RESULTS_DIR = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Results"

X_PATH = os.path.join(DATA_DIR, "X_forge.npy")
Y_PATH = os.path.join(DATA_DIR, "y_forge.npy")
META_PATH = os.path.join(DATA_DIR, "forge_meta")

# ── Signal configuration (update after inspecting forge_meta) ────────────────
SAMPLING_RATE = 500.0    # Hz — update this from forge_meta if different
PRE_EVENT_S   = 0.5      # seconds of pre-event noise window

# ── Pilot configuration ──────────────────────────────────────────────────────
N_CLEAR_EVENTS    = 30
N_CLEAR_NOISE     = 30
N_AMBIGUOUS       = 20   # highest model uncertainty
STRATEGIES        = ["P1_raw", "P2_grounded", "P3_uncertainty",
                     "P4_constrained", "P5_safe_gem"]


# ─── Step 1: Load and inspect data ───────────────────────────────────────────

def load_and_inspect():
    """Load numpy arrays and print shape/label info."""
    print("=" * 60)
    print("  SAFE-GEM 30-SAMPLE PILOT")
    print("=" * 60)

    # Read forge_meta for sampling rate if available
    fs = SAMPLING_RATE
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta_text = f.read()
        print(f"\nforge_meta:\n{meta_text}\n")
        # Try to extract sampling rate from meta text
        for line in meta_text.lower().splitlines():
            if "sampling" in line or "sample_rate" in line or "fs" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        fs = float(parts[1].strip().split()[0])
                        print(f"Sampling rate from meta: {fs} Hz")
                    except Exception:
                        pass

    print(f"\nLoading {X_PATH}")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"\nX_forge shape : {X.shape}")
    print(f"y_forge shape : {y.shape}")
    print(f"Label counts  : {Counter(y.tolist())}")
    print(f"  0 = noise/non-event")
    print(f"  1 = microseismic event")

    # Infer dimensions
    if X.ndim == 3:
        n_samples, n_channels, n_timepoints = X.shape
        print(f"\nFormat: {n_samples} samples × {n_channels} channels × {n_timepoints} timepoints")
        print(f"Duration per sample: {n_timepoints / fs:.2f} s at {fs} Hz")
    elif X.ndim == 2:
        n_samples, n_timepoints = X.shape
        n_channels = 1
        print(f"\nFormat: {n_samples} samples × {n_timepoints} timepoints (single channel)")
        print(f"Duration per sample: {n_timepoints / fs:.2f} s at {fs} Hz")
    elif X.ndim == 4:
        n_samples, n_modalities, n_channels, n_timepoints = X.shape
        print(f"Format: {n_samples} x {n_modalities} x {n_channels} x {n_timepoints}")
        print(f"Duration: {n_timepoints / fs:.2f} s at {fs} Hz")
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    return X, y, fs, n_channels


# ─── Step 2: Load model predictions ──────────────────────────────────────────

def load_or_compute_predictions(X, y, n_channels):
    """
    Try to load saved model predictions. If not available, use STA/LTA.
    Returns (predictions, confidences) arrays of shape (n_samples,).
    """
    pred_path = os.path.join(DATA_DIR, "forge_predictions.npy")
    conf_path = os.path.join(DATA_DIR, "forge_confidences.npy")

    if os.path.exists(pred_path) and os.path.exists(conf_path):
        predictions = np.load(pred_path)
        confidences = np.load(conf_path)
        print(f"\nLoaded saved predictions from {pred_path}")
        acc = np.mean(predictions == y)
        print(f"Model accuracy on FORGE: {acc:.3f}")
        return predictions, confidences

    print("\nNo saved predictions found. Computing STA/LTA-based confidence...")
    print("(For real results, save your URDI model predictions as forge_predictions.npy)")

    from scipy import signal as scipy_signal

    predictions = []
    confidences = []

    for i in range(len(X)):
        if X.ndim == 4:
            trace = X[i, 0].mean(axis=0)   # (1,361,2400) → mean channels → (2400,)
        elif X.ndim == 3:
            trace = X[i].mean(axis=0)
        else:
            trace = X[i]

        # Simple STA/LTA characteristic function
        cf = trace ** 2
        sta_n = max(1, int(0.01 * 500))
        lta_n = max(sta_n + 1, int(0.10 * 500))
        sta = np.convolve(cf, np.ones(sta_n) / sta_n, mode='same')
        lta = np.convolve(cf, np.ones(lta_n) / lta_n, mode='same')
        ratio = sta / (lta + 1e-10)
        peak_ratio = float(np.max(ratio))

        # Convert to pseudo-probability
        prob = float(np.clip((peak_ratio - 2) / 10, 0, 1))
        pred = 1 if prob >= 0.5 else 0

        predictions.append(pred)
        confidences.append(max(prob, 1 - prob))  # confidence = distance from 0.5

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    acc = np.mean(predictions == y)
    print(f"STA/LTA accuracy on FORGE: {acc:.3f}")

    return predictions, confidences


# ─── Step 3: Select 30 pilot samples ─────────────────────────────────────────

def select_pilot_samples(y, predictions, confidences):
    """
    Select 30 samples:
    - 10 clear events:     label=1, prediction=1, high confidence
    - 10 clear non-events: label=0, prediction=0, high confidence
    - 10 ambiguous:        any label, low confidence (borderline cases)
    """
    indices = np.arange(len(y))

    correct_mask   = predictions == y
    event_mask     = y == 1
    nonevent_mask  = y == 0

    # Select by ground truth label + highest confidence
    # (robust when STA/LTA accuracy is low on noisy data)
    event_idx    = indices[event_mask]
    event_conf   = confidences[event_mask]
    clear_events = event_idx[np.argsort(event_conf)[::-1]][:N_CLEAR_EVENTS].tolist()

    noise_idx    = indices[nonevent_mask]
    noise_conf   = confidences[nonevent_mask]
    clear_noises = noise_idx[np.argsort(noise_conf)[::-1]][:N_CLEAR_NOISE].tolist()

    # Ambiguous: confidence closest to 0.5 (most uncertain)
    used = set(clear_events + clear_noises)
    remaining = [i for i in indices if i not in used]
    remaining_conf = confidences[remaining]
    ambiguity_score = np.abs(remaining_conf - 0.5)   # 0 = most ambiguous
    ambiguous_sorted = np.array(remaining)[np.argsort(ambiguity_score)]
    ambiguous = ambiguous_sorted[:N_AMBIGUOUS].tolist()

    print(f"\nPilot sample selection:")
    print(f"  Clear events    : {len(clear_events)} samples")
    print(f"  Clear non-events: {len(clear_noises)} samples")
    print(f"  Ambiguous       : {len(ambiguous)} samples")
    print(f"  Total           : {len(clear_events) + len(clear_noises) + len(ambiguous)}")

    return {
        "clear_event":    clear_events,
        "clear_nonevent": clear_noises,
        "ambiguous":      ambiguous,
    }


# ─── Step 4: Run the full pilot ───────────────────────────────────────────────

def run_pilot(X, y, predictions, confidences, sample_groups, fs, n_channels):
    """Run SAFE-GEM + Gemma 4 on all 30 pilot samples."""

    from safe_gem import SafeGEMPipeline, SafeGEMBatch
    from generate_explanation import ExplanationGenerator, run_full_sample

    pipeline = SafeGEMPipeline(pre_event_s=PRE_EVENT_S)
    gen      = ExplanationGenerator()   # uses gemma4:e4b by default
    batch    = SafeGEMBatch(pipeline)

    gen.print_status()

    all_indices = (
        [("E", i) for i in sample_groups["clear_event"]] +
        [("N", i) for i in sample_groups["clear_nonevent"]] +
        [("A", i) for i in sample_groups["ambiguous"]]
    )

    print(f"\nRunning {len(all_indices)} samples × {len(STRATEGIES)} strategies "
          f"= {len(all_indices) * len(STRATEGIES)} total SAFE-GEM calls\n")

    for group_tag, idx in all_indices:
        sample_id = f"FORGE_{group_tag}_{idx:04d}"

        # Extract signal
        signal = X[idx]
        if signal.ndim == 3:
            das_data = signal[0]
        elif signal.ndim == 2:
            das_data = signal
        else:
            das_data = signal

        geo_data = das_data.mean(axis=0) if das_data.ndim == 2 else das_data
        geo_conf = float(confidences[idx]) if int(predictions[idx]) == 1 \
                   else 1.0 - float(confidences[idx])

        gt   = int(y[idx])
        pred = int(predictions[idx])
        conf = float(confidences[idx])

        print(f"\n{'─'*50}")
        print(f"Sample {sample_id}  GT={gt}  Pred={pred}  Conf={conf:.2f}  "
              f"({'CORRECT' if pred == gt else 'WRONG'})")

        results = run_full_sample(
            pipeline          = pipeline,
            gen               = gen,
            sample_id         = sample_id,
            das_data          = das_data,
            fs                = fs,
            ground_truth      = gt,
            model_prediction  = pred,
            model_confidence  = conf,
            geo_data          = geo_data,
            geo_confidence    = geo_conf,
            strategies        = STRATEGIES,
        )

        for r in results:
            batch.add_result(r)

    return batch


# ─── Step 5: Save and print results ──────────────────────────────────────────

def save_results(batch):
    """Print summary table and save results to CSV."""
    print("\n" + "=" * 60)
    print("  PILOT RESULTS SUMMARY")
    print("=" * 60)
    batch.print_summary_table()

    # Save per-sample results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "pilot_results.csv")

    with open(csv_path, "w") as f:
        f.write("sample_id,ground_truth,prediction,correct,confidence,"
                "strategy,sgs,ucr,eirrs_equal,eirrs_error,eirrs_hf,"
                "risk_tier,AER,EC,UTC,SA,UO,MI,"
                "signal_ambiguity,multimodal_agreement\n")

        for r in batch.results:
            f.write(
                f"{r.sample_id},{r.ground_truth},{r.model_prediction},"
                f"{int(r.model_correct)},{r.model_confidence:.3f},"
                f"{r.explanation_strategy},"
                f"{r.grounding_report.signal_grounding_score:.3f},"
                f"{r.grounding_report.unsupported_claim_rate:.3f},"
                f"{r.eirrs_result.eirrs_equal:.3f},"
                f"{r.eirrs_result.eirrs_error_heavy:.3f},"
                f"{r.eirrs_result.eirrs_hf_heavy:.3f},"
                f"{r.eirrs_result.risk_tier},"
                f"{r.eirrs_result.AER},{r.eirrs_result.EC},"
                f"{r.eirrs_result.UTC},{r.eirrs_result.SA},"
                f"{r.eirrs_result.UO},{r.eirrs_result.MI},"
                f"{r.symbolic_evidence.signal_ambiguity},"
                f"{r.symbolic_evidence.multimodal_agreement}\n"
            )

    print(f"\nResults saved to: {csv_path}")
    print("Open this CSV in Excel to review per-sample EIRRS scores.")


# ─── Step 6: Vocabulary validation ───────────────────────────────────────────

def validate_vocabulary(batch, n_show=5):
    """
    Print the first N claim extraction reports so you can manually
    check whether the controlled vocabulary is catching real claims.
    This is the calibration check before scaling to the full dataset.
    """
    print("\n" + "=" * 60)
    print("  VOCABULARY VALIDATION (first 5 P1_raw samples)")
    print("  Check: are the extracted claims correct?")
    print("=" * 60)

    shown = 0
    for r in batch.results:
        if r.explanation_strategy != "P1_raw":
            continue
        if shown >= n_show:
            break

        print(f"\n[{r.sample_id}]")
        print(f"Explanation: {r.explanation_text[:200]}...")
        print(f"Claims found: {r.grounding_report.total_claims}")
        for detail in r.grounding_report.claim_details[:6]:
            icon = {"supported": "✓", "contradicted": "✗", "unavailable": "?"}.get(
                detail.result, "?"
            )
            print(f"  {icon} {detail.claim.feature_group}/{detail.claim.polarity} "
                  f'"{detail.claim.raw_phrase}" → {detail.result}')
        shown += 1

    print("\nIf claims look wrong, edit VOCABULARY in safe_gem/claim_extractor.py")
    print("and re-run. The vocabulary calibration step is critical.\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data
    X, y, fs, n_channels = load_and_inspect()

    # 2. Load or compute predictions
    predictions, confidences = load_or_compute_predictions(X, y, n_channels)

    # 3. Select 30 pilot samples
    sample_groups = select_pilot_samples(y, predictions, confidences)

    # 4. Confirm before running (generation takes time)
    n_total = sum(len(v) for v in sample_groups.values())
    n_calls  = n_total * len(STRATEGIES)
    print(f"\nReady to run {n_calls} explanation calls via Gemma 4.")
    print("Make sure Ollama is running: ollama serve")
    ans = input("Proceed? (y/n): ").strip().lower()
    if ans != "y":
        print("Aborted.")
        exit(0)

    # 5. Run pilot
    batch = run_pilot(X, y, predictions, confidences, sample_groups, fs, n_channels)

    # 6. Save and display results
    save_results(batch)

    # 7. Vocabulary validation
    validate_vocabulary(batch)

    print("\nPilot complete.")
    print("Next: review pilot_results.csv, calibrate vocabulary, then scale to full dataset.")
