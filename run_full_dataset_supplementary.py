"""
run_full_dataset_supplementary.py
----------------------------------
SAFE-GEM Step 3: full-dataset supplementary analysis for the 494-sample Utah FORGE set.

What it does
------------
1. Loads X_forge.npy, y_forge.npy, forge_predictions.npy, forge_confidences.npy.
2. Generates explanations for every sample using the five SAFE-GEM prompting strategies.
3. Runs the existing SAFE-GEM pipeline on each explanation.
4. Saves row-level results and summary tables:
   - Results/full_dataset_safe_gem_results.csv
   - Results/full_dataset_safe_gem_summary.xlsx
   - Results/full_dataset_safe_gem_latex_tables.tex

Important
---------
This script does NOT change claim_extractor.py or the main SAFE-GEM pipeline.
It only runs the same pipeline on the full dataset and summarizes the results.

Recommended use
---------------
First test with a few samples:
    python .\run_full_dataset_supplementary.py --max-samples 3

Then run the full dataset:
    python .\run_full_dataset_supplementary.py

The script supports resume. If interrupted, run it again and it will skip
completed sample-strategy pairs already present in the CSV.
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Allow running from the safe_gem project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = Path(r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Dataset")
DEFAULT_RESULTS_DIR = Path(r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Results")

DEFAULT_STRATEGIES = [
    "P1_raw",
    "P2_grounded",
    "P3_uncertainty",
    "P4_constrained",
    "P5_safe_gem",
]

SAMPLING_RATE = 500.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Folder containing X_forge.npy, y_forge.npy, forge_predictions.npy, forge_confidences.npy")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Output folder for CSV/XLSX/LaTeX results")
    p.add_argument("--model", type=str, default="qwen2.5:7b",
                   help="Ollama/local model name used by generate_explanation.ExplanationGenerator")
    p.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES),
                   help="Comma-separated strategy names")
    p.add_argument("--start-index", type=int, default=0,
                   help="First dataset index to process")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Maximum number of samples to process; use for testing")
    p.add_argument("--retries", type=int, default=1,
                   help="Retries per failed generation call")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip generator warmup")
    return p.parse_args()


def ev_summary(ev) -> str:
    """Short string summary of symbolic evidence."""
    fields = [
        ("SNR", "snr_level"),
        ("FreqBand", "frequency_band"),
        ("HF", "high_freq_content"),
        ("Onset", "onset_character"),
        ("Amp", "amplitude_contrast"),
        ("DAS", "das_coherence"),
        ("Moveout", "moveout"),
        ("GeoSupport", "geophone_event_support"),
        ("IAC", "multimodal_agreement"),
        ("EvtSupport", "event_support"),
        ("Ambiguity", "signal_ambiguity"),
    ]
    parts = []
    for label, attr in fields:
        val = getattr(ev, attr, None)
        if val not in (None, "", "unavailable"):
            parts.append(f"{label}:{val}")
    return " | ".join(parts)


def load_existing_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["sample_index", "strategy", "generation_success"])
        # Skip only successful rows. Failed rows will be retried.
        ok = df[df["generation_success"].astype(str).str.lower().isin(["true", "1"])]
        return set(zip(ok["sample_index"].astype(int), ok["strategy"].astype(str)))
    except Exception:
        return set()


def append_row(csv_path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_float(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def generate_with_retries(gen, strategy, prediction, confidence, evidence, retries: int):
    last_result = None
    for attempt in range(retries + 1):
        try:
            result = gen.generate_full(
                strategy=strategy,
                prediction=prediction,
                confidence=confidence,
                evidence=evidence,
            )
            last_result = result
            if getattr(result, "success", False) and getattr(result, "explanation", None):
                return result
        except Exception as e:
            last_result = e
        time.sleep(1.0)
    return last_result


def row_fieldnames() -> List[str]:
    return [
        "sample_index", "sample_id", "strategy", "ground_truth", "prediction", "confidence",
        "correct", "generation_success", "generation_duration_s", "error_message",
        "explanation_text", "symbolic_evidence_summary",
        "total_claims", "supported_claims", "contradicted_claims", "unavailable_claims",
        "sgs", "ucr", "supported_claim_rate",
        "eirrs_equal", "eirrs_error_heavy", "eirrs_hf_heavy", "risk_tier",
        "snr_level", "frequency_band", "high_freq_content", "onset_character",
        "amplitude_contrast", "das_coherence", "moveout", "geophone_event_support",
        "multimodal_agreement", "event_support", "signal_ambiguity", "ambiguity_score",
    ]


def summarize_results(results_csv: Path, results_dir: Path) -> None:
    if not results_csv.exists():
        print("No results CSV found; summary skipped.")
        return

    df = pd.read_csv(results_csv)
    ok = df[df["generation_success"].astype(str).str.lower().isin(["true", "1"])]
    if ok.empty:
        print("No successful generations; summary skipped.")
        return

    numeric_cols = ["sgs", "ucr", "eirrs_equal", "eirrs_error_heavy", "eirrs_hf_heavy", "confidence"]
    for c in numeric_cols:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    def pct_tier(s: pd.Series, tier: str) -> float:
        return 100.0 * (s.astype(str).str.lower() == tier).mean()

    rows = []
    for strategy, g in ok.groupby("strategy"):
        e = g["eirrs_equal"].dropna().to_numpy()
        rows.append({
            "Strategy": strategy,
            "N successful": len(g),
            "Mean SGS": g["sgs"].mean(),
            "Mean UCR": g["ucr"].mean(),
            "Mean EIRRS": g["eirrs_equal"].mean(),
            "SD EIRRS": g["eirrs_equal"].std(ddof=1),
            "Low-risk %": pct_tier(g["risk_tier"], "low"),
            "Medium-risk %": pct_tier(g["risk_tier"], "medium"),
            "High-risk %": pct_tier(g["risk_tier"], "high"),
            "Mean total claims": g["total_claims"].mean(),
            "Mean contradicted claims": g["contradicted_claims"].mean(),
        })
    summary = pd.DataFrame(rows).sort_values("Mean EIRRS", ascending=True)
    summary["Rank by EIRRS"] = range(1, len(summary) + 1)

    # Threshold sensitivity: percentage screened as lower-risk at several provisional thresholds.
    threshold_rows = []
    for thr in [0.25, 0.33, 0.40, 0.50]:
        row = {"Threshold": thr}
        for strategy, g in ok.groupby("strategy"):
            row[f"{strategy} lower-risk %"] = 100.0 * (g["eirrs_equal"] <= thr).mean()
        threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)

    # Ambiguity breakdown.
    amb_rows = []
    for (strategy, amb), g in ok.groupby(["strategy", "signal_ambiguity"]):
        amb_rows.append({
            "Strategy": strategy,
            "Signal ambiguity": amb,
            "N": len(g),
            "Mean EIRRS": g["eirrs_equal"].mean(),
            "Mean SGS": g["sgs"].mean(),
            "Mean UCR": g["ucr"].mean(),
        })
    amb_df = pd.DataFrame(amb_rows).sort_values(["Strategy", "Signal ambiguity"])

    # Save Excel workbook.
    xlsx_path = results_dir / "full_dataset_safe_gem_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Full dataset summary", index=False)
        threshold_df.to_excel(writer, sheet_name="Threshold sensitivity", index=False)
        amb_df.to_excel(writer, sheet_name="Ambiguity breakdown", index=False)
        ok.to_excel(writer, sheet_name="Successful row results", index=False)

    # Save LaTeX tables.
    tex_path = results_dir / "full_dataset_safe_gem_latex_tables.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by run_full_dataset_supplementary.py\n\n")
        f.write("% Full-dataset summary table\n")
        f.write(summary.to_latex(index=False, float_format="%.3f"))
        f.write("\n\n% Threshold sensitivity table\n")
        f.write(threshold_df.to_latex(index=False, float_format="%.1f"))
        f.write("\n\n% Ambiguity breakdown table\n")
        f.write(amb_df.to_latex(index=False, float_format="%.3f"))

    print("\n=== Full-dataset summary ===")
    print(summary.to_string(index=False))
    print("\n=== Threshold sensitivity ===")
    print(threshold_df.to_string(index=False))
    print(f"\nSaved summary workbook: {xlsx_path}")
    print(f"Saved LaTeX tables:     {tex_path}")


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    from safe_gem import SafeGEMPipeline
    from generate_explanation import ExplanationGenerator

    print("Loading FORGE arrays...")
    X = np.load(args.dataset_dir / "X_forge.npy", mmap_mode="r")
    y = np.load(args.dataset_dir / "y_forge.npy")
    pred = np.load(args.dataset_dir / "forge_predictions.npy")
    conf = np.load(args.dataset_dir / "forge_confidences.npy")

    n_total = len(y)
    end_index = n_total if args.max_samples is None else min(n_total, args.start_index + args.max_samples)
    sample_indices = list(range(args.start_index, end_index))

    print(f"Dataset size: {n_total}")
    print(f"Processing indices: {args.start_index} to {end_index - 1} ({len(sample_indices)} samples)")
    print(f"Strategies: {strategies}")
    print(f"Planned calls this run: {len(sample_indices) * len(strategies)}")

    csv_path = args.results_dir / "full_dataset_safe_gem_results.csv"
    completed = load_existing_keys(csv_path)
    print(f"Already completed successful sample-strategy pairs: {len(completed)}")

    pipeline = SafeGEMPipeline(pre_event_s=0.5)
    gen = ExplanationGenerator(model=args.model)
    if not args.no_warmup:
        print("Warming up generator...")
        gen.warmup()

    fields = row_fieldnames()
    call_counter = 0
    t0 = time.time()

    for idx in sample_indices:
        das_data = X[idx, 0] if X.ndim == 4 else X[idx]
        geo_data = das_data.mean(axis=0) if getattr(das_data, "ndim", 1) == 2 else das_data
        gt = int(y[idx])
        p = int(pred[idx])
        c = float(conf[idx])
        geo_conf = c if p == 1 else 1.0 - c
        sample_id = f"FORGE_FULL_{idx:04d}"

        # First obtain symbolic evidence using a placeholder explanation.
        init = pipeline.run(
            sample_id=sample_id,
            das_data=das_data,
            fs=SAMPLING_RATE,
            ground_truth=gt,
            model_prediction=p,
            model_confidence=c,
            explanation="placeholder",
            strategy_name="_init",
            geo_data=geo_data,
            geo_confidence=geo_conf,
        )
        evidence = init.symbolic_evidence

        for strategy in strategies:
            if (idx, strategy) in completed:
                print(f"SKIP {sample_id} {strategy} already completed")
                continue

            call_counter += 1
            print(f"[{idx+1}/{n_total}] {sample_id} {strategy} GT={gt} Pred={p} Conf={c:.3f}", end=" ", flush=True)
            gen_result = generate_with_retries(gen, strategy, p, c, evidence, args.retries)

            success = bool(getattr(gen_result, "success", False) and getattr(gen_result, "explanation", None))
            duration = safe_float(getattr(gen_result, "duration_s", float("nan")))
            explanation = getattr(gen_result, "explanation", "") if success else ""
            error_msg = "" if success else str(gen_result)[:300]

            row = {k: "" for k in fields}
            row.update({
                "sample_index": idx,
                "sample_id": sample_id,
                "strategy": strategy,
                "ground_truth": gt,
                "prediction": p,
                "confidence": round(c, 6),
                "correct": int(p == gt),
                "generation_success": success,
                "generation_duration_s": duration,
                "error_message": error_msg,
                "explanation_text": explanation,
                "symbolic_evidence_summary": ev_summary(evidence),
            })

            if success:
                try:
                    result = pipeline.run(
                        sample_id=sample_id,
                        das_data=das_data,
                        fs=SAMPLING_RATE,
                        ground_truth=gt,
                        model_prediction=p,
                        model_confidence=c,
                        explanation=explanation,
                        strategy_name=strategy,
                        geo_data=geo_data,
                        geo_confidence=geo_conf,
                    )
                    gr = result.grounding_report
                    er = result.eirrs_result
                    ev = result.symbolic_evidence

                    row.update({
                        "total_claims": gr.total_claims,
                        "supported_claims": gr.supported,
                        "contradicted_claims": gr.contradicted,
                        "unavailable_claims": gr.unavailable,
                        "sgs": round(gr.signal_grounding_score, 6),
                        "ucr": round(gr.unsupported_claim_rate, 6),
                        "supported_claim_rate": round(gr.supported_claim_rate, 6),
                        "eirrs_equal": round(safe_float(getattr(er, "eirrs_equal", float("nan"))), 6),
                        "eirrs_error_heavy": round(safe_float(getattr(er, "eirrs_error_heavy", float("nan"))), 6),
                        "eirrs_hf_heavy": round(safe_float(getattr(er, "eirrs_hf_heavy", float("nan"))), 6),
                        "risk_tier": getattr(er, "risk_tier", ""),
                        "snr_level": ev.snr_level,
                        "frequency_band": ev.frequency_band,
                        "high_freq_content": ev.high_freq_content,
                        "onset_character": ev.onset_character,
                        "amplitude_contrast": ev.amplitude_contrast,
                        "das_coherence": ev.das_coherence,
                        "moveout": ev.moveout,
                        "geophone_event_support": ev.geophone_event_support,
                        "multimodal_agreement": ev.multimodal_agreement,
                        "event_support": ev.event_support,
                        "signal_ambiguity": ev.signal_ambiguity,
                        "ambiguity_score": ev.ambiguity_score,
                    })
                    print(f"done EIRRS={row['eirrs_equal']} tier={row['risk_tier']}")
                except Exception as e:
                    row["generation_success"] = False
                    row["error_message"] = f"Pipeline failed after generation: {e}"[:300]
                    print(f"PIPELINE FAILED: {e}")
            else:
                print("GENERATION FAILED")

            append_row(csv_path, row, fields)

    elapsed = time.time() - t0
    print(f"\nRun finished. New calls attempted: {call_counter}. Elapsed: {elapsed/60:.1f} min")
    print(f"Row-level CSV: {csv_path}")

    summarize_results(csv_path, args.results_dir)


if __name__ == "__main__":
    main()
