"""
multi_llm_evaluation.py
------------------------
Runs the SAFE-GEM pipeline across three LLMs to assess whether
EIRRS rankings are model-agnostic, as recommended in Section 8.6
(Future Work) of the paper.

LLMs evaluated:
    M1  Qwen 2.5 7B    (qwen2.5:7b)        — main paper model
    M2  Llama 3.2 3B   (llama3.2:latest)   — already installed
    M3  Mistral 7B     (mistral:7b-instruct) — pull if needed

Evaluation:
    - Same 80 Utah FORGE samples as the main experiment
    - Same 5 prompting strategies (P1–P5)
    - Full EIRRS scoring for each model × strategy combination
    - Produces multi_llm_results.csv + comparison figures

Pull models first:
    ollama pull qwen2.5:7b
    ollama pull llama3.2:latest
    ollama pull mistral:7b-instruct
    ollama serve

Run from your safe_gem folder:
    python multi_llm_evaluation.py
"""

import numpy as np
import os
import sys
import csv
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Dataset"
RESULTS_DIR = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Results"

X_PATH    = os.path.join(DATA_DIR, "X_forge.npy")
Y_PATH    = os.path.join(DATA_DIR, "y_forge.npy")
PRED_PATH = os.path.join(DATA_DIR, "forge_predictions.npy")
CONF_PATH = os.path.join(DATA_DIR, "forge_confidences.npy")

SAMPLING_RATE = 500.0
PRE_EVENT_S   = 0.5

# ── LLM configurations ────────────────────────────────────────────────────────
MODELS = [
    {
        "id":          "M1",
        "name":        "Qwen 2.5 7B",
        "ollama_tag":  "qwen2.5:7b",
        "temperature": 0.3,
        "max_tokens":  220,
    },
    {
        "id":          "M2",
        "name":        "Llama 3.2 3B",
        "ollama_tag":  "llama3.2:latest",
        "temperature": 0.3,
        "max_tokens":  220,
    },
    {
        "id":          "M3",
        "name":        "Mistral 7B",
        "ollama_tag":  "mistral:7b-instruct",
        "temperature": 0.3,
        "max_tokens":  220,
    },
    {
        "id":          "M4",
        "name":        "Gemma 2 9B",
        "ollama_tag":  "gemma2:9b",
        "temperature": 0.3,
        "max_tokens":  220,
    },
    {
        "id":          "M5",
        "name":        "Phi-3.5 Mini",
        "ollama_tag":  "phi3.5:latest",
        "temperature": 0.3,
        "max_tokens":  220,
    },
    {
        "id":          "M6",
        "name":        "DeepSeek-R1 7B",
        "ollama_tag":  "deepseek-r1:7b",
        "temperature": 0.3,
        "max_tokens":  220,
    },
]

STRATEGIES = ["P1_raw", "P2_grounded", "P3_uncertainty",
              "P4_constrained", "P5_safe_gem"]

# Sample counts — same as main experiment
N_CLEAR_EVENTS   = 30
N_CLEAR_NOISE    = 30
N_AMBIGUOUS      = 20


# ── Imports ───────────────────────────────────────────────────────────────────

from safe_gem import SafeGEMPipeline, SafeGEMBatch
from generate_explanation import ExplanationGenerator, STRATEGIES as STRAT_FUNCS


# ── Sample selection (identical to main experiment) ───────────────────────────

def select_samples(y, pred, conf):
    indices     = np.arange(len(y))
    event_mask  = y == 1
    noise_mask  = y == 0

    ev_idx   = indices[event_mask]
    ev_conf  = conf[event_mask]
    events   = ev_idx[np.argsort(ev_conf)[::-1]][:N_CLEAR_EVENTS].tolist()

    no_idx   = indices[noise_mask]
    no_conf  = conf[noise_mask]
    noises   = no_idx[np.argsort(no_conf)[::-1]][:N_CLEAR_NOISE].tolist()

    used     = set(events + noises)
    rem      = [i for i in indices if i not in used]
    rem_conf = conf[rem]
    amb_sort = np.array(rem)[np.argsort(np.abs(rem_conf - 0.5))]
    ambig    = amb_sort[:N_AMBIGUOUS].tolist()

    return {
        "clear_event":    events,
        "clear_nonevent": noises,
        "ambiguous":      ambig,
    }


def get_group(sample_id):
    if "_E_" in sample_id: return "E"
    if "_N_" in sample_id: return "N"
    return "A"


# ── Ollama connectivity check ─────────────────────────────────────────────────

import requests

def check_model(ollama_tag):
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(ollama_tag in m for m in models)
    except Exception:
        return False


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_multi_llm():
    print("=" * 70)
    print("  SAFE-GEM MULTI-LLM EVALUATION")
    print("=" * 70)

    # Load data
    X    = np.load(X_PATH, mmap_mode="r")
    y    = np.load(Y_PATH)
    pred = np.load(PRED_PATH)
    conf = np.load(CONF_PATH)
    print(f"\nData loaded: {X.shape}, {int(y.sum())} events, {int((y==0).sum())} noise")

    # Select samples
    groups = select_samples(y, pred, conf)
    all_indices = (
        [("E", i) for i in groups["clear_event"]] +
        [("N", i) for i in groups["clear_nonevent"]] +
        [("A", i) for i in groups["ambiguous"]]
    )
    print(f"Samples: {len(all_indices)} total "
          f"(E={len(groups['clear_event'])}, "
          f"N={len(groups['clear_nonevent'])}, "
          f"A={len(groups['ambiguous'])})")

    # Check model availability
    print("\nChecking model availability:")
    available_models = []
    for m in MODELS:
        ok = check_model(m["ollama_tag"])
        status = "✓ available" if ok else "✗ not found — run: ollama pull " + m["ollama_tag"]
        print(f"  {m['id']} {m['name']:15s} ({m['ollama_tag']}): {status}")
        if ok:
            available_models.append(m)

    if not available_models:
        print("\nERROR: No models available. Start Ollama and pull models first.")
        return

    total_calls = len(available_models) * len(all_indices) * len(STRATEGIES)
    print(f"\nWill run: {len(available_models)} models × "
          f"{len(all_indices)} samples × {len(STRATEGIES)} strategies "
          f"= {total_calls} calls")

    ans = input("Proceed? (y/n): ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    # Initialise pipeline (shared across all models)
    pipeline = SafeGEMPipeline(pre_event_s=PRE_EVENT_S)

    # Results storage
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "multi_llm_results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "model_name", "sample_id", "group",
            "ground_truth", "prediction", "correct", "confidence",
            "strategy", "sgs", "ucr",
            "eirrs_equal", "eirrs_error", "eirrs_hf",
            "risk_tier", "AER", "EC", "UTC", "SA", "UO", "MI",
            "signal_ambiguity", "explanation_length",
        ])

    all_results = defaultdict(list)

    for model_cfg in available_models:
        model_id   = model_cfg["id"]
        model_name = model_cfg["name"]
        tag        = model_cfg["ollama_tag"]

        print(f"\n{'='*70}")
        print(f"  Model: {model_id} — {model_name} ({tag})")
        print(f"{'='*70}")

        gen = ExplanationGenerator(
            model       = tag,
            temperature = model_cfg["temperature"],
            max_tokens  = model_cfg["max_tokens"],
        )
        gen.warmup()

        for group_tag, idx in all_indices:
            sample_id = f"FORGE_{group_tag}_{idx:04d}"

            signal   = X[idx, 0] if X.ndim == 4 else X[idx]
            das_data = signal
            geo_data = das_data.mean(axis=0) if das_data.ndim == 2 else das_data
            gt       = int(y[idx])
            p        = int(pred[idx])
            c        = float(conf[idx])
            geo_conf = c if p == 1 else 1.0 - c

            # Extract evidence once per sample (model-independent)
            init_result = pipeline.run(
                sample_id=sample_id, das_data=das_data, fs=SAMPLING_RATE,
                ground_truth=gt, model_prediction=p, model_confidence=c,
                explanation="placeholder", strategy_name="_init",
                geo_data=geo_data, geo_confidence=geo_conf,
            )
            evidence = init_result.symbolic_evidence

            print(f"\n  [{model_id}] {sample_id}  GT={gt}  Pred={p}  "
                  f"Conf={c:.2f}  {'OK' if p==gt else 'WRONG'}")

            for strategy in STRATEGIES:
                print(f"    {strategy}...", end=" ", flush=True)
                t0 = time.time()

                gen_result = gen.generate_full(
                    strategy   = strategy,
                    prediction = p,
                    confidence = c,
                    evidence   = evidence,
                )

                if not gen_result.success or not gen_result.explanation:
                    print(f"FAILED ({gen_result.error})")
                    continue

                result = pipeline.run(
                    sample_id=sample_id, das_data=das_data, fs=SAMPLING_RATE,
                    ground_truth=gt, model_prediction=p, model_confidence=c,
                    explanation=gen_result.explanation,
                    strategy_name=strategy,
                    geo_data=geo_data, geo_confidence=geo_conf,
                )

                elapsed = time.time() - t0
                print(f"EIRRS={result.eirrs_result.eirrs_equal:.3f} "
                      f"({elapsed:.1f}s)")

                row_data = {
                    "model_id":         model_id,
                    "model_name":       model_name,
                    "sample_id":        sample_id,
                    "group":            group_tag,
                    "ground_truth":     gt,
                    "prediction":       p,
                    "correct":          int(p == gt),
                    "confidence":       round(c, 3),
                    "strategy":         strategy,
                    "sgs":              round(result.grounding_report.signal_grounding_score, 3),
                    "ucr":              round(result.grounding_report.unsupported_claim_rate, 3),
                    "eirrs_equal":      round(result.eirrs_result.eirrs_equal, 3),
                    "eirrs_error":      round(result.eirrs_result.eirrs_error_heavy, 3),
                    "eirrs_hf":         round(result.eirrs_result.eirrs_hf_heavy, 3),
                    "risk_tier":        result.eirrs_result.risk_tier,
                    "AER":              result.eirrs_result.AER,
                    "EC":               result.eirrs_result.EC,
                    "UTC":              result.eirrs_result.UTC,
                    "SA":               result.eirrs_result.SA,
                    "UO":               result.eirrs_result.UO,
                    "MI":               result.eirrs_result.MI,
                    "signal_ambiguity": result.symbolic_evidence.signal_ambiguity,
                    "explanation_length": len(gen_result.explanation),
                }

                all_results[model_id].append(row_data)

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(row_data.values()))

    print(f"\n\nResults saved to: {csv_path}")
    summarise(all_results, available_models)
    plot_multi_llm(csv_path)


# ── Summary table ─────────────────────────────────────────────────────────────

def summarise(all_results, available_models):
    print("\n" + "=" * 70)
    print("  MULTI-LLM SUMMARY")
    print("=" * 70)

    strat_order = ["P1_raw", "P2_grounded", "P3_uncertainty",
                   "P4_constrained", "P5_safe_gem"]

    header = f"{'Strategy':<22}"
    for m in available_models:
        header += f"  {m['id']} EIRRS"
    print(header)
    print("-" * (22 + len(available_models) * 10))

    for strategy in strat_order:
        row = f"{strategy:<22}"
        for m in available_models:
            rows   = [r for r in all_results[m["id"]] if r["strategy"] == strategy]
            if rows:
                mean_e = sum(r["eirrs_equal"] for r in rows) / len(rows)
                row += f"  {mean_e:7.3f}"
            else:
                row += "       —"
        print(row)

    print("\nRanking stability check:")
    rankings = {}
    for m in available_models:
        means = {}
        for s in strat_order:
            rows = [r for r in all_results[m["id"]] if r["strategy"] == s]
            if rows:
                means[s] = sum(r["eirrs_equal"] for r in rows) / len(rows)
        sorted_strats = sorted(means, key=means.get)
        rankings[m["id"]] = sorted_strats
        print(f"  {m['id']} {m['name']}: {' < '.join(sorted_strats)}")

    # Check if P5 is best in all models
    p5_best = all(rankings[m["id"]][0] == "P5_safe_gem"
                  for m in available_models if m["id"] in rankings)
    print(f"\n  P5 safest across all models: {'YES — ranking stable' if p5_best else 'NO — ranking differs'}")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_multi_llm(csv_path):
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/pandas not available — skipping plots")
        return

    df = pd.read_csv(csv_path)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    strat_order  = ["P1_raw", "P2_grounded", "P3_uncertainty",
                    "P4_constrained", "P5_safe_gem"]
    strat_labels = ["P1\nRaw", "P2\nGrounded", "P3\nUncertain", "P4\nConstrained", "P5\nSAFE-GEM"]
    models       = df["model_id"].unique()
    model_names  = {row["model_id"]: row["model_name"]
                    for _, row in df[["model_id","model_name"]].drop_duplicates().iterrows()}

    model_colors = {"M1": "#3498db", "M2": "#e74c3c", "M3": "#27ae60",
                    "M4": "#9b59b6", "M5": "#f39c12", "M6": "#1abc9c"}
    x = np.arange(len(strat_order))
    w = 0.25

    # ── Figure A: EIRRS by model × strategy ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Multi-LLM EIRRS Comparison — SAFE-GEM Framework",
                 fontsize=12, fontweight="bold")

    for mi, model_id in enumerate(sorted(models)):
        means = [df[(df["model_id"]==model_id) & (df["strategy"]==s)]["eirrs_equal"].mean()
                 for s in strat_order]
        stds  = [df[(df["model_id"]==model_id) & (df["strategy"]==s)]["eirrs_equal"].std()
                 for s in strat_order]
        offset = (mi - (len(models)-1)/2) * w
        label  = model_names.get(model_id, model_id)
        ax.bar(x + offset, means, w, label=label,
               color=model_colors.get(model_id, "#888"),
               alpha=0.82, yerr=stds, capsize=3,
               error_kw={"linewidth": 1.1, "ecolor": "#555"})

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=9)
    ax.set_ylabel("Mean EIRRS (equal-weight)")
    ax.set_ylim(0, 0.55)
    ax.axhline(0.33, color="#e74c3c", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Medium-risk threshold")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_a = os.path.join(RESULTS_DIR, "multi_llm_eirrs.png")
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_a}")

    # ── Figure B: UCR by model × strategy ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Multi-LLM Unsupported Claim Rate — SAFE-GEM Framework",
                 fontsize=12, fontweight="bold")

    for mi, model_id in enumerate(sorted(models)):
        means = [df[(df["model_id"]==model_id) & (df["strategy"]==s)]["ucr"].mean()
                 for s in strat_order]
        offset = (mi - (len(models)-1)/2) * w
        label  = model_names.get(model_id, model_id)
        ax.bar(x + offset, means, w, label=label,
               color=model_colors.get(model_id, "#888"), alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=9)
    ax.set_ylabel("Mean UCR (lower = safer)")
    ax.set_ylim(0, 0.35)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_b = os.path.join(RESULTS_DIR, "multi_llm_ucr.png")
    plt.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_b}")

    # ── Figure C: P5 EIRRS distribution by model (box plot) ──────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("P5 SAFE-GEM EIRRS Distribution by Model",
                 fontsize=12, fontweight="bold")

    data   = [df[(df["model_id"]==m) & (df["strategy"]=="P5_safe_gem")]["eirrs_equal"].values
              for m in sorted(models)]
    labels = [model_names.get(m, m) for m in sorted(models)]
    colors = [model_colors.get(m, "#888") for m in sorted(models)]

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_ylabel("EIRRS (equal-weight)")
    ax.set_ylim(0, 0.6)
    ax.axhline(0.33, color="#e74c3c", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Medium-risk threshold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_c = os.path.join(RESULTS_DIR, "multi_llm_p5_distribution.png")
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_c}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_multi_llm()
