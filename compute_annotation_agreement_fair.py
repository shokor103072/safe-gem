"""
compute_annotation_agreement_fair.py
------------------------------------
Reads completed annotation Excel sheets and computes:
  - Inter-annotator agreement: Cohen's kappa
  - Precision, recall, and F1 of the automated claim extractor vs human annotations

Run from your safe_gem folder after both annotators complete the sheets:
    python .\compute_annotation_agreement_fixed.py

Expected files in RESULTS_DIR:
  - annotation_sheet_A.xlsx
  - annotation_sheet_B.xlsx
  - annotation_samples.csv

Main fixes compared with the previous script:
  1. Forward-fills merged Annot ID/Strategy cells before dropping blank claim rows.
  2. Supports both import styles: safe_gem.claim_extractor and local claim_extractor.
  3. Uses containment-aware token matching for human/extractor phrase alignment.
  4. Saves a clearer results workbook.
"""

import os
import sys
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.environ.get(
    "SAFE_GEM_RESULTS_DIR",
    r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Results",
)

SHEET_A = os.path.join(RESULTS_DIR, "annotation_sheet_A.xlsx")
SHEET_B = os.path.join(RESULTS_DIR, "annotation_sheet_B.xlsx")
SAMPLES = os.path.join(RESULTS_DIR, "annotation_samples.csv")

VALID_VERDICTS = ["SUPPORTED", "CONTRADICTED", "UNAVAILABLE"]
VALID_VERDICT_SET = set(VALID_VERDICTS)
MATCH_THRESHOLD = 0.50
STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "with", "in", "on", "for", "that", "this", "is", "are", "be", "as", "by", "it", "there", "while", "also", "from", "into"}


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_sheet(path, annotator_label):
    """Load annotation sheet from the Annotations tab."""
    df = pd.read_excel(path, sheet_name="Annotations", header=1)
    df.columns = [
        "annot_id",
        "strategy",
        "gt_pred_conf",
        "evidence",
        "explanation",
        "claim_num",
        "claim_phrase",
        "feature_group",
        "verdict",
        "notes",
    ]

    # Important: forward-fill before dropping rows, because Excel merged cells become NaN.
    df["annot_id"] = df["annot_id"].ffill()
    df["strategy"] = df["strategy"].ffill()
    df["gt_pred_conf"] = df["gt_pred_conf"].ffill()
    df["evidence"] = df["evidence"].ffill()
    df["explanation"] = df["explanation"].ffill()

    df["claim_phrase"] = df["claim_phrase"].apply(normalize_text)
    df = df[(df["claim_phrase"] != "") & (df["claim_phrase"].str.upper() != "N/A")].copy()

    df["verdict"] = df["verdict"].apply(lambda x: normalize_text(x).upper())
    df["verdict"] = df["verdict"].apply(lambda v: v if v in VALID_VERDICT_SET else "UNAVAILABLE")
    df["annotator"] = annotator_label

    print(f"  {annotator_label}: {len(df)} claims from {df['annot_id'].nunique()} explanations")
    return df


def token_overlap(a, b):
    """Phrase similarity for claim alignment.

    The original script used only SequenceMatcher on token lists. That is too
    strict when one annotator writes a short phrase (e.g., "high SNR") and
    another/source extractor writes a containing phrase (e.g.,
    "high signal-to-noise ratio (SNR)"). This function keeps SequenceMatcher
    but also uses token containment, which is more appropriate for phrase-level
    claim matching.
    """
    import re

    def toks(x):
        return [
            t for t in re.findall(r"[a-z0-9]+", str(x).lower())
            if t not in STOPWORDS
        ]

    ta = toks(a)
    tb = toks(b)
    if not ta or not tb:
        return 0.0

    sa, sb = set(ta), set(tb)
    intersection = sa & sb
    containment = max(len(intersection) / len(sa), len(intersection) / len(sb))
    jaccard = len(intersection) / len(sa | sb)
    sequence = SequenceMatcher(None, ta, tb).ratio()
    return max(containment, jaccard, sequence)


def align_annotations(df_a, df_b):
    """Match claims between annotators within each explanation by phrase similarity."""
    pairs = []
    matched_rows = []

    for annot_id in sorted(set(df_a["annot_id"]).union(set(df_b["annot_id"]))):
        a_rows = df_a[df_a["annot_id"] == annot_id].reset_index(drop=True)
        b_rows = df_b[df_b["annot_id"] == annot_id].reset_index(drop=True)
        used_b = set()

        for i, arow in a_rows.iterrows():
            best_j, best_sim = -1, 0.0
            for j, brow in b_rows.iterrows():
                if j in used_b:
                    continue
                sim = token_overlap(arow["claim_phrase"], brow["claim_phrase"])
                if sim > best_sim:
                    best_sim, best_j = sim, j

            if best_j >= 0 and best_sim >= MATCH_THRESHOLD:
                brow = b_rows.loc[best_j]
                pairs.append((arow["verdict"], brow["verdict"]))
                matched_rows.append({
                    "annot_id": annot_id,
                    "claim_A": arow["claim_phrase"],
                    "verdict_A": arow["verdict"],
                    "claim_B": brow["claim_phrase"],
                    "verdict_B": brow["verdict"],
                    "similarity": round(best_sim, 3),
                })
                used_b.add(best_j)

    return pairs, pd.DataFrame(matched_rows)


def cohen_kappa(pairs):
    if not pairs:
        return 0.0, 0
    try:
        from sklearn.metrics import cohen_kappa_score
        ra, rb = zip(*pairs)
        return round(float(cohen_kappa_score(ra, rb, labels=VALID_VERDICTS)), 3), len(pairs)
    except Exception:
        # Small fallback implementation if sklearn is unavailable.
        ra, rb = zip(*pairs)
        n = len(pairs)
        observed = sum(1 for a, b in pairs if a == b) / n
        pa = {v: ra.count(v) / n for v in VALID_VERDICTS}
        pb = {v: rb.count(v) / n for v in VALID_VERDICTS}
        expected = sum(pa[v] * pb[v] for v in VALID_VERDICTS)
        kappa = (observed - expected) / (1 - expected) if expected != 1 else 0.0
        return round(kappa, 3), n


def load_claim_extractor():
    try:
        from safe_gem.claim_extractor import ClaimExtractor
        return ClaimExtractor
    except Exception:
        from claim_extractor import ClaimExtractor
        return ClaimExtractor


def compare_auto_vs_human(samples_csv, df_a, df_b):
    """Compare automated extracted claim phrases against human union annotations."""
    ClaimExtractor = load_claim_extractor()
    samples_df = pd.read_csv(samples_csv)
    extractor = ClaimExtractor()

    TP = FP = FN = 0
    details = []

    for _, row in samples_df.iterrows():
        annot_id = str(row["annot_id"])
        expl = str(row["explanation_text"])

        human_a = set(df_a[df_a["annot_id"] == annot_id]["claim_phrase"].str.lower().tolist())
        human_b = set(df_b[df_b["annot_id"] == annot_id]["claim_phrase"].str.lower().tolist())
        human_claims = human_a | human_b

        try:
            auto_claims = set(c.raw_phrase.lower() for c in extractor.extract_claims(expl))
        except Exception as e:
            print(f"  WARNING: extractor failed for {annot_id}: {e}")
            auto_claims = set()

        local_tp = local_fp = local_fn = 0
        for hc in human_claims:
            matched = any(token_overlap(hc, ac) >= MATCH_THRESHOLD for ac in auto_claims)
            if matched:
                TP += 1
                local_tp += 1
            else:
                FN += 1
                local_fn += 1

        for ac in auto_claims:
            matched = any(token_overlap(ac, hc) >= MATCH_THRESHOLD for hc in human_claims)
            if not matched:
                FP += 1
                local_fp += 1

        details.append({
            "annot_id": annot_id,
            "strategy": row.get("strategy", ""),
            "human_claims_union": len(human_claims),
            "auto_claims": len(auto_claims),
            "TP": local_tp,
            "FP": local_fp,
            "FN": local_fn,
        })

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "details": pd.DataFrame(details),
    }


def save_results(kappa, n_pairs, auto_metrics, df_a, df_b, matched_df, out_dir):
    out_dir = Path(out_dir)
    out_path = out_dir / "annotation_agreement_results_fair_match.xlsx"

    summary_df = pd.DataFrame([
        ["Inter-annotator kappa (κ)", kappa, "≥ 0.70", "YES" if kappa >= 0.70 else "NO — review/re-annotate"],
        ["Matched pairs for kappa", n_pairs, "≥ 20", "YES" if n_pairs >= 20 else "LOW"],
        ["Auto extractor Precision", auto_metrics["precision"], "≥ 0.70", ""],
        ["Auto extractor Recall", auto_metrics["recall"], "≥ 0.65", ""],
        ["Auto extractor F1", auto_metrics["f1"], "≥ 0.70", "YES" if auto_metrics["f1"] >= 0.70 else "BELOW TARGET"],
        ["True Positives", auto_metrics["TP"], "", ""],
        ["False Positives", auto_metrics["FP"], "", ""],
        ["False Negatives", auto_metrics["FN"], "", ""],
        ["Total A claims", len(df_a), "", ""],
        ["Total B claims", len(df_b), "", ""],
    ], columns=["Metric", "Value", "Threshold", "Pass?"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Agreement Results", index=False)
        matched_df.to_excel(writer, sheet_name="Matched human claims", index=False)
        auto_metrics["details"].to_excel(writer, sheet_name="Auto vs human by sample", index=False)
        df_a.to_excel(writer, sheet_name="Annotator A cleaned", index=False)
        df_b.to_excel(writer, sheet_name="Annotator B cleaned", index=False)

    print(f"\nResults saved: {out_path}")


def main():
    print("Loading completed annotation sheets...")
    for path in [SHEET_A, SHEET_B, SAMPLES]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected file: {path}")

    df_a = load_sheet(SHEET_A, "A")
    df_b = load_sheet(SHEET_B, "B")

    print("\nComputing inter-annotator agreement...")
    pairs, matched_df = align_annotations(df_a, df_b)
    kappa, n_pairs = cohen_kappa(pairs)
    print(f"  Cohen's kappa: {kappa} ({n_pairs} matched pairs)")

    print("\nComputing automated extractor precision/recall...")
    auto_metrics = compare_auto_vs_human(SAMPLES, df_a, df_b)
    print(f"  Precision: {auto_metrics['precision']}")
    print(f"  Recall:    {auto_metrics['recall']}")
    print(f"  F1:        {auto_metrics['f1']}")
    print(f"  TP={auto_metrics['TP']} FP={auto_metrics['FP']} FN={auto_metrics['FN']}")

    save_results(kappa, n_pairs, auto_metrics, df_a, df_b, matched_df, RESULTS_DIR)

    print("\n=== Summary for paper Section 6.4 ===")
    print(f"  Annotator agreement: κ = {kappa}")
    print(
        f"  Automated extractor: P={auto_metrics['precision']} "
        f"R={auto_metrics['recall']} F1={auto_metrics['f1']}"
    )


if __name__ == "__main__":
    main()
