"""
check_datasets.py
------------------
Inspect and compare X.npy (Stage 14) vs X_forge.npy (FORGE subset).
Helps decide which dataset to use for the SAFE-GEM pilot.

Run:
    python check_datasets.py
"""

import numpy as np
import os

DATA_DIR = r"C:\Users\Valued User\Desktop\Office PhD\agentic_egs\agentic_egs\agentic_egs\Dataset"

# ─── Load both datasets ───────────────────────────────────────────────────────

def load_dataset(name):
    x_path = os.path.join(DATA_DIR, f"{name}.npy")
    y_path = os.path.join(DATA_DIR, f"{'y' if name == 'X' else 'y_forge'}.npy")

    print(f"\n{'='*50}")
    print(f"  {name}.npy")
    print(f"{'='*50}")

    if not os.path.exists(x_path):
        print(f"  NOT FOUND: {x_path}")
        return None, None

    X = np.load(x_path, mmap_mode='r')   # memory-map for large files
    y = np.load(y_path)

    n_events = int((y == 1).sum())
    n_noise  = int((y == 0).sum())
    n_total  = len(y)

    print(f"  Shape        : {X.shape}")
    print(f"  dtype        : {X.dtype}")
    print(f"  File size    : {os.path.getsize(x_path) / 1e9:.2f} GB")
    print(f"  Total samples: {n_total}")
    print(f"  Events       : {n_events} ({n_events/n_total*100:.1f}%)")
    print(f"  Noise        : {n_noise}  ({n_noise/n_total*100:.1f}%)")
    print(f"  Balance      : {'balanced' if abs(n_events - n_noise) / n_total < 0.1 else 'imbalanced'}")

    # Infer shape meaning
    if X.ndim == 4:
        ns, nm, nc, nt = X.shape
        print(f"\n  Dimensions   : {ns} samples x {nm} modality x {nc} channels x {nt} timepoints")
        print(f"  Channels     : {nc} DAS channels")
        print(f"  Time points  : {nt} samples")
    elif X.ndim == 3:
        ns, nc, nt = X.shape
        print(f"\n  Dimensions   : {ns} samples x {nc} channels x {nt} timepoints")
    elif X.ndim == 2:
        ns, nt = X.shape
        nc = 1
        print(f"\n  Dimensions   : {ns} samples x {nt} timepoints (single channel)")

    # Signal quality check on first 20 samples
    print(f"\n  Signal quality (first 20 samples):")
    snrs = []
    for i in range(min(20, n_total)):
        if X.ndim == 4:
            trace = X[i, 0].mean(axis=0)   # mean across channels
        elif X.ndim == 3:
            trace = X[i].mean(axis=0)
        else:
            trace = X[i]

        pre  = len(trace) // 4
        noise_rms = np.sqrt(np.mean(trace[:pre]**2)) + 1e-10
        event_rms = np.sqrt(np.mean(trace[pre:]**2))
        snr = 20 * np.log10(event_rms / noise_rms)
        snrs.append(snr)

    print(f"  Mean SNR     : {np.mean(snrs):.1f} dB")
    print(f"  Median SNR   : {np.median(snrs):.1f} dB")
    print(f"  Min SNR      : {np.min(snrs):.1f} dB")
    print(f"  Max SNR      : {np.max(snrs):.1f} dB")

    # Value range
    sample = X[0, 0] if X.ndim == 4 else X[0]
    print(f"\n  Value range  : [{float(sample.min()):.4f}, {float(sample.max()):.4f}]")
    print(f"  Normalised   : {'yes' if abs(float(sample.max())) <= 1.05 else 'no (raw counts)'}")

    return X, y


# ─── Comparison summary ────────────────────────────────────────────────────────

def compare(X1, y1, name1, X2, y2, name2):
    print(f"\n{'='*50}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*50}")

    rows = [
        ("Samples",    len(y1),                  len(y2)),
        ("Events",     int((y1==1).sum()),        int((y2==1).sum())),
        ("Noise",      int((y1==0).sum()),        int((y2==0).sum())),
        ("Channels",   X1.shape[-2] if X1.ndim==4 else X1.shape[-2] if X1.ndim==3 else 1,
                       X2.shape[-2] if X2.ndim==4 else X2.shape[-2] if X2.ndim==3 else 1),
        ("Timepoints", X1.shape[-1],              X2.shape[-1]),
        ("File size",  f"{os.path.getsize(os.path.join(DATA_DIR,'X.npy'))/1e9:.1f} GB",
                       f"{os.path.getsize(os.path.join(DATA_DIR,'X_forge.npy'))/1e9:.1f} GB"),
    ]

    print(f"  {'Metric':<14} {name1:>12} {name2:>12}")
    print(f"  {'-'*40}")
    for label, v1, v2 in rows:
        print(f"  {label:<14} {str(v1):>12} {str(v2):>12}")

    print(f"\n  RECOMMENDATION:")
    n1, n2 = len(y1), len(y2)
    if n1 > 500 and n2 <= 500:
        print(f"  Use {name1} — larger dataset ({n1} vs {n2} samples)")
        print(f"  {name2} may be too small for robust pilot statistics.")
    elif n2 > n1:
        print(f"  Use {name2} — larger dataset.")
    else:
        print(f"  Both are usable. {name1} has more samples.")

    print(f"\n  For the SAFE-GEM pilot (30 samples), either works.")
    print(f"  For the full experiment (80 samples), use the larger one.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Checking datasets...")
    print(f"Data directory: {DATA_DIR}\n")

    X1, y1 = load_dataset("X")
    X2, y2 = load_dataset("X_forge")

    if X1 is not None and X2 is not None:
        compare(X1, y1, "X (Stage14)", X2, y2, "X_forge")

    print("\nDone. Update DATA_DIR in pilot_30samples.py to use your chosen dataset.")
