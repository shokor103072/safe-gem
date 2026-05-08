"""
fix_pilot.py
-------------
Run this once to patch pilot_30samples.py.
Place in same folder as pilot_30samples.py and run:
    python fix_pilot.py
"""
import os

PILOT_FILE = "pilot_30samples.py"

with open(PILOT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ── Fix 1: add sys.path so safe_gem package is found ─────────────────────────
if "sys.path.insert(0" not in content:
    content = content.replace(
        "from collections import Counter",
        "from collections import Counter\n\nimport sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
    )
    changes += 1
    print("Fix 1 applied: sys.path fix added")
else:
    print("Fix 1 skipped: sys.path already present")

# ── Fix 2: label-based sample selection (robust to low-SNR STA/LTA) ──────────
OLD_SELECTION = """\
    # Clear events: correct prediction, high confidence
    clear_event_idx = indices[correct_mask & event_mask]
    clear_event_conf = confidences[correct_mask & event_mask]
    clear_event_sorted = clear_event_idx[np.argsort(clear_event_conf)[::-1]]
    clear_events = clear_event_sorted[:N_CLEAR_EVENTS].tolist()

    # Clear non-events: correct prediction, high confidence
    clear_noise_idx  = indices[correct_mask & nonevent_mask]
    clear_noise_conf = confidences[correct_mask & nonevent_mask]
    clear_noise_sorted = clear_noise_idx[np.argsort(clear_noise_conf)[::-1]]
    clear_noises = clear_noise_sorted[:N_CLEAR_NOISE].tolist()"""

NEW_SELECTION = """\
    # Select by ground truth label + highest confidence
    # (robust when STA/LTA accuracy is low on noisy data)
    event_idx    = indices[event_mask]
    event_conf   = confidences[event_mask]
    clear_events = event_idx[np.argsort(event_conf)[::-1]][:N_CLEAR_EVENTS].tolist()

    noise_idx    = indices[nonevent_mask]
    noise_conf   = confidences[nonevent_mask]
    clear_noises = noise_idx[np.argsort(noise_conf)[::-1]][:N_CLEAR_NOISE].tolist()"""

if OLD_SELECTION in content:
    content = content.replace(OLD_SELECTION, NEW_SELECTION)
    changes += 1
    print("Fix 2 applied: label-based sample selection")
else:
    print("Fix 2 skipped: selection already updated")

with open(PILOT_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n{changes} fix(es) applied to {PILOT_FILE}")
print("Now run: python pilot_30samples.py")
