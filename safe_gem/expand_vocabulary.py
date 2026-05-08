"""
expand_vocabulary.py
---------------------
Expands the controlled vocabulary in safe_gem/claim_extractor.py
to match Gemma 4's actual explanation language patterns.

Based on pilot results showing only 0-2 claims extracted per explanation.
Gemma 4 uses phrases like "spectral characteristics", "poor signal coherence",
"unfavorable spectral" that the original vocabulary did not cover.

Run from your safe_gem/ folder:
    python expand_vocabulary.py
"""
import os
import re

EXTRACTOR_FILE = os.path.join("safe_gem", "claim_extractor.py")

with open(EXTRACTOR_FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ── Helper ────────────────────────────────────────────────────────────────────
def add_phrases(content, group, polarity, new_phrases):
    """Insert new phrases into an existing vocabulary group/polarity list."""
    # Find the list for this group and polarity
    pattern = rf'("{group}".*?"{polarity}":\s*\[)(.*?)(\])'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  WARNING: could not find {group}/{polarity}")
        return content, 0

    existing_block = match.group(2)
    # Extract already-present phrases to avoid duplicates
    existing = re.findall(r'"([^"]+)"', existing_block)
    to_add = [p for p in new_phrases if p not in existing]
    if not to_add:
        print(f"  SKIP {group}/{polarity}: all phrases already present")
        return content, 0

    addition = ",\n            " + ",\n            ".join(f'"{p}"' for p in to_add)
    new_block = match.group(1) + match.group(2).rstrip() + addition + "\n        " + match.group(3)
    content = content[:match.start()] + new_block + content[match.end():]
    print(f"  ADDED {len(to_add)} phrases to {group}/{polarity}: {to_add}")
    return content, len(to_add)


# ── SNR — Gemma 4 uses "signal-to-noise ratio" without "high/low" prefix ─────
content, n = add_phrases(content, "snr", "positive", [
    "signal-to-noise ratio is high",
    "favorable signal-to-noise",
    "good signal-to-noise",
    "adequate snr",
    "sufficient snr",
    "strong signal relative to noise",
])
changes += n

content, n = add_phrases(content, "snr", "negative", [
    "signal-to-noise ratio",          # catches "poor signal-to-noise ratio"
    "poor signal-to-noise",
    "unfavorable signal-to-noise",
    "low signal-to-noise ratio",
    "insufficient snr",
    "elevated noise",
    "noise-contaminated",
    "noise floor",
    "buried in noise",
])
changes += n

# ── Amplitude — Gemma 4 uses "weak signal", "amplitude response" ─────────────
content, n = add_phrases(content, "amplitude", "positive", [
    "strong signal",
    "high amplitude signal",
    "elevated amplitude",
    "clear amplitude",
    "prominent amplitude",
    "distinct amplitude",
])
changes += n

content, n = add_phrases(content, "amplitude", "negative", [
    "weak signal",
    "low amplitude signal",
    "poor amplitude",
    "weak amplitude response",
    "low energy signal",
    "insufficient amplitude",
    "subdued amplitude",
])
changes += n

# ── Frequency — Gemma 4 uses "spectral characteristics", "energy distribution" 
content, n = add_phrases(content, "frequency", "positive", [
    "spectral characteristics consistent with",
    "high-frequency content",
    "frequency content",
    "high frequency energy",
    "rich frequency content",
    "favorable spectral",
    "spectral energy",
    "energy distribution consistent",
    "frequency signature",
    "high-frequency signature",
])
changes += n

content, n = add_phrases(content, "frequency", "negative", [
    "unfavorable spectral characteristics",
    "poor spectral characteristics",
    "spectral characteristics inconsistent",
    "low frequency dominated",
    "low-frequency characteristics",
    "frequency characteristics of noise",
    "lacks high-frequency",
    "absence of high-frequency",
    "low spectral energy",
])
changes += n

# ── Onset — Gemma 4 uses "lacks impulsive", "no clear onset" ─────────────────
content, n = add_phrases(content, "onset", "positive", [
    "impulsive signal",
    "impulsive character",
    "clear impulsive",
    "distinct onset",
    "well-defined onset",
    "rapid onset",
    "abrupt increase",
])
changes += n

content, n = add_phrases(content, "onset", "negative", [
    "lacks impulsive",
    "no impulsive",
    "lacks the sharp",
    "no clear onset",
    "no distinct onset",
    "gradual increase",
    "no abrupt",
    "slow rise",
    "emergent signal",
])
changes += n

# ── DAS coherence — Gemma 4 uses "poor coherence", "lack of coherence" ────────
content, n = add_phrases(content, "das_coherence", "positive", [
    "coherent signal",
    "good coherence",
    "high coherence",
    "coherent across",
    "strong coherence",
    "spatially consistent",
])
changes += n

content, n = add_phrases(content, "das_coherence", "negative", [
    "poor coherence",
    "poor signal coherence",
    "lack of coherence",
    "low coherence",
    "lacks coherence",
    "incoherent signal",
    "absence of coherence",
    "inconsistent signal",
])
changes += n

# ── Geophone — Gemma 4 may use "sensor data", "independent confirmation" ──────
content, n = add_phrases(content, "geophone_support", "positive", [
    "independent confirmation",
    "sensor confirms",
    "corroborated by",
    "supported by sensor",
    "sensor data supports",
])
changes += n

content, n = add_phrases(content, "geophone_support", "negative", [
    "no independent confirmation",
    "sensor does not confirm",
    "not corroborated",
    "sensor data does not support",
    "no sensor confirmation",
])
changes += n

# ── Event claim — Gemma 4 variations ─────────────────────────────────────────
content, n = add_phrases(content, "event_claim", "positive", [
    "seismic origin",
    "seismic source",
    "tectonic event",
    "earthquake signal",
    "genuine event",
    "real event",
])
changes += n

content, n = add_phrases(content, "event_claim", "negative", [
    "not a seismic event",
    "not microseismic",
    "likely noise",
    "classified as noise",
    "noise classification",
    "non-seismic",
    "instrumental noise",
    "anthropogenic noise",
])
changes += n

# ── Uncertainty — Gemma 4 uses "it is possible", "it appears" ────────────────
content, n = add_phrases(content, "uncertainty", "positive", [
    "it is possible",
    "it appears",
    "it seems",
    "may be",
    "could indicate",
    "potential",
    "probable",
    "not certain",
    "further analysis",
    "further review",
    "warrants review",
])
changes += n

content, n = add_phrases(content, "uncertainty", "negative", [
    "it is clear",
    "it is evident",
    "it is definitive",
    "unquestionably",
    "there is no doubt",
    "it is obvious",
])
changes += n

# ── Write updated file ────────────────────────────────────────────────────────
with open(EXTRACTOR_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal: {changes} phrases added to {EXTRACTOR_FILE}")
print("\nNext steps:")
print("1. Save URDI predictions (run extract_urdi_predictions.py)")
print("2. Re-run pilot: python pilot_30samples.py")
