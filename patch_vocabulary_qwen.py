"""
patch_vocabulary_qwen.py
-------------------------
Final vocabulary patch based on actual Qwen 2.5 explanation texts
from the mini pilot (explanations_full.txt analysis).

Run from your safe_gem folder:
    python patch_vocabulary_qwen.py
"""

import os
import re
import ast

EXTRACTOR = os.path.join("safe_gem", "claim_extractor.py")

with open(EXTRACTOR, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0


def add_phrases(content, group, polarity, new_phrases):
    pattern = rf'("{group}".*?"{polarity}":\s*\[)(.*?)(\n        \])'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  WARNING: {group}/{polarity} not found")
        return content, 0
    existing = re.findall(r'"([^"]+)"', match.group(2))
    to_add = [p for p in new_phrases if p not in existing]
    if not to_add:
        return content, 0
    block = match.group(2).rstrip().rstrip(',')
    addition = ',\n            ' + ',\n            '.join(f'"{p}"' for p in to_add) + ','
    new_block = match.group(1) + block + addition + match.group(3)
    content = content[:match.start()] + new_block + content[match.end():]
    print(f"  ADDED to {group}/{polarity}: {to_add}")
    return content, len(to_add)


# ── Frequency — Qwen uses "low-frequency vibrations", "frequency patterns" ───

content, n = add_phrases(content, "frequency", "negative", [
    "low-frequency vibrations",
    "low frequency vibrations",
    "low-frequency signal",
    "low frequency characteristics",
    "low frequency dominated signal",
    "dominated by low frequency",
])
changes += n

content, n = add_phrases(content, "frequency", "positive", [
    "unique frequency patterns",
    "specific frequency patterns",
    "frequency patterns",
    "frequency signature",
    "spectral signature",
    "distinct frequency",
    "rich frequency",
])
changes += n

# ── Amplitude — Qwen uses "low energy and random fluctuations" ───────────────

content, n = add_phrases(content, "amplitude", "negative", [
    "low energy",
    "random fluctuations",
    "low amplitude fluctuations",
    "weak amplitude signal",
    "low energy signal",
    "minimal energy",
])
changes += n

content, n = add_phrases(content, "amplitude", "positive", [
    "strong energy burst",
    "high energy signal",
    "significant energy burst",
    "strong signal energy",
])
changes += n

# ── Onset / temporal — Qwen uses "temporal occurrences", "temporal patterns" ─

content, n = add_phrases(content, "onset", "positive", [
    "temporal occurrences",
    "temporal patterns",
    "distinct temporal",
    "transient signal",
    "transient event",
    "clear temporal",
    "impulsive waveform",
])
changes += n

content, n = add_phrases(content, "onset", "negative", [
    "random temporal",
    "irregular temporal",
    "no distinct temporal",
    "diffuse temporal",
])
changes += n

# ── DAS coherence — Qwen uses "waveforms", "consistent waveforms" ─────────────

content, n = add_phrases(content, "das_coherence", "positive", [
    "coherent waveforms",
    "consistent waveforms",
    "coherent signal pattern",
    "spatially coherent waveform",
    "coherent seismic waveform",
    "waveform consistency",
])
changes += n

content, n = add_phrases(content, "das_coherence", "negative", [
    "incoherent waveform",
    "random waveforms",
    "inconsistent waveforms",
    "waveform variability",
    "waveform incoherence",
])
changes += n

# ── SNR — Qwen uses "signal quality", "data quality" ─────────────────────────

content, n = add_phrases(content, "snr", "positive", [
    "signal quality is good",
    "good signal quality",
    "high data quality",
    "clear signal quality",
    "strong signal quality",
    "excellent snr",
    "signal is clear and distinct",
])
changes += n

content, n = add_phrases(content, "snr", "negative", [
    "poor signal quality",
    "low data quality",
    "degraded signal quality",
    "signal quality is poor",
])
changes += n

# ── Event claim — Qwen uses "seismic signature", "does not meet criteria" ─────

content, n = add_phrases(content, "event_claim", "positive", [
    "seismic signature",
    "microseismic signature",
    "seismic activity",
    "microseismic activity",
    "seismic occurrence",
    "seismic source signature",
    "genuine seismic",
    "true seismic event",
])
changes += n

content, n = add_phrases(content, "event_claim", "negative", [
    "does not meet the criteria",
    "does not represent an event",
    "not consistent with an event",
    "classified as noise",
    "identified as noise",
    "noise or non-event",
    "no event detected",
    "not indicative of an event",
])
changes += n

# ── Geophone — Qwen uses "sensor evidence", "sensor support" ─────────────────

content, n = add_phrases(content, "geophone_support", "positive", [
    "sensor evidence supports",
    "sensor support",
    "sensor data confirms",
    "supported by sensor data",
    "corroborated by sensor",
])
changes += n

# ── Verify and write ──────────────────────────────────────────────────────────

if ',,' in content:
    print("\nERROR: double comma found — check file manually")
else:
    try:
        ast.parse(content)
        with open(EXTRACTOR, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nTotal: {changes} phrases added")
        print(f"Syntax check: OK")
        print(f"Written to: {EXTRACTOR}")
        print("\nNow run: python pilot_30samples.py")
    except SyntaxError as e:
        print(f"\nSYNTAX ERROR line {e.lineno}: {e.msg}")
