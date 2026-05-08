"""
test_prompts.py
----------------
Tests each prompting strategy with a single call and shows the raw response.
Diagnoses why P2/P4/P5 return empty text.
Also applies a fix to generate_explanation.py.

Run from safe_gem folder:
    python test_prompts.py
"""

import sys, os, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OLLAMA_URL = "http://localhost:11434"
MODEL      = "gemma4:e4b"

# ── Simple test call ──────────────────────────────────────────────────────────

def call(system, user, label):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }, timeout=120)

    content = r.json().get("message", {}).get("content", "").strip()
    print(f"\n{'─'*60}")
    print(f"Strategy: {label}")
    print(f"System: {str(system)[:80] if system else 'None'}")
    print(f"Response length: {len(content)} chars")
    print(f"Response: {content[:300] if content else '*** EMPTY ***'}")
    return content


# ── Test prompts ──────────────────────────────────────────────────────────────

USER_BASE = """Prediction : MICROSEISMIC EVENT
Confidence : 1.00

Signal features:
  - SNR: high SNR
  - Dominant frequency: high dominant frequency
  - High-frequency energy: strong high-frequency energy
  - Onset character: moderate onset
  - DAS coherence: incoherent
  - Event support: strong geophone/model support

Explain the prediction in 3-4 sentences."""

print("Testing all 5 strategies with a sample event prediction...")
print(f"Model: {MODEL}\n")

# P1 raw
r1 = call(None, USER_BASE.replace("Signal features:\n  - SNR: high SNR\n  - Dominant frequency: high dominant frequency\n  - High-frequency energy: strong high-frequency energy\n  - Onset character: moderate onset\n  - DAS coherence: incoherent\n  - Event support: strong geophone/model support\n\n", ""), "P1_raw")

# P2 grounded
r2 = call(None, USER_BASE, "P2_grounded")

# P3 uncertainty
r3 = call(
    "You are a cautious geophysicist. Use hedging language: 'possibly', 'may indicate', 'appears to'. Mention at least one alternative interpretation.",
    USER_BASE,
    "P3_uncertainty"
)

# P4 constrained — ORIGINAL (likely causing empty)
r4_orig = call(
    "You are a geophysicist. STRICT RULE: You may only describe signal characteristics that are explicitly listed in the provided feature block. Do not mention any feature not in the list. Do not invent evidence.",
    USER_BASE,
    "P4_constrained (original — strict)"
)

# P4 constrained — RELAXED FIX
r4_fix = call(
    "You are a geophysicist explaining a seismic detection. Base your explanation on the provided signal features. Prefer the listed features but you may use general seismological context to explain them.",
    USER_BASE,
    "P4_constrained (relaxed fix)"
)

# P5 safe_gem — ORIGINAL
r5_orig = call(
    "You are a cautious geophysicist. Rules: 1. Base explanation on provided features. 2. Use hedging language. 3. Mention sensor agreement if relevant. 4. Separate model confidence from signal evidence. 5. Acknowledge limitations. 6. Do not claim certainty.",
    USER_BASE,
    "P5_safe_gem (original)"
)

print(f"\n{'='*60}")
print("DIAGNOSIS:")
print(f"  P1_raw          : {'OK' if r1 else 'EMPTY'}")
print(f"  P2_grounded     : {'OK' if r2 else 'EMPTY'}")
print(f"  P3_uncertainty  : {'OK' if r3 else 'EMPTY'}")
print(f"  P4 original     : {'OK' if r4_orig else 'EMPTY — fix needed'}")
print(f"  P4 relaxed fix  : {'OK' if r4_fix else 'EMPTY — still failing'}")
print(f"  P5 original     : {'OK' if r5_orig else 'EMPTY — fix needed'}")

# ── Apply fix to generate_explanation.py if needed ────────────────────────────

if not r4_orig and r4_fix:
    print("\nApplying fix to generate_explanation.py...")

    gen_path = "generate_explanation.py"
    with open(gen_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace P4 system prompt
    old_p4 = (
        '"You are a geophysicist writing a signal explanation. "\n'
        '        "STRICT RULE: You may only describe signal characteristics that are "\n'
        '        "explicitly listed in the provided feature block. "\n'
        '        "Do not mention P-waves, S-waves, moveout, coherence, SNR, frequency, "\n'
        '        "amplitude, or any other feature unless it appears in the feature list. "\n'
        '        "If you are uncertain, say so. Do not invent evidence."'
    )
    new_p4 = (
        '"You are a geophysicist explaining a seismic detection result. "\n'
        '        "Base your explanation primarily on the provided signal features. "\n'
        '        "You may use general seismological context to explain the features, "\n'
        '        "but prefer the listed evidence. If a feature is not listed, "\n'
        '        "acknowledge uncertainty rather than inventing evidence."'
    )

    # Replace P5 system prompt
    old_p5 = (
        '"You are a cautious geophysicist writing a pre-deployment explanation "\n'
        '        "for an AI microseismic detection system. "\n'
        '        "Follow all of these rules:\\n"\n'
        '        "1. Only use evidence explicitly listed in the feature block.\\n"\n'
        '        "2. Use hedging language: \'appears to\', \'suggests\', \'may indicate\'.\\n"\n'
        '        "3. Explicitly mention any sensor disagreement if present.\\n"\n'
        '        "4. Separate model confidence from signal evidence strength.\\n"\n'
        '        "5. End with one sentence acknowledging uncertainty or limitations.\\n"\n'
        '        "6. Do not invent features. Do not claim certainty."'
    )
    new_p5 = (
        '"You are a cautious geophysicist writing a careful explanation "\n'
        '        "for an AI microseismic detection result. "\n'
        '        "Use hedging language such as \'appears to\', \'suggests\', \'may indicate\'. "\n'
        '        "Base your explanation on the provided signal features. "\n'
        '        "Mention if sensors disagree. "\n'
        '        "End with one sentence acknowledging the limits of the evidence."'
    )

    changed = 0
    if old_p4 in content:
        content = content.replace(old_p4, new_p4)
        changed += 1
        print("  Fixed P4 system prompt")
    else:
        print("  P4 prompt not found by exact match — manual fix needed")

    if old_p5 in content:
        content = content.replace(old_p5, new_p5)
        changed += 1
        print("  Fixed P5 system prompt")
    else:
        print("  P5 prompt not found by exact match — manual fix needed")

    if changed > 0:
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved {changed} fix(es) to {gen_path}")
    else:
        print("\n  Auto-fix failed. Use the relaxed prompts shown above manually.")

print("\nNow re-run: python pilot_30samples.py")
