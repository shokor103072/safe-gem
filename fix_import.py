"""
fix_import.py
--------------
Diagnoses and fixes the safe_gem import issue.
Run from the same folder as pilot_30samples.py:
    python fix_import.py
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "safe_gem")

print(f"Script location : {HERE}")
print(f"Package location: {PACKAGE_DIR}")
print()

# ── Step 1: Diagnose ──────────────────────────────────────────────────────────
print("=== DIAGNOSIS ===")

init_path = os.path.join(PACKAGE_DIR, "__init__.py")
if not os.path.exists(PACKAGE_DIR):
    print(f"ERROR: safe_gem/ folder not found at {PACKAGE_DIR}")
    print("Please copy the safe_gem/ folder here.")
    sys.exit(1)

files = os.listdir(PACKAGE_DIR)
print(f"Files in safe_gem/: {files}")

if "__init__.py" not in files:
    print("PROBLEM: __init__.py is MISSING from safe_gem/")
else:
    with open(init_path, "r", encoding="utf-8") as f:
        init_content = f.read()
    if "SafeGEMPipeline" not in init_content:
        print("PROBLEM: __init__.py exists but does not export SafeGEMPipeline")
        print(f"Current content:\n{init_content[:300]}")
    else:
        print("__init__.py looks correct")

# ── Step 2: Write correct __init__.py ────────────────────────────────────────
print()
print("=== APPLYING FIX ===")

CORRECT_INIT = '''"""SAFE-GEM: Safety-aware GenAI explanation auditing for microseismic monitoring."""

from .pipeline import SafeGEMPipeline, SafeGEMBatch, SampleResult, BatchSummary
from .feature_extraction import SignalFeatures, MultimodalFeatures
from .symbolic_converter import ThresholdConfig, SymbolicEvidence, SymbolicConverter
from .claim_extractor import ClaimExtractor, ExtractedClaim, ClaimCheckResult, GroundingReport
from .eirrs_scorer import EIRRSScorer, EIRRSResult, WEIGHT_SCHEMES

__all__ = [
    "SafeGEMPipeline",
    "SafeGEMBatch",
    "SampleResult",
    "BatchSummary",
    "SignalFeatures",
    "MultimodalFeatures",
    "ThresholdConfig",
    "SymbolicEvidence",
    "SymbolicConverter",
    "ClaimExtractor",
    "ExtractedClaim",
    "ClaimCheckResult",
    "GroundingReport",
    "EIRRSScorer",
    "EIRRSResult",
    "WEIGHT_SCHEMES",
]
'''

with open(init_path, "w", encoding="utf-8") as f:
    f.write(CORRECT_INIT)
print(f"Wrote correct __init__.py to {init_path}")

# ── Step 3: Fix sys.path in pilot_30samples.py ────────────────────────────────
PILOT_FILE = os.path.join(HERE, "pilot_30samples.py")

if os.path.exists(PILOT_FILE):
    with open(PILOT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    PATH_FIX = f'''import sys as _sys
import os as _os
_sys.path.insert(0, r"{HERE}")
'''

    if r'_sys.path.insert' not in content and 'sys.path.insert' not in content:
        # Insert right after the first import line
        content = PATH_FIX + content
        with open(PILOT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print("Added sys.path fix to pilot_30samples.py")
    else:
        # Replace existing path fix with absolute path version
        import re
        content = re.sub(
            r'sys\.path\.insert\(0,.*?\)',
            f'sys.path.insert(0, r"{HERE}")',
            content
        )
        with open(PILOT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print("Updated sys.path in pilot_30samples.py with absolute path")

# ── Step 4: Verify the import works ──────────────────────────────────────────
print()
print("=== VERIFICATION ===")

sys.path.insert(0, HERE)

# Clear any cached broken safe_gem
for key in list(sys.modules.keys()):
    if "safe_gem" in key:
        del sys.modules[key]

try:
    from safe_gem import SafeGEMPipeline, SafeGEMBatch
    import safe_gem
    print(f"Import SUCCESS: safe_gem loaded from {safe_gem.__file__}")
    print(f"SafeGEMPipeline: {SafeGEMPipeline}")
    print()
    print("All fixes applied. Run: python pilot_30samples.py")
except ImportError as e:
    print(f"Import still failing: {e}")
    print()
    print("Required files in safe_gem/:")
    required = ["__init__.py", "pipeline.py", "feature_extraction.py",
                "symbolic_converter.py", "claim_extractor.py", "eirrs_scorer.py"]
    for f in required:
        path = os.path.join(PACKAGE_DIR, f)
        status = "FOUND" if os.path.exists(path) else "MISSING"
        print(f"  {status}: {f}")
