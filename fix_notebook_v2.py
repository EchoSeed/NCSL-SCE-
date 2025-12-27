#!/usr/bin/env python3
import json

# Read the notebook
with open('EchoSeed_v4.2_fixed.ipynb', 'r') as f:
    nb = json.load(f)

# Find cells
json_patch_idx = None
imports_idx = None

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    if '_np_default' in source and 'json.dump' in source:
        json_patch_idx = i
    if 'import matplotlib.pyplot as plt' in source:
        imports_idx = i

# Move the json patch cell to AFTER the imports cell
if json_patch_idx is not None and imports_idx is not None:
    # Remove the json patch cell
    patch_cell = nb['cells'].pop(json_patch_idx)
    # Insert it after imports (adjust index if needed)
    if json_patch_idx < imports_idx:
        nb['cells'].insert(imports_idx, patch_cell)
    else:
        nb['cells'].insert(imports_idx + 1, patch_cell)
    print(f"Moved json patch cell from position {json_patch_idx} to after imports at {imports_idx}")

# Save modified notebook
with open('EchoSeed_v4.2_final.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Fixed notebook saved as EchoSeed_v4.2_final.ipynb")
