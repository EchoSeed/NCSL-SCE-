#!/usr/bin/env python3
import json
import sys

# Read the notebook
with open('EchoSeed_v4.2_persistent.ipynb', 'r') as f:
    nb = json.load(f)

# Modify the first cell to remove Colab drive mounting
for i, cell in enumerate(nb['cells']):
    if 'from google.colab import drive' in ''.join(cell.get('source', [])):
        # Replace with local directory setup
        cell['source'] = [
            "# === Local log directory ===\n",
            "import os, pathlib\n",
            "LOG_DIR = \"./EchoSeed_Logs\"  # local directory\n",
            "pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)\n",
            "print(f\"🔗 Logs will be saved to {LOG_DIR}\")\n"
        ]
        break

# Save modified notebook
with open('EchoSeed_v4.2_local.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Modified notebook saved as EchoSeed_v4.2_local.ipynb")
