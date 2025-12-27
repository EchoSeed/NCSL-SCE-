#!/usr/bin/env python3
import json

# Read the notebook
with open('EchoSeed_v4.2_local.ipynb', 'r') as f:
    nb = json.load(f)

# Find and fix the json.dump patch cell
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    if '_np_default' in source and 'json.dump' in source:
        # Replace with a better patch that doesn't interfere with other libraries
        cell['source'] = [
            "# --- PATCH: make json.dump handle numpy arrays automatically ---\n",
            "import numpy as np, json\n",
            "\n",
            "# Store original for fallback\n",
            "_json_dump_original = json.dump\n",
            "_json_dumps_original = json.dumps\n",
            "\n",
            "def _np_default(o, _orig_default=None):\n",
            "    if isinstance(o, np.ndarray):\n",
            "        return o.tolist()\n",
            "    if isinstance(o, np.generic):  # e.g., np.int32, np.float64\n",
            "        return o.item()\n",
            "    # Fall back to original default if provided\n",
            "    if _orig_default is not None:\n",
            "        return _orig_default(o)\n",
            "    raise TypeError(f'{type(o).__name__} not JSON serialisable')\n",
            "\n",
            "def _json_dump_np(obj, fp, *args, default=None, **kwargs):\n",
            "    if default is None:\n",
            "        final_default = _np_default\n",
            "    else:\n",
            "        # Chain the defaults together\n",
            "        final_default = lambda o: _np_default(o, default)\n",
            "    return _json_dump_original(obj, fp, *args, default=final_default, **kwargs)\n",
            "\n",
            "# Only patch if we're not already patched\n",
            "if not hasattr(json.dump, '_np_patched'):\n",
            "    json.dump = _json_dump_np\n",
            "    json.dump._np_patched = True\n",
            "# --- END PATCH ---\n"
        ]
        print(f"Fixed cell {i}: json.dump patch")
        break

# Save modified notebook
with open('EchoSeed_v4.2_fixed.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Fixed notebook saved as EchoSeed_v4.2_fixed.ipynb")
